# Person 3 report sections: HMM/Viterbi (5), Folium maps (6), model subsections (3.2)

Every number below comes from a file in the repository:

- `experiments/results/person3_hmm_comparison_summary.json` and `..._per_trip.csv`, written by `hmm/tune_decoder.py`
- `hmm/tuning/grid_search.csv` and `best_params.json`
- `person2/experiments/logs/<run>/metrics.json` for the classifier runs

Partitions: tuning used the 80 training trips only. The student-test trips (10) were only scored. The evaluator partition was never used.

---

## 5. HMM and Viterbi route reconstruction

### 5.1 Hidden states
At GPS observation *t*, the hidden states are that observation's candidate arcs: the rows of the frozen candidate table, all arcs within 60 m. Each state carries its own `arc_id`, so the candidate-to-road-segment mapping is preserved. The decoder returns exactly one `arc_id` per GPS observation (`hmm/decoder.py`).

### 5.2 Emission probabilities
For candidate arc *a* at observation *t*:

  log e_t(a) = w · log p_t(a) + (1 − w) · ( −½ (d_t(a) / σ)² )

- *p_t(a)* is the probability from the selected classifier (tuned XGBoost on the 13 features of Section 3.1, student-test ECE 0.005). For the training trips it is the out-of-fold probability, so no trip is scored by a model that saw it.
- *d_t(a)* is the perpendicular distance in metres from the GPS point to the arc.
- σ = 5.5 m is the GPS-noise scale (tuned). The median distance from a GPS point to its reference route is 5.6 m on the training trips and 5.2 m on the student-test trips (section 2.2).
- w = 0.35 is the classifier weight (tuned).

### 5.3 Transition probabilities (Newson & Krumm formulation)
From arc *a* at *t−1* to arc *b* at *t*:

- route(a, b) = o_b − o_a if *a* = *b* and the point moved forward (or back by less than 25 m of GPS jitter).
- Otherwise, route(a, b) = (L_a − o_a) + sp(end(a), start(b)) + o_b.
- log T(a, b) = −| route(a, b) − gc(t−1, t) | / β, with β = 10 m.

Here *o* is the position of the GPS projection along the arc, *L* is the arc length, *sp* is the directed shortest road path (SciPy Dijkstra, cut off at 1,000 m), and *gc* is the straight-line distance between the two GPS points.

A real drive covers about as much road as the GPS moved, so the cost is close to 0. A jump to a parallel or unconnected street needs a long detour, so the cost is large. Arcs are directed: the opposite-direction twin of a two-way street only fits by "going round the block".

### 5.4 Viterbi recurrence (log space)
- V₀(s) = log e₀(s)
- V_t(s) = log e_t(s) + max_{s'} [V_{t−1}(s') + log T(s', s)]

The best path is found by backtracking the stored argmax pointers. Log space turns products of many small probabilities into sums, so they do not underflow.

### 5.5 Missing candidates and GPS gaps
- An observation with no candidates splits the sequence and is returned as −1. This never happened: every observation had at least one candidate.
- A time gap above 30 s (tuned; values tried: 10 s and 30 s) splits the trip into independent segments, because after a long dropout the car could be anywhere.
- Each segment is decoded separately.

### 5.6 Disconnected consecutive observations
If arc *b* cannot be reached from *a* within 1,000 m of road (one-way conflict, a different component, or an implausible detour), the transition gets a fixed log-probability of −25 (tuned; values tried: 10, 25, 50) instead of −∞. The decoder can therefore still recover when the candidate set is wrong.

Connectivity EDA (`docs/eda/road_network_connectivity.md`) showed that 89 of 90 road networks are fragmented, but no reference route crosses fragments, so a finite penalty is enough.

### 5.7 Tuning
- **Method:** a coarse grid over w × σ × β (36 combinations: w ∈ {0.25, 0.5, 0.75, 1.0}, σ ∈ {7, 10, 15} m, β ∈ {10, 20, 50} m; `hmm/tuning/grid_search_coarse.csv`). Its best values lay on the edges of the grid, so a refined grid followed (27 combinations: w ∈ {0.15, 0.25, 0.35}, σ ∈ {4, 5.5, 7} m, β ∈ {5, 7.5, 10} m), then the unreachable-transition penalty, then gap × backtrack tolerance (`hmm/tuning/grid_search.csv`).
- **Objective:** pooled pointwise accuracy on the 80 training trips, using out-of-fold probabilities.
- **Chosen:** w = 0.35, σ = 5.5 m, β = 10 m, penalty 25, gap 30 s, backtrack tolerance 25 m (`hmm/tuning/best_params.json`, written to `hmm/config.yaml`).
- **Probabilities:** Person 2's logged run `20261118-181912_xgboost_tuned` (the deployed model).
- **Speed:** decoding takes 0.18 s per student-test trip (0.15 s prepare + 0.04 s Viterbi).

### 5.8 Did the HMM improve the route? (student-test, 10 trips, 19,160 observations)

| Method | Pointwise acc. | Route continuity | Arc-change continuity | Sequence score |
|---|---|---|---|---|
| Classifier only (argmax per point) | 0.8472 | 0.9336 | 0.7539 | 0.7735 |
| **Classifier + HMM (deployed)** | **0.8728** | **0.9743** | **0.8640** | **0.8633** |
| HMM, distance-only emissions (reference) | 0.8599 | 0.9724 | 0.8574 | 0.8425 |

On the training trips (out-of-fold probabilities), the same three rows give 0.7727, 0.7984 and 0.7884 pointwise accuracy.

- **Pointwise accuracy:** the share of GPS points matched to the reference arc. Each point's reference arc is assigned exactly as the training labels were (`match_gps_to_route_arcs`).
- **Route continuity:** the share of consecutive predictions (u, v) with u = v or end(u) = start(v).
- **Arc-change continuity:** the same, counted only where the arc changes. At 1 Hz most pairs stay on the same arc, which flatters plain continuity.
- **Sequence score:** the difflib similarity of the predicted and reference routes, each written as an ordered list of distinct arcs.

**Where it helped.** The HMM raised pointwise accuracy on 7 of the 10 trips (by up to 8.4 points) and corrected 995 observations the classifier got wrong. Its clearest gain is continuity at arc changes, from 0.754 to 0.864: the classifier alone occasionally jumps to a neighbouring or parallel arc for a single point, and the transition model removes those jumps. Stops are a typical case: a stationary car's GPS drifts, and the classifier alone can move to the parallel road (Section 6 map).

**Where it did not help.**
- **New errors.** The HMM broke 505 points the classifier had right, about 1 for every 2 it fixed. Only 5 of them were within 3 points of a segment start, so most are short stretches where the decoder held on to a plausible but wrong arc. On 3 trips this cost slightly more than it gained (at most −1.3 points).
- **Errors neither method could fix.** 1,933 points stay wrong with or without the HMM. For 486 of them the true arc was not in the candidate set at all, so no decoder could fix them.
- **Classifier vs geometry.** The classifier adds information beyond distance: with it the HMM scores 0.8728 against 0.8599 for distance-only emissions on the student-test trips (0.7984 against 0.7884 on the training trips), and it is better on 8 of the 10 test trips.

---

## 6. Geographical visualisation (Folium)

`maps/folium_generator.py` draws toggleable layers (Leaflet layer control) with a legend:

- raw GPS points (blue dots)
- road network within 150 m of the trace (grey)
- reference route (green)
- classifier-only route (orange, report maps only)
- predicted HMM route (red, dashed)

The view is fitted to the GPS trace. Only nearby roads are drawn, because a trip's road file covers the whole region: 165,000 arcs for the largest trip.

The map the API returns to the UI never contains the reference route; this is checked in `tests/test_api.py`.

Figures were generated by `maps/export_report_maps.py` for student-test trip 00000077, the trip where the HMM changed the most points (classifier-only 0.6778 → HMM 0.7614, 475 observations corrected):

- `docs/figures/person3_trip_00000077_full.png`: the whole trip.
- `docs/figures/person3_trip_00000077_zoom.png`: observations 3758–3847, where the HMM corrected 43 of 90 points. The car is stopped and its GPS points form a cloud between two parallel roads. The classifier alone (orange) puts them on the main road, while the HMM (red) stays on the reference route (green).

The interactive HTML files are written to `maps/html/`, which is git-ignored; regenerate them with the same script.

---

## 3.2 Model subsections (Person 3)

Both models were trained through Person 2's shared harness: 80 trips, trip-grouped 5-fold CV, full validation folds, seed 42, and the 13 features of Section 3.1 (5 frozen plus 8 context features).

### Random Forest (run `20261118-173230_random_forest`)
**A1. Why suitable.** The useful signal is non-linear and interacting: a small distance matters only when the heading agrees, and where along the arc the point projects changes what "close" means. Bagged trees capture these interactions without any feature scaling.

**A2. Strengths and limitations.**
- **Strengths:** high positive recall (0.951) and a Top-1 within 0.3 points of XGBoost (0.8444 against 0.8466); classifier-only route continuity 0.9326.
- **Limitations:** less well calibrated probabilities. `class_weight="balanced_subsample"` inflates positive scores, giving a student-test ECE of 0.043 against 0.004 for XGBoost and a log loss of 0.1020 against 0.0571. That is why its probabilities were not used as HMM emissions. It is also the slowest to train (113 s) and scores in 22.3 ms per trip.

**B5. Hyperparameters.** n_estimators = 100, max_depth = 12, class_weight = balanced_subsample.

**B6. Objective.** Each tree greedily minimises Gini impurity at its splits, and the forest averages the trees' votes.

**C7/C8. Student-test results.**
- Top-1: 0.8444
- Macro F1: 0.8844
- Log loss: 0.1020

**C9. Over- or underfitting?** CV macro F1 (0.8678 ± 0.0379) and student-test macro F1 (0.8844) agree, so there is no sign of overfitting at depth 12.

### Linear SVM (runs `20261118-174412_linear_svm` and `20261118-184834_linear_svm_platt`)
**A1. Why suitable.** It is a fast, maximum-margin linear baseline from a different family (kernel-based). It tests how much of the problem a single weighted combination of the features can solve.

**A2. Strengths and limitations.**
- **Strengths:** among the fastest models (7.0 s to train on all 80 training trips, 2.2 ms per trip to score; `person2/experiments/results/timing.csv`) and it does not overfit.
- **Limitations:**
  - It cannot represent interactions between features, so it underfits: student-test Top-1 0.8176 against 0.8466 for XGBoost.
  - Its output is a margin, not a probability. Raw scores give log loss 0.3586 and ECE 0.239. Trip-grouped Platt calibration (`--calibrate sigmoid`) brings these to 0.1018 and 0.024 without changing the ranking (Top-1 is unchanged).
  - Classifier-only route continuity is 0.9132 (arc-change 0.6944).

**B5. Hyperparameters.** C = 1.0, class_weight = balanced, max_iter = 5,000. Features are standardised inside the pipeline.

**B6. Objective.** L2-regularised squared hinge loss (LinearSVC). The Platt step then fits a logistic regression on the out-of-fold margins.

**C7/C8. Student-test results.**

| Version | Top-1 | Macro F1 | Log loss |
|---|---|---|---|
| Raw | 0.8176 | 0.7202 | 0.3586 |
| Platt-calibrated | 0.8176 | 0.8514 | 0.1018 |

**C9. Over- or underfitting?** It underfits. In-sample macro F1 is 0.6976 against CV 0.6965, a gap of only 0.001, and both are well below the tree models. The model is too simple for the problem, not memorising it.
