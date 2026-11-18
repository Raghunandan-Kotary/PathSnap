# Person 3 report sections: HMM/Viterbi (5), Folium maps (6), model subsections (3.2)

Every number below comes from a file in the repository:

- `experiments/results/person3_hmm_comparison_summary.json` and `..._per_trip.csv`, written by `hmm/tune_decoder.py`
- `hmm/tuning/grid_search.csv` and `best_params.json`
- `person2/experiments/logs/<run>/metrics.json` for the classifier runs
- `experiments/results/person3_fit_diagnostics.json`

Partitions: tuning used the 80 training trips only. The student-test trips (10) were only scored. The evaluator partition was never used.

---

## 5. HMM and Viterbi route reconstruction

### 5.1 Hidden states
At GPS observation *t*, the hidden states are that observation's candidate arcs: the rows of the frozen candidate table, all arcs within 60 m. Each state carries its own `arc_id`, so the candidate-to-road-segment mapping is preserved. The decoder returns exactly one `arc_id` per GPS observation (`hmm/decoder.py`).

### 5.2 Emission probabilities
For candidate arc *a* at observation *t*:

  log e_t(a) = w · log p_t(a) + (1 − w) · ( −½ (d_t(a) / σ)² )

- *p_t(a)* is the calibrated probability from the selected classifier (tuned XGBoost, student-test ECE 0.004). For the training trips it is the out-of-fold probability, so no trip is scored by a model that saw it.
- *d_t(a)* is the perpendicular distance in metres from the GPS point to the arc.
- σ = 7 m is the GPS-noise scale. Person 1 measured a median GPS-to-road error of 4.7 m.
- w = 0.25 is the classifier weight.

### 5.3 Transition probabilities (Newson & Krumm formulation)
From arc *a* at *t−1* to arc *b* at *t*:

- route(a, b) = o_b − o_a if *a* = *b* and the point moved forward (or back by less than 50 m of GPS jitter).
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
If arc *b* cannot be reached from *a* within 1,000 m of road (one-way conflict, a different component, or an implausible detour), the transition gets a fixed log-probability of −10 (tuned; values tried: 10, 25, 50) instead of −∞. The decoder can therefore still recover when the candidate set is wrong.

Connectivity EDA (`docs/eda/road_network_connectivity.md`) showed that 89 of 90 road networks are fragmented, but no reference route crosses fragments, so a finite penalty is enough.

### 5.7 Tuning
- **Method:** a grid over w × σ × β (36 combinations, with w > 0 so the selected model is used), then the penalty, then gap × backtrack tolerance (45 runs in total, `hmm/tuning/grid_search.csv`).
- **Objective:** pooled pointwise accuracy on the 80 training trips, using out-of-fold probabilities.
- **Probabilities:** reproduced from Person 2's logged run `20261118-151126_xgboost_tuned`. Same table, parameters and seed; CV macro F1 0.7918 vs 0.7920. The differences come from multi-threaded XGBoost.
- **Speed:** decoding takes 0.19 s per student-test trip (0.16 s prepare + 0.04 s Viterbi).

### 5.8 Did the HMM improve the route? (student-test, 10 trips, 19,160 observations)

| Method | Pointwise acc. | Route continuity | Arc-change continuity | Sequence score |
|---|---|---|---|---|
| Classifier only (argmax per point) | 0.7252 | 0.8785 | 0.6686 | 0.5504 |
| **Classifier + HMM (deployed)** | **0.8536** | **0.9696** | **0.8354** | **0.8364** |
| HMM, distance-only emissions (reference) | 0.8585 | 0.9732 | 0.8610 | 0.8442 |

On the training trips (out-of-fold probabilities), the same three rows give 0.6460, 0.7821 and 0.7871 pointwise accuracy.

- **Pointwise accuracy:** the share of GPS points matched to the reference arc. Each point's reference arc is assigned exactly as the training labels were (`match_gps_to_route_arcs`).
- **Route continuity:** the share of consecutive predictions (u, v) with u = v or end(u) = start(v).
- **Arc-change continuity:** the same, counted only where the arc changes. At 1 Hz most pairs stay on the same arc, which flatters plain continuity.
- **Sequence score:** the difflib similarity of the predicted and reference routes, each written as an ordered list of distinct arcs.

**Where it helped.** The HMM improved every one of the 10 trips: +3 to +21 points pointwise (per-trip CSV). It corrected 2,909 observations the classifier got wrong.

The main gain is direction on two-way streets. In 69% of observations the nearest distance is tied between two arcs: the two directions of the same street. The classifier must guess direction from a noisy heading, while the HMM takes it from the order of movement.

The second gain is stops. A stationary car's GPS drifts, and the classifier jumps onto side streets (Section 6 map). The HMM's continuity at arc changes rises from 0.669 to 0.835.

**Where it did not help.**
- **New errors.** The HMM broke 448 points the classifier had right, about 1 for every 6.5 it fixed. Only 4 of them were within 3 points of a segment start, so most are short stretches where the decoder held on to a plausible but wrong arc.
- **Errors neither method could fix.** 2,357 points stay wrong with or without the HMM. For 486 of them the true arc was not in the candidate set at all, so no decoder could fix them.
- **The classifier adds little on top of geometry.** Distance-only emissions are slightly better (0.8585 vs 0.8536 on student-test; 0.7871 vs 0.7821 on train). On training trips the best weight overall was w = 0 (`grid_search.csv`), and accuracy fell steadily as w rose to 1. The classifier's main extra signal is heading, which mostly duplicates what the transition model already infers from movement, and its confident errors cost more than its correct confidence gains.
- **Why the deployed decoder still uses the classifier.** We deploy w = 0.25, the best value that still uses the selected model (guideline section 8), and report the distance-only decoder for honesty.

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

Figures were generated by `maps/export_report_maps.py` for student-test trip 00000077, the trip where the HMM corrected the most points (classifier-only 0.5211 → HMM 0.7331):

- `docs/figures/person3_trip_00000077_full.png`: the whole trip.
- `docs/figures/person3_trip_00000077_zoom.png`: observations 780–869, where the HMM corrected 68 of 90 points. The car is stopped; its GPS points drift into the yard beside the road, and the classifier alone (orange) jumps onto a side street, while the HMM (red) stays on the main road along the reference (green).

The interactive HTML files are written to `maps/html/`, which is git-ignored; regenerate them with the same script.

---

## 3.2 Model subsections (Person 3)

Both models were trained through Person 2's shared harness on the frozen table: 80 trips, trip-grouped 5-fold CV, full validation folds, seed 42, and features dist_to_arc, arc_length, heading_diff, speed and time_gap.

### Random Forest (run `20261118-142520_random_forest`)
**A1. Why suitable.** The useful signal is non-linear and interacting: small distance matters only when the heading agrees, and long arcs change what "close" means. Bagged trees capture these interactions without any feature scaling.

**A2. Strengths and limitations.**
- **Strengths:** stable across folds (CV macro F1 0.7463 ± 0.0083) and a high positive recall of 0.899.
- **Limitations:** poorly calibrated probabilities. `class_weight="balanced_subsample"` inflates positive scores, giving a student-test ECE of 0.109 against 0.004 for XGBoost. That is why its probabilities were not used as HMM emissions. It is also slower at inference: 26.8 ms per trip.

**B5. Hyperparameters.** n_estimators = 100, max_depth = 12, class_weight = balanced_subsample.

**B6. Objective.** Each tree greedily minimises Gini impurity at its splits, and the forest averages the trees' votes.

**C7/C8. Student-test results.**
- Top-1: 0.7224
- Macro F1: 0.7526
- Log loss: 0.2257

**C9. Over- or underfitting?** CV macro F1 (0.7463) and student-test macro F1 (0.7526) agree, and the per-fold spread is small, so there is no sign of overfitting at depth 12.

**Route continuity.** The classifier-only figure needs this run's probability files, which Person 2 holds. Person 2 can fill it in with `python evaluation/route_metrics_cli.py person2/experiments/logs/20261118-142520_random_forest`.

### Linear SVM (runs `20261118-105309_linear_svm` and `20261118-105347_linear_svm_platt`)
**A1. Why suitable.** It is a fast, maximum-margin linear baseline from a different family (kernel-based). It tests how much of the problem a single weighted combination of the features can solve.

**A2. Strengths and limitations.**
- **Strengths:** the fastest model at 1.8 ms per trip, it trains in 3 s, and it does not overfit.
- **Limitations:**
  - It cannot represent interactions between features (for example, heading only mattering when the point is close), so it underfits: student-test Top-1 0.6404.
  - Its output is a margin, not a probability. Raw scores give log loss 0.556 and ECE 0.36. Trip-grouped Platt calibration (`--calibrate sigmoid`) brings these to 0.167 and 0.037 without changing the ranking (Top-1 is unchanged).
  - Classifier-only route continuity is 0.8022 (arc-change 0.5224), the lowest of the measured models.

**B5. Hyperparameters.** C = 1.0, class_weight = balanced, max_iter = 5,000. Features are standardised inside the pipeline.

**B6. Objective.** L2-regularised squared hinge loss (LinearSVC). The Platt step then fits a logistic regression on the out-of-fold margins.

**C7/C8. Student-test results.**

| Version | Top-1 | Macro F1 | Log loss |
|---|---|---|---|
| Raw | 0.6404 | 0.6039 | 0.5560 |
| Platt-calibrated | 0.6404 | 0.6172 | 0.1667 |

**C9. Over- or underfitting?** It underfits. In-sample macro F1 is 0.6158 against CV 0.6108, a gap of only 0.005 (`person3_fit_diagnostics.json`), and both are well below the tree models. The model is too simple for the problem, not memorising it.
