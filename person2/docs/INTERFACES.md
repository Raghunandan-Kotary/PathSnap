# PathSnap: Person 2 interfaces and hand-offs

Owner: Person 2 (shared harness, model comparison, calibration, Bayesian optimisation).
Status: **proposal for the team to confirm in writing**. Rows marked ⚠ need a reply.

## 1. What I can do without anyone else vs. what blocks me

| Work | Needs input? | State |
|---|---|---|
| EDA on 80 train + 10 student_test trips | No | Done (`eda/`) |
| Harness, metrics, logging, calibration wrapper | Only the table schema below | Done, tested on a stand-in table (`experiments/`) |
| XGBoost and Gaussian NB runners | Frozen feature table | Runners ready; **real runs wait for Person 1** |
| Candidate / label statistics, class balance (my EDA focus) | Person 1's candidate table | Waiting |
| Comparison table and plots | The other three people's logged runs | Waiting |
| Bayesian optimisation (Optuna) | Team picks the model from the comparison table | Waiting |
| Probabilities to Person 3 | Frozen table + folds | Format fixed below |

**Blocking inputs (in order):**

1. ⚠ **Person 1: frozen candidate table, observation table and fold file** with the schema in section 2. Until then nobody can report a real number. Column names are the only thing I need agreed on day one; the table can still change after, but only by re-running everyone.
2. ⚠ **Person 1: label definition.** See finding F4: "arc is anywhere on the route" gives ~3.5 positives per GPS point, which makes Top-1 lenient.
3. ⚠ **Person 3: `route_metrics(pred_df) -> {"route_continuity":…, "sequence_score":…}`**. The harness already has a hook (`route_metrics_fn`) and reports N/A until it exists. Not blocking for training.
4. ⚠ **Person 4: nothing to start.** They receive my model file and loader after the team picks the final model.
5. **Whole team:** agree the random seed (`config.GLOBAL_SEED = 42`), the metric definitions in section 4, and the log format. All three are already in code, so reply "OK" or say what to change.

## 2. Table schema (proposal for Person 1's output)

`features/candidates.parquet`, one row per (GPS point, candidate arc):

| column | type | meaning |
|---|---|---|
| `trip_id` | str, 8 digits | folder name, e.g. `00000054` |
| `obs_idx` | int | 0-based row in the trip's `.track` |
| `arc_id` | int | 0-based row in the trip's `.arcs` |
| `label` | int8 0/1 | 1 = candidate is a correct match (definition: see F4) |
| `partition` | str | `train` or `student_test` only. The evaluator partition must never appear |
| `fold` | int8 | 0..K-1 for train trips (trip-grouped, one fold per trip); −1 for student_test |
| `f_*` | float | every feature column starts with `f_`. Fit imputers/scalers on train only |

`features/observations.parquet`, one row per GPS point, **including points with zero candidates**:
`trip_id, obs_idx, partition, n_candidates, n_positive`. Without it the candidate miss rate can only be computed over points that have candidates, and the harness flags that.

`load_table()` in `experiments/harness.py` rejects: missing columns, a trip in two partitions, a train trip with two folds, any partition other than `train`/`student_test`, and NaN features.

**Row order is part of the contract.** My probability files keep the table's row order and its `(trip_id, obs_idx, arc_id)` keys unchanged, so Person 3 can join without reordering.

## 3. What I hand over

| To | What | Where |
|---|---|---|
| Everyone training | `run_experiment(spec, df, feature_cols, obs, params, …)` and `experiments/run_model.py` | `experiments/` |
| Person 3 | Out-of-fold probabilities for train trips and final-model probabilities for student_test | `experiments/logs/<run_id>/oof_proba_train.parquet`, `proba_student_test.parquet` |
| Person 3 | Same, for the tuned model | `tuning/…` (after Optuna) |
| Person 4 | `models/load.py`: `load_model()` then `model.predict_proba(feature_df)` (loader is ready and tested). `models/final/model.joblib` appears once the team selects a model and I run `models/save_final.py` | loader now; model file later |

Train-trip probabilities are **out-of-fold** (each trip scored by a model that never saw it). Person 3 should tune the decoder on these, not on in-sample probabilities.

## 4. Metric definitions (shared by every model)

* **Candidate level**: every row is a binary example. *Macro F1 / precision / recall* average over the classes {0, 1}; threshold 0.5. Also logged: positive-class P/R/F1, log loss, Brier, ECE (10 bins), ROC-AUC, confusion matrix.
* **Observation level**: the highest-probability candidate per GPS point is the prediction; ties go to the smaller `arc_id`.
  * `top1_acc` = correct ÷ points that have ≥1 candidate (headline).
  * `top1_acc_hit_only` = correct ÷ points whose candidate set contains ≥1 positive.
  * `candidate_miss_rate` = points with no positive candidate ÷ all points (zero-candidate points count as misses).
* **CV macro F1** = mean over the 5 trip-grouped folds of per-fold macro F1 (std also logged).
* **Route continuity, sequence-level score**: Person 3's definitions; N/A until supplied.
* **Timing**: `train_time_s` (final fit on all 80 trips), `infer_ms_per_1k_candidates`, `infer_ms_per_trip` (student_test; warm-up excluded).

*Primary selection metric (proposal):* CV macro F1, with CV log loss and calibration error (ECE) as the tie-break/gate, because the HMM consumes probabilities. The team decides after seeing the comparison table; the choice and reason go in report 3.1.

## 5. Findings from my EDA run that affect everyone

Counts are from `eda/outputs/dataset_inventory.csv`; the partition is stated for each.

* **F1: the trips are worldwide, so there is no single UTM zone.** The 90 trips cover latitudes −35° to 62° (Europe, Russia, the Americas, Asia, Africa). I first used UTM 32N and got `inf`/`NaN` distances for non-European trips. Use `geo.LocalProjector`, a per-trip azimuthal-equidistant projection (scale error ~2e-5), for every distance in candidate generation, features and decoding. Degree-Euclidean distance is wrong everywhere here.
* **F2: the reference route does not always cover the GPS trace.** Train trips 00000054 (11% of GPS points within 30 m of the route), 00000033 (29%), 00000002 (86%), 00000043 (87%). 10/80 train and 1/10 student_test trips have route length < 50% of GPS path length. 13/80 train trips start >100 m from the route's first arc. Labels on those points mean little. Decide a rule (e.g. exclude such trips from training/CV, chosen from **train only**) and write it in the data-quality log.
* **F3: timestamps are irregular.** Overall 96.7% of steps are exactly 1 s (train + student_test pooled, point-weighted). 45/80 train trips and 2/10 student_test trips contain a gap > 5 s. 22/80 train trips have a gap > 60 s; the largest is 153,064 s (train 00000090). 13/80 train trips have < 90% 1 s steps. Person 3 needs a gap-restart rule; speed features need a guard for large `dt`.
* **F4: label definition.** On my stand-in table (20 trips, 50 m radius, **indicative only**) "arc ∈ reference route" gives a mean of 3.5 positive arcs per GPS point; 80% of points have ≥2 positives. A random candidate scores 40% Top-1 and a nearest-arc rule 58.7%. Top-1 then mostly measures "on the route", not "the right arc at this moment". Options: (a) keep it and say so in the report, (b) align the GPS sequence monotonically to the route order and label only the aligned arc. Person 1 owns this call.
* **F5: duplicate arcs.** 53/80 train trips (1,775 duplicate `(from,to)` rows in total) and 7/10 student_test trips contain arcs that share both endpoints under different `arc_id`s. The route uses only one of them, so the twin is labelled negative though it is geometrically identical. This is irreducible label noise unless Person 1 collapses duplicates.
* **F6: route anomalies.** 10/80 train and 1/10 student_test trips have a route with a break (consecutive arcs not connected). 9/80 train and 1/10 student_test routes repeat arcs (student_test 00000056 repeats 255 of 1,323). Set-membership labels and HMM continuity must tolerate both.
* **F7: network connectivity.** No isolated nodes anywhere. Median weakly connected components: 20 (train), 11 (student_test); largest component holds 69–100% of nodes. Every reference route lies inside one weak component. Strong connectivity is lower (one-way streets), so decoder transitions must respect arc direction.
* **F8: partition sizes.** The work-split DOCX says "100 trips"; students only hold 90 (80 train + 10 student_test). The 10 evaluator trips are not in our folder and must stay untouched.

## 6. Academic-integrity rules applied to my track

* The evaluator partition is not loaded anywhere; `load_table` rejects it.
* Labels live only in the offline tables. The harness never writes labels into anything the API or UI reads.
* Every number in the report comes from `experiments/logs/runs.csv` or `tuning/`. Runs on the stand-in table are tagged `smoke=True`, live under `experiments/smoke/`, and never enter the comparison table.
* Seeds, parameters, library versions and the table's SHA-1 are logged per run.
* No predictions are hard-coded; the model is trained, saved and loaded.
* I commit my own work under my own name so repository history shows who wrote what.
