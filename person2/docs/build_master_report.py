"""Assemble docs/report/CONSOLIDATED_TECHNICAL_REPORT.md (and the PDF) from the generated section files.

    python person2/docs/build_master_report.py            # markdown only
    python person2/docs/build_master_report.py --pdf      # markdown + docs/report/PathSnap_Technical_Report.pdf

Sources: person2/docs/report/PERSON2_REPORT_SECTIONS.md (generated from logs), docs/report/PERSON3_REPORT_SECTIONS.md,
the final pipeline results in person2/experiments/results/, and api/logs/requests.log.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "docs" / "report"
P2R = ROOT / "person2" / "experiments" / "results"


def section(text: str, start: str, end: str | None) -> str:
    a = text.index(start)
    b = text.index(end, a + 1) if end else len(text)
    return text[a:b].strip() + "\n"


def run_metrics(run: str) -> tuple[dict, dict]:
    d = ROOT / "person2" / "experiments" / "logs" / run
    return json.loads((d / "config.json").read_text()), json.loads((d / "metrics.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()

    p2 = (ROOT / "person2" / "docs" / "report" / "PERSON2_REPORT_SECTIONS.md").read_text(encoding="utf-8")
    p3 = (REP / "PERSON3_REPORT_SECTIONS.md").read_text(encoding="utf-8")
    fin = {p: json.loads((P2R / f"final_pipeline_{p}.json").read_text()) for p in ("student_test", "evaluator_test")}
    fcsv = {p: pd.read_csv(P2R / f"final_pipeline_{p}.csv") for p in fin}
    timing = pd.read_csv(P2R / "timing.csv").set_index("label")
    _, lr = run_metrics("20261118-173204_logistic_regression")
    _, hg = run_metrics("20261118-174235_hist_gradient_boosting")
    st, ev = fin["student_test"], fin["evaluator_test"]

    out = []
    w = out.append
    w("# PathSnap: Distributed Map-Matching and Route Reconstruction\n")
    w("**Technical report** · UE24CS352A Machine Learning · Department of Computer Science & Engineering, PES University, RR Campus · November 2026\n")
    w("| Track | Owner | Report sections |\n|---|---|---|\n| Data, candidates and features | Person 1 | 2 |\n| Evaluation harness, model comparison, Bayesian optimisation | Person 2 | 3, 4 |\n| HMM/Viterbi decoding and Folium maps | Person 3 | 5, 6 |\n| API, UI, deployment and assembly | Person 4 | 7 |\n")
    w("Every number in this report comes from a logged run in the repository; the source file is named next to each table.\n")

    w("# 1. Summary\n")
    w("Noisy GPS points are matched to the road network in two stages. A classifier scores every candidate road arc near each GPS point "
      "with the probability that it is the arc the vehicle was on; a log-space Viterbi decoder over a hidden Markov model then chooses the most "
      "likely connected sequence of arcs, using road connectivity. The system runs on two laptops: a FastAPI server holds the model, the decoder and "
      "the reference routes (Laptop 1), and a Streamlit UI sends a trip ID and shows the result (Laptop 2).\n")
    w("| Final pipeline (`person2/experiments/results/final_pipeline_*.json`) | Trips | Pooled accuracy | Mean trip accuracy | Mean route continuity |\n|---|---|---|---|---|")
    for name, s in (("Student-test trips", st), ("Evaluator trips", ev)):
        w(f"| {name} | {s['n_trips']} | {s['pooled_pointwise_accuracy']:.4f} | {s['mean_trip_accuracy']:.4f} | {s['mean_route_continuity']:.4f} |")
    w("\nThe selected classifier is a tuned XGBoost on 13 features (student-test log loss 0.0549, Top-1 0.8472 before decoding). "
      "The evaluator trips were scored once, after every choice had been fixed on the training and student-test trips.\n")

    # ------------------------------------------------------------------ 2
    w("# 2. Dataset understanding and exploratory data analysis\n")
    w("## 2.1 Dataset inventory\n")
    w("Three partitions, split by whole trip: 80 training trips (fitting, cross-validation, tuning), 10 student-test trips (model comparison) and "
      "10 evaluator trips (final evaluation only). Each trip has four files: `.track` (longitude, latitude, time; about 1 Hz), `.nodes` (node coordinates), "
      "`.arcs` (directed road segments between two nodes) and `.route` (the reference sequence of arcs driven).\n")
    w("| Quantity | All 90 known trips | Train (80) | Student-test (10) |\n|---|---|---|---|\n"
      "| GPS observations | 227,577 | 208,417 | 19,160 |\n| Road network nodes | 2,024,174 | 1,784,930 | 239,244 |\n"
      "| Road network arcs | 4,061,945 | 3,581,643 | 480,302 |\n| Reference route arcs | 34,813 | 31,018 | 3,795 |\n")
    w("Source: `experiments/results/dataset_inventory.csv`, `person2/eda/outputs/dataset_inventory.csv`.\n")
    w("## 2.2 Preprocessing and data quality\n")
    w("- **Missing and invalid values.** No GPS, node or arc file has missing or out-of-range values; timestamps never decrease; every GPS point lies inside its trip's road network. No rows were removed.\n"
      "- **Coordinates.** Trips lie on several continents (longitude −100.43° to 137.25°, latitude −34.65° to 61.71°), so no single projection fits. Each trip is projected to a local metric plane centred on its own road network; distances in degrees are never used (one degree of longitude is about 111 km at the equator but about 54 km at 61° N).\n"
      "- **Sampling.** 96.7% of consecutive GPS steps are exactly 1 s apart; 47 of 90 trips contain gaps above 5 s, the largest 153,064 s. Gaps are kept and handled by the decoder (sequences split at gaps above 30 s).\n"
      "- **Road networks.** No isolated nodes; 89 of 90 networks have more than one weakly connected component, but every reference route lies inside one component. 9 trips contain 24 self-loop arcs.\n"
      "- **Reference routes.** `.route` lists each traversed arc once, so it is not aligned row by row with the GPS file. Each GPS point is assigned its reference arc by a windowed sequential projection onto the route (`features/candidate_generator.py`). In trips 00000054 and 00000033 the GPS points lie a median 139 m and 122 m from the reference route, so about half of their points have no reference arc nearby; the cause was not determined, and those points act as all-negative examples (a label limitation).\n")
    w("## 2.3 Candidates and features\n")
    w("For each GPS point, candidate arcs are the arcs within 60 m (KD-tree over arc start, middle and end points, with a 5-nearest fallback). "
      "The candidate set misses the reference arc for 8.24% of GPS points overall (2.5% on the student-test trips); no model can be right on those points. "
      "Each candidate gets 13 features, all computed from the raw files without labels:\n")
    w("- *Frozen features (Person 1):* distance to the arc, arc length, heading difference, speed, time gap.\n"
      "- *Context features (`person2/experiments/context_features.py`):* where along the arc the point projects and the distance to the arc's nearer end (separates adjacent arcs at junctions), a direction-aware heading smoothed over five GPS points (separates the two directions of a street), the mean distance of the neighbouring GPS points to the arc, and the candidate's distance relative to the other candidates of the same point (difference to the nearest, rank, number of candidates). Section 3.1.1 shows the evidence for adding them.\n")
    for fig, cap in (("eda_trajectory_sizes.png", "GPS points per trip"), ("eda_sampling_intervals.png", "Sampling intervals"),
                     ("eda_geographic_spread.png", "Geographic spread of the trips"), ("eda_road_network_properties.png", "Road network properties"),
                     ("eda_gps_to_route_error.png", "Distance from GPS points to the reference route")):
        w(f"![{cap}](../figures/{fig})\n")
    w("An interactive Folium map of a training trip (GPS points, road network, reference route) is in `person2/eda/outputs/folium_train_00000003.html`.\n")

    # ------------------------------------------------------------------ 3, 4
    w("# 3. Model experimentation and comparison\n")
    s31 = section(p2, "## 3.1 Overall model comparison", "## 3.2 Model subsections")
    w(s31)
    w("## 3.2 Model subsections\n")
    w(section(p2, "### XGBoost (gradient-boosted decision trees)", "### Gaussian Naive Bayes"))
    w(section(p2, "### Gaussian Naive Bayes", "## 3.3 Visualisations"))
    w(section(p3, "### Random Forest (run", "### Linear SVM (runs"))
    w(section(p3, "### Linear SVM (runs", None))
    lt, ht = lr["student_test"], hg["student_test"]
    w("### Logistic regression (run `20261118-173204_logistic_regression`)\n")
    w(f"**A. Model understanding.** 1. *Why suitable:* a linear model on standardised features is the simplest discriminative baseline and shows how much of the problem a weighted sum of the clues explains. "
      f"2. *Strengths and limitations:* very fast ({timing.loc['logistic_regression','fit_time_s']:.1f} s to train, {timing.loc['logistic_regression','infer_ms_per_trip']:.1f} ms per trip) and a good ranker "
      f"(AUC {lt['roc_auc']:.4f}), but it cannot model interactions, and `class_weight=\"balanced\"` inflates positive probabilities (positive precision {lt['pos_precision']:.3f} at recall {lt['pos_recall']:.3f}; ECE {lt['ece']:.3f}).\n")
    w("**B. Training and configuration.** 5. *Hyperparameters:* L2 penalty with C = 1, class_weight = balanced, max_iter = 1000, standard scaling inside the pipeline. 6. *Objective:* weighted binary cross-entropy (log loss) with an L2 penalty, minimised by L-BFGS.\n")
    w(f"**C. Experimental results.** 7–8. *Student-test:* Top-1 {lt['top1_acc']:.4f}, macro F1 {lt['macro_f1']:.4f}, log loss {lt['log_loss']:.4f}. CV macro F1 {lr['cv_macro_f1_mean']:.4f} ± {lr['cv_macro_f1_std']:.4f}. "
      f"9. *Over- or underfitting:* underfitting; CV and student-test macro F1 agree ({lr['cv_macro_f1_mean']:.4f} and {lt['macro_f1']:.4f}) and both are far below the tree models. No epoch-wise history, so no training curve.\n")
    w("### Histogram gradient boosting (run `20261118-174235_hist_gradient_boosting`)\n")
    w(f"**A. Model understanding.** 1. *Why suitable:* a second boosting implementation with histogram-binned splits, which is fast on millions of rows and checks whether XGBoost's advantage is about the algorithm family or the implementation. "
      f"2. *Strengths and limitations:* Top-1 {ht['top1_acc']:.4f}, close to XGBoost, and fast to train ({timing.loc['hist_gradient_boosting','fit_time_s']:.0f} s), but `class_weight=\"balanced\"` worsens its probabilities (log loss {ht['log_loss']:.4f}, ECE {ht['ece']:.3f}) and it is the slowest to score ({timing.loc['hist_gradient_boosting','infer_ms_per_trip']:.1f} ms per trip).\n")
    w("**B. Training and configuration.** 5. *Hyperparameters:* learning_rate = 0.05, max_iter = 150, class_weight = balanced. 6. *Objective:* binary log loss, minimised by gradient boosting on histogram-binned features.\n")
    w(f"**C. Experimental results.** 7–8. *Student-test:* Top-1 {ht['top1_acc']:.4f}, macro F1 {ht['macro_f1']:.4f}, log loss {ht['log_loss']:.4f}. CV macro F1 {hg['cv_macro_f1_mean']:.4f} ± {hg['cv_macro_f1_std']:.4f}. "
      f"9. *Over- or underfitting:* no sign of overfitting; CV and student-test macro F1 agree ({hg['cv_macro_f1_mean']:.4f} and {ht['macro_f1']:.4f}).\n")
    w("### Other models tried\n")
    w("Distance-weighted KNN (k = 15) and a shallow MLP (64-32, ReLU, early stopping) were trained by Person 1 on the 5 frozen features with a different protocol "
      "(training on a stratified 100,000-150,000-row sample). Their results are in `docs/report/SECTION_2_EDA.md` and `experiments/results/model2_knn_results.json` / `model5_mlp_results.json`. "
      "They were not re-run on the final 13-feature table, so they are not part of the comparison above.\n")
    s33 = section(p2, "## 3.3 Visualisations", "## 4. Bayesian")
    w(s33)
    for fig, cap in (("cv_macro_f1.png", "CV macro F1 per model"), ("student_test_top1_macro_f1.png", "Student-test Top-1 and macro F1"),
                     ("log_loss.png", "Log loss per model"), ("reliability.png", "Calibration curves"), ("confusion_matrices.png", "Confusion matrices"),
                     ("xgboost_learning_curve.png", "XGBoost loss curve, default settings"), ("xgboost_learning_curve_tuned.png", "XGBoost loss curve, tuned settings")):
        w(f"![{cap}](../../person2/experiments/results/{fig})\n")
    w("# 4. Bayesian hyperparameter optimisation\n")
    s4 = section(p2, "## 4. Bayesian hyperparameter optimisation (XGBoost)", "## Final end-to-end results")
    w(s4.replace("## 4. Bayesian hyperparameter optimisation (XGBoost)\n", "Model: XGBoost.\n"))
    w("![Optimisation history](../../person2/tuning/studies/xgboost_v2_cv_log_loss/optimization_history.png)\n")
    w("![Hyperparameter importance](../../person2/tuning/studies/xgboost_v2_cv_log_loss/param_importance.png)\n")

    # ------------------------------------------------------------------ 5, 6
    w(section(p3, "## 5. HMM and Viterbi route reconstruction", "## 6. Geographical").replace("## 5. HMM", "# 5. HMM").replace("\n### 5.", "\n## 5.").replace("---\n", ""))
    s6 = section(p3, "## 6. Geographical visualisation (Folium)", "## 3.2 Model subsections (Person 3)").replace("## 6. Geographical", "# 6. Geographical").replace("---\n", "")
    w(s6)
    w("![Trip 00000077, whole trip](../figures/person3_trip_00000077_full.png)\n")
    w("![Trip 00000077, zoom on a stop](../figures/person3_trip_00000077_zoom.png)\n")

    # ------------------------------------------------------------------ 7
    w("# 7. API, UI and deployment\n")
    w("## 7.1 Architecture\n")
    w("- **Laptop 1 (model server):** FastAPI (`api/main.py`) on port 8000 with `GET /health`, `GET /trips` (lists the 10 evaluator trips when they are installed) and "
      "`POST /predict/{trip_id}`. A prediction runs the full pipeline in `api/pipeline.py`: load the trip, generate candidates and the 13 features, score them with "
      "`person2/models/final/model.joblib`, decode with the tuned Viterbi decoder (`hmm/config.yaml`) and score the result against the reference route on the server.\n"
      "- **Laptop 2 (evaluator UI):** Streamlit (`ui/app.py`) on port 8501. The API address is configurable in the sidebar; the UI lists the trips, sends the chosen trip ID and shows accuracy, continuity, latency and the route map.\n"
      "- **Deployment:** `deploy/run_laptop1_api.bat|.sh` and `deploy/run_laptop2_ui.bat|.sh`; the API binds to all interfaces, CORS is enabled, and both laptops share a network (a phone hotspot as fallback; see `deploy/README.md`).\n")
    w("## 7.2 Ground truth isolation and logging\n")
    w("Reference routes stay on Laptop 1. The response contains the predicted arcs, the accuracy figures and timing; the map sent to the UI never contains the reference route "
      "(`tests/test_api.py`). Predictions are computed for every request; nothing is hard-coded. Every request is logged with trip ID, time, client, latency, accuracy and execution mode (`api/logs/requests.log`).\n")
    w("## 7.3 Results of the full pipeline\n")
    w("| Partition | Trips | Pooled accuracy | Mean trip accuracy | Trip accuracy range | Mean route continuity | Mean server latency |\n|---|---|---|---|---|---|---|")
    for name, key in (("Student-test", "student_test"), ("Evaluator", "evaluator_test")):
        s, d = fin[key], fcsv[key]
        w(f"| {name} | {s['n_trips']} | {s['pooled_pointwise_accuracy']:.4f} | {s['mean_trip_accuracy']:.4f} | {d.pointwise_accuracy.min():.3f}–{d.pointwise_accuracy.max():.3f} | "
          f"{s['mean_route_continuity']:.4f} | {d.total_latency_ms.mean():.0f} ms |")
    w("\nAll 20 trips ran in `model_plus_viterbi` mode (the trained model and the decoder). Source: `person2/experiments/evaluate_pipeline.py`, which calls the same `api/pipeline.py` as the server.\n")
    w("## 7.4 UI screenshots\n")
    w("![UI: trip list](../figures/ui_trip_list.png)\n")
    w("![UI: result for evaluator trip 00000015](../figures/ui_evaluation_result.png)\n")
    log = (ROOT / "api" / "logs" / "requests.log").read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    log = [line.replace(str(ROOT), "<repo>") for line in log]  # no local folder names in the public report
    if log:
        w("## 7.5 API request log (Laptop 1)\n")
        if (ROOT / "docs" / "figures" / "api_cmd_log.png").exists():
            w("Command window of the API server while it receives prediction requests for evaluator trips 00000015 and 00000049 "
              "(each request is logged with trip ID, client, latency, accuracy and continuity):\n")
            w("![API server command window](../figures/api_cmd_log.png)\n")
        w("Last entries of `api/logs/requests.log` from the run shown above:\n")
        w("```\n" + "\n".join(log[-4:]) + "\n```\n")

    # ------------------------------------------------------------------ 8
    w("# 8. Academic integrity and AI assistance\n")
    w("- The evaluator trips were not used for training, feature design, tuning, thresholds or model choice; they were scored once with the final pipeline. Reference labels never reach the model inputs or the UI.\n"
      "- Every reported number is produced by a script in the repository from a logged run (seeds, parameters, library versions and the feature-table hash are recorded per run).\n"
      "- AI coding assistants were used during development, mainly to help write and debug code and to structure the documentation. All experiments and results come from logged runs in this repository, and the team reviewed them.\n")

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    path = REP / "CONSOLIDATED_TECHNICAL_REPORT.md"
    path.write_text(md, encoding="utf-8")
    print("wrote", path)
    if a.pdf:
        build_pdf(md, REP / "PathSnap_Technical_Report.pdf")


def build_pdf(md: str, pdf: Path) -> None:
    import markdown
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    css = """
    body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; margin: 0; }
    h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin-top: 26px; page-break-after: avoid; }
    h2 { font-size: 13.5pt; margin-top: 18px; page-break-after: avoid; }
    h3 { font-size: 11.5pt; margin-top: 14px; page-break-after: avoid; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 8.6pt; page-break-inside: avoid; }
    th, td { border: 1px solid #bbb; padding: 3px 5px; text-align: left; vertical-align: top; }
    th { background: #eee; }
    img { max-width: 100%; max-height: 520px; display: block; margin: 8px auto 14px; page-break-inside: avoid; }
    code { font-family: Consolas, monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2px; }
    pre { background: #f3f3f3; padding: 6px; font-size: 7.5pt; white-space: pre-wrap; word-break: break-all; }
    blockquote { color: #555; border-left: 3px solid #ccc; margin: 6px 0; padding-left: 8px; }
    """
    html_path = pdf.with_suffix(".html")
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8")
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception:
            browser = p.chromium.launch(channel="msedge")
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.pdf(path=str(pdf), format="A4", margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
                 print_background=True, display_header_footer=True, header_template="<span></span>",
                 footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'>PathSnap technical report · page <span class='pageNumber'></span> of <span class='totalPages'></span></div>")
        browser.close()
    html_path.unlink()
    print("wrote", pdf)


if __name__ == "__main__":
    main()
