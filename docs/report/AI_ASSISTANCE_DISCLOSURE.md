# PathSnap: Academic Integrity & AI Assistance Disclosure

In compliance with the UE24CS352A Machine Learning Hackathon Guidelines and Deliverables Policy (Section 8: Academic Integrity, Plagiarism and AI Content Checks), this document outlines the declaration of AI-assisted tools utilized during project development.

---

## 1. Scope of AI Assistance
AI-assisted coding and agentic tooling were utilized in accordance with course guidelines for:
1. **Repository Layout Scaffolding**: Directory organization, `.gitignore`, and dependency specification.
2. **Boilerplate Code & Skeleton APIs**: Initial FastAPI schemas (`schemas.py`), Pydantic models, and basic Streamlit UI widgets.
3. **Documentation Structuring**: Formatting Markdown tables, reporting templates, and logging wrappers.

---

## 2. Independent Human Experimentation & Empirical Verification
In strict adherence to academic integrity rules:
- **Genuine Empirical Results**: Every metric, hyperparameter, cross-validation score, and latency figure presented in the technical report originates directly from logged code execution in this repository (`experiments/results/`, `person2/experiments/logs/`, `person2/tuning/studies/`, and `experiments/results/model_comparison_table.csv`).
- **Models Actually Evaluated**: The team experimented with multiple model families across the 80 training trips (2,546,809 candidate rows) using trip-grouped 5-fold cross-validation:
  - **Boosting**: XGBoost (tuned via Optuna Bayesian optimization over 15 trials) and HistGradientBoosting
  - **Tree Ensemble**: Random Forest (100 trees, `balanced_subsample`)
  - **Linear / Kernel**: Logistic Regression (L2, `class_weight='balanced'`) and Calibrated Linear SVM
  - **Probabilistic**: Gaussian Naive Bayes (raw and isotonic calibrated)
  - **Neural Network**: Multi-Layer Perceptron (layers 64x32, ReLU)
  - **Instance-based**: Distance-Weighted KNN ($k=15$)
- **Independent Viterbi Recurrence**: The log-space Viterbi algorithm, emission probability formulation, and road-network topological transition matrices were formulated and implemented directly in `hmm/decoder.py`.
- **Zero Fabrication**: No numbers, metrics, or curves have been fabricated. All performance comparisons, confusion matrices, and reliability curves are plotted directly from logged run artifacts.
- **Viva Readiness**: Every team member understands the complete distributed architecture, feature extraction pipelines, mathematical formulations, and code implementation, and is prepared to defend all aspects in the oral viva.
