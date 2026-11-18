# UE24CS352A Machine Learning: PathSnap Technical Report
## Distributed Map-Matching & Route Reconstruction
**Team Track Assembly & Master Template**

---

## 1. Executive Summary & System Overview
- **Problem Statement**: GPS trajectory errors caused by multipath interference, urban canyons, and sparse sampling intervals.
- **Solution Strategy**: Machine learning candidate classification coupled with Hidden Markov Model (HMM) log-space Viterbi decoding.
- **Deployment**: Distributed two-laptop architecture (FastAPI backend on Laptop 1, Streamlit evaluator UI on Laptop 2).
- **Team Work Split**:
  - Person 1: Data, Features & Candidates (Section 2)
  - Person 2: Model Experimentation & Bayesian Optimization (Sections 3 & 4)
  - Person 3: HMM/Viterbi & Route Evaluation (Sections 5 & 6)
  - Person 4: API, UI, Deployment & Integration (Section 7 + Assembly)

---

## 2. Dataset Understanding and Exploratory Data Analysis (Person 1)
### 2.1 Dataset Inventory & Statistics
- Trip counts across partitions (`train`: 80, `student_test`: 10, `evaluator_test`: 10).
- Observation counts, sampling frequencies (97% at 1.0 s), temporal continuity.
### 2.2 Coordinate System & Geodesic Distance Justification
- Explanation of WGS84 degree convergence at 56° N latitude.
- Mathematical formulation of local Cartesian projection vs degree Euclidean distance.
### 2.3 Candidate Generation & Miss Rate
- Spatial index radius selection (60 m).
- Candidate miss rate analysis on training trajectories.
### 2.4 Feature Engineering
- Feature descriptions: perpendicular distance, heading divergence, speed, time gap, arc length.
- Trajectory plots, road network topology visualizations, and Folium exploratory maps.

---

## 3. Model Experimentation and Comparison (Person 2 & Person 1)
### 3.1 Overall Model Comparison Table
| Model | Family | Key Configuration | CV Macro F1 | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model 1 | Linear | Logistic Regression (L2) | ... | ... | ... | ... | ... | ... ms |
| Model 2 | Instance-based | KNN (Distance-weighted) | ... | ... | ... | ... | ... | ... ms |
| Model 3 | Tree Ensemble | Random Forest (100 trees) | ... | ... | ... | ... | ... | ... ms |
| Model 4 | Boosting | LightGBM / HistGradient | ... | ... | ... | ... | ... | ... ms |
| Model 5 | Neural Network| MLP (64x32 ReLU) | ... | ... | ... | ... | ... | ... ms |

### 3.2 Individual Model Analysis (Compulsory Questions for Every Model)
- Model 1 (Logistic Regression)
- Model 2 (Distance-Weighted KNN)
- Model 3 (Random Forest)
- Model 4 (Gradient Boosting)
- Model 5 (Multi-Layer Perceptron)
*(Covering algorithm suitability, strengths/limitations, hyperparameters, loss function, test performance, overfitting/underfitting evidence)*

### 3.3 Comparative Visualizations
- CV Macro F1 comparison bar chart.
- Student-test Top-1 accuracy and Log Loss charts.
- Confusion matrices and loss curves.

---

## 4. Bayesian Hyperparameter Optimization (Person 2)
- Target Model: LightGBM / Gradient Boosting.
- Search Space and Distributions (Learning rate, max_iter, num_leaves, l2_reg).
- Objective Function: Trip-grouped 5-fold CV Macro F1 score.
- Number of Trials: 30 trials using Optuna.
- Untuned vs Tuned Model Performance Comparison Table.

---

## 5. HMM and Viterbi Route Reconstruction (Person 3)
- Hidden States: Candidate road arcs per GPS observation.
- Emission Probability Formulation: Combining classifier probability and spatial distance.
- Transition Probability Formulation: Network topological connectivity and shortest path distance vs GPS displacement.
- Edge Case Handling: Missing candidates, GPS dropouts (>5s), and disconnected segments.
- Classifier-Only vs Classifier + HMM Comparison Table and Analysis.

---

## 6. Geographical Visualizations Using Folium (Person 3)
- Multi-layer interactive maps showing:
  - Original noisy GPS points (Blue)
  - Road network topology (Gray)
  - Ground truth reference route (Green)
  - Predicted reconstructed route (Red)
- Layer controls, legends, and screenshots for selected evaluation trips.

---

## 7. API, UI, Deployment and Distributed Integration (Person 4)
*(Full content from `docs/report/SECTION_7_API_UI_DEPLOYMENT.md`)*
- Distributed Two-Laptop Architecture diagram and configuration.
- Endpoints specification (`/health`, `/trips`, `/predict/{trip_id}`).
- Server-side ground truth isolation and audit logging.
- Evaluator UI workflow and KPI metric dashboard.
- Full 10-trip dry run benchmark table (10/10 passed).
- Screenshots of UI and receiving CMD log window.

---

## 8. Academic Integrity & AI Assistance Disclosure
*(Full declaration from `docs/report/AI_ASSISTANCE_DISCLOSURE.md`)*
- Declaration of AI-assisted scaffolding tools.
- Confirmation of independent code development, training runs, and empirical verification.
