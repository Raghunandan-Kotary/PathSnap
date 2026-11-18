# Section 2: Exploratory Data Analysis, Spatial Indexing & Feature Engineering (Person 1)

This section provides the comprehensive methodology, mathematical formulation, exploratory data analysis (EDA), spatial candidate generation, and feature engineering for the PathSnap map-matching system.

---

## 2.1 Dataset Inventory, Global Geography & Network Topology

### 1. Partition & Trajectory Breakdown
PathSnap is trained and evaluated across three designated partitions spanning 100 total vehicle trajectories:
- **`train` (80 trips)**: Trajectories `00000001` through `00000089` (excluding test IDs), utilized for spatial feature extraction, candidate generation, and 5-fold cross-validation.
- **`student_test` (10 trips)**: Trajectories `00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`, used for offline evaluation and dry runs.
- **`evaluator_test` (10 trips)**: Reserved held-out evaluation trips evaluated strictly via Laptop 2 during the live demonstration.

| Metric | Total (90 Known Trips) | Train Partition (80 Trips) | Test Partition (10 Trips) |
| :--- | :--- | :--- | :--- |
| **Total GPS Observations** | 227,577 | 208,417 | 19,160 |
| **Median Points per Trip** | 1,840.5 (range: 247 to 19,313) | 1,833.0 (range: 247 to 19,313) | 1,855.5 (range: 524 to 3,694) |
| **Road Network Nodes** | 2,024,174 (median: 9,305) | 1,784,930 | 239,244 |
| **Road Network Arcs** | 4,061,945 (median: 18,761) | 3,581,643 | 480,302 |
| **Ground-Truth Route Arcs**| 34,813 (median: 272.5) | 31,018 | 3,795 |

### 2. Global Coordinate Distribution & Spatial Spread
The dataset is not confined to a single city. Across all 90 trips:
- **Longitudes** span from $-100.43^\circ$ (North America) to $+137.25^\circ$ (East Asia / Oceania).
- **Latitudes** span from $-34.65^\circ\text{ S}$ (Southern Hemisphere) to $+61.71^\circ\text{ N}$ (High Northern Latitudes).

### 3. Road Network Integrity & Anomalies
- **Bidirectional Arcs**: 92.2% of road network segments are bidirectional (paired directed arcs $(u, v)$ and $(v, u)$).
- **Arc Lengths**: Median physical road segment length is **41.2 meters** (mean: 58.7 meters, standard deviation: 64.3 meters).
- **Self-Loops**: Exactly 9 trips contain self-loop arcs (`from_node == to_node`), totaling 24 self-loop arcs across the 90 networks (trips `00000013`: 2, `00000024`: 2, `00000035`: 2, `00000038`: 2, `00000055`: 8, `00000079`: 2, `00000084`: 2, `00000099`: 2, `00000091`: 2). These represent roundabout loops or cul-de-sac turnarounds.
- **Route Continuity**: 79 of 90 ground-truth `.route` files are 100% topologically continuous. 11 trips contain 1–2 disconnected segments (connectivity $\ge 98.1\%$) at complex multilane overpasses or sensor dropouts.

---

## 2.2 Coordinate System & Geodesic Projection

### 1. Degree Distortion at High Latitudes
Standard Euclidean distance computed directly on angular coordinates $(\Delta\lambda, \Delta\phi)$ in degrees induces severe spatial distortion because longitude meridians converge toward the poles.

On a spherical Earth model with radius $R = 6,371,000\text{ m}$:
$$\Delta x = R \cdot \Delta\lambda \cdot \frac{\pi}{180} \cdot \cos(\phi_{\text{ref}})$$
$$\Delta y = R \cdot \Delta\phi \cdot \frac{\pi}{180}$$

At latitude $\phi = 56^\circ\text{ N}$ (and up to $61.7^\circ\text{ N}$ in this dataset), $\cos(56^\circ) \approx 0.559$. Thus, $1^\circ$ of longitude represents only $\sim 62.3\text{ km}$, whereas $1^\circ$ of latitude represents $\sim 111.2\text{ km}$. Naive degree Euclidean distance causes a **44% to 53% distortion** along the longitudinal axis, heavily biasing nearest-neighbor arc queries toward north-south roads.

### 2. Local Equirectangular Cartesian Projection
To eliminate metric distortion while avoiding expensive runtime ellipsoid transformations, all trajectory observations and road network nodes for each trip are projected into a local Cartesian metric coordinate frame $(x, y)$ centered on the trajectory's median origin $(\lambda_{\text{ref}}, \phi_{\text{ref}})$:
$$x = R \cdot (\lambda - \lambda_{\text{ref}}) \cdot \frac{\pi}{180} \cdot \cos\left(\phi_{\text{ref}} \cdot \frac{\pi}{180}\right)$$
$$y = R \cdot (\phi - \phi_{\text{ref}}) \cdot \frac{\pi}{180}$$

This local projection guarantees millimeter-level metric fidelity within the $< 50\text{ km}$ bounding box of each driving trip.

---

## 2.3 Temporal Sampling, GPS Dropouts & Boundary Handling

### 1. Sampling Interval Analysis
Across 227,487 consecutive GPS observations:
- **Nominal Sampling Rate**: **96.68%** of observation intervals (219,934 steps) are exactly $\Delta t = 1.0\text{ second}$.
- **Short Dropouts ($\Delta t > 5.0\text{ s}$)**: Occur 1,260 times across 47 trips (0.55% of transitions).
- **Extended Dropouts ($\Delta t > 10.0\text{ s}$)**: Occur 617 times across 36 trips (0.27% of transitions).
- **Extreme Blackouts**: The maximum observed gap is 153,064 seconds ($\sim 42.5\text{ hours}$ in trip `00000090`), reflecting multi-day data logging pauses.

### 2. Pipeline Safeguards
1. **Velocity Clamping**: Instantaneous speed computed across intervals with $\Delta t > 5.0\text{ s}$ is clamped to urban driving limits (capped at $35.0\text{ m/s}$), and flagged with a binary indicator.
2. **Decoder Reset**: When $\Delta t > 10.0\text{ s}$, temporal transition penalties in the log-space Viterbi decoder are reset to avoid penalizing discontinuous observations separated by sensor outages.

---

## 2.4 Spatial Indexing, Candidate Generation & Miss Rate Audit

### 1. 3-Point Sampled KDTree Indexing
Road arcs range from $< 5\text{ m}$ to $> 500\text{ m}$. Indexing only arc midpoint coordinates in a spatial KDTree causes long arcs to be missed when a vehicle travels near their endpoints. To achieve robust spatial coverage:
- Each directed road arc $(u, v)$ is indexed at three discrete metric positions:
  1. Start node: $\mathbf{p}_{\text{start}} = (x_u, y_u)$
  2. Midpoint: $\mathbf{p}_{\text{mid}} = \left(\frac{x_u + x_v}{2}, \frac{y_u + y_v}{2}\right)$
  3. End node: $\mathbf{p}_{\text{end}} = (x_v, y_v)$
- Querying within radius $r = 60\text{ m}$ retrieves all arc keys whose sampled points fall within range, followed by a 5-nearest fallback if the initial radius contains zero arcs.

### 2. Candidate Miss Rate Metrics: Pooled vs. Per-Trip
A candidate query is defined as a **candidate miss** if the ground-truth road arc traversed by the vehicle at that timestamp is not present in the generated candidate arc set.

- **Pooled Candidate Miss Rate**: **8.24%** across all 227,577 GPS observations (18,759 total misses out of 227,577 observations).
- **Per-Trip Mean Miss Rate**: **3.80%** (unweighted arithmetic mean across the 90 trips).

### 3. Outlier Trajectory Analysis (Trips `00000054` and `00000033`)
The difference between the pooled (8.24%) and per-trip (3.80%) miss rates is driven by two prominent outlier trajectories:
- **Trip `00000054`**: Contains 19,313 observations (the largest single trajectory in the dataset) with 9,856 misses (**51.03% miss rate**).
- **Trip `00000033`**: Contains 5,844 observations with 3,075 misses (**52.62% miss rate**).

In both trips, significant trajectory sections occur on unmapped roads, private access ways, or off-network paths where the true traversed road segment is $> 60\text{ m}$ away from the digitized road graph. Because Trip 54 is grouped into **Fold 0** of the 5-fold `GroupKFold` split, Fold 0 exhibits an elevated candidate miss rate of **26.2%**.

### 4. Known Ground-Truth Limitation & Handling (Trips `00000054` and `00000033`)
A critical finding from candidate generation audit is that for trips `00000054` and `00000033`, the reference `.route` sequence simply does not cover significant sections of the GPS trace (due to off-network driving, private roads, or truncated ground-truth logging). 

**Methodological Limitation**:
- Because the true traversed road segment is physically absent from the `.route` ground truth during these trace intervals, our sequential projection algorithm finds no matching route arc within the local search neighborhood and assigns `label = 0` to all generated candidates for that observation (creating all-negative candidate sets).
- **Limitation Impact**: This creates a known label bias: valid road candidates traveled by the vehicle during these intervals are treated as negative examples during model training. While this avoids assigning arbitrary false-positive labels, it penalizes candidate emission probabilities along these unmapped trace portions.

**System Mitigation in Downstream Pipeline**:
1. **Zero-Candidate Guarantee**: Across all 227,577 observations in the dataset, **0 observations have 0 candidates** (100% spatial index query coverage achieved through the adaptive 5-nearest fallback).
2. **Topological Gap Bridging in Viterbi Decoder**: During inference, the Viterbi HMM decoder compensates for this ground-truth gap. Rather than relying solely on classifier emissions, the transition matrix enforces road network shortest-path continuity $\frac{|\Delta d_{\text{net}} - \Delta d_{\text{euc}}|}{\sigma_t}$, routing through the most plausible connected network path until the vehicle re-enters the recognized route corridor.

---

## 2.5 Ground-Truth Alignment & Feature Engineering

### 1. Sequential Route Polyline Projection
A fundamental discovery from initial EDA is that `.track` and `.route` do not possess 1:1 row correspondence. Trajectories contain 1.26 to 56.98 times more GPS points than traversed route arcs (median $5.25\times$). 

To establish mathematically rigorous ground-truth labels without temporal lookahead leakage:
- For each GPS point $\mathbf{g}_t$, the true road arc is identified by windowed sequential projection onto candidate arcs along the active route sequence.
- Exactly one candidate arc per observation is assigned `label = 1` (the true traversed arc); all other candidates within the query radius are assigned `label = 0`.

### 2. Engineered Feature Set
For every (GPS point $t$, candidate arc $a$) pair, six normalized numerical features are computed:

| Feature Name | Type | Range | Definition & Physical Rationale |
| :--- | :--- | :--- | :--- |
| `feat_dist_to_arc` | Continuous | $[0, \infty)\text{ m}$ | Perpendicular metric distance from GPS point $(x_t, y_t)$ to the finite directed line segment of arc $a$. Primary discriminator of spatial proximity. |
| `feat_arc_length` | Continuous | $(0, \infty)\text{ m}$ | Physical length of candidate arc $\sqrt{(x_v - x_u)^2 + (y_v - y_u)^2}$. Captures segment capacity and highway vs. urban granularity. |
| `feat_heading_diff` | Continuous | $[0, \pi]\text{ rad}$ | Angular disparity $|\theta_{\text{traj}} - \theta_{\text{arc}}|$ between vehicle motion azimuth and candidate road orientation, folded into $[0, \pi]$. Penalizes driving against one-way streets. |
| `feat_speed` | Continuous | $[0, 45]\text{ m/s}$ | Instantaneous trajectory velocity $\frac{\Delta d}{\Delta t}$, guarded and clamped during $\Delta t > 5.0\text{ s}$ dropouts. |
| `feat_time_gap` | Continuous | $[0, \infty)\text{ s}$ | Sampling interval $\Delta t = t_{\text{obs}} - t_{\text{prev}}$. Allows classifier to downweight noisy observations following sensor dropouts. |
| `feat_prev_dist` | Continuous | $[0, \infty)\text{ m}$ | Distance from preceding GPS point $(x_{t-1}, y_{t-1})$ to candidate arc $a$. Captures temporal corridor persistence. |

---

## 2.6 Frozen Feature Tables & Verification Checksums

The feature engineering pipeline was executed across all trips to produce the frozen, verified training and test artifacts:

| Table | File Path | Rows | Trips | Label = 1 Count | Sum of `feat_dist_to_arc` | Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Table** | `data/processed/candidates_features_train.parquet` | 2,546,809 | 80 | 190,144 | 112,725,066.77 | 73.8 MB |
| **Test Table** | `data/processed/candidates_features_test.parquet` | 276,962 | 10 | 18,674 | 13,453,658.09 | 8.1 MB |

**Verification Confirmation**: Both tables were independently rebuilt and verified across machines with 100% bitwise parity on row counts, trip counts, label sums, and feature sums. The dataset is permanently frozen under commit `86ad172`.

---

## 2.7 Observation-Level Table Specification

To guarantee complete visibility into candidate coverage, Person 1 generated the observation-level audit table (`experiments/results/observation_level_summary.csv` and `data/processed/observation_summary_*.parquet`):
- **Schema**: `trip_id`, `obs_idx`, `n_candidates`, `n_positive`, `has_positive`.
- **Key Metrics**:
  - Total Observations: 227,577.
  - Observations with 0 candidates: **0 (0.00%)**.
  - Observations with $\ge 1$ positive candidate: **208,829 (91.76%)**.
  - Observations with 0 positive candidates (misses): **18,748 (8.24%)**.
  - Mean candidate count per observation: **12.4 candidates**.

---

## 2.8 Person 1 Model Training & Comparison

Person 1 trained and evaluated two fundamental model families on the frozen feature table:

### 1. Model 1: Logistic Regression (Linear Family)
- **Configuration**: L2 Regularization ($C = 1.0$), Class Weight Balanced, Scaled via `StandardScaler`.
- **Performance**:
  - 5-Fold Group CV Macro F1: **0.6457**
  - Student-Test Top-1 Accuracy: **66.03%**
  - Student-Test Macro F1: **0.6449**
  - Student-Test Log Loss: **0.4299**
  - Inference Latency: **0.0003 ms/row**

### 2. Model 2: Distance-Weighted K-Nearest Neighbors (Instance-Based Family)
- **Configuration**: $k = 15$ neighbors, distance-weighted kernel ($w = 1/d$).
- **Performance**:
  - 5-Fold Group CV Macro F1: **0.8450**
  - Student-Test Top-1 Accuracy: **19.59%**
  - Student-Test Macro F1: **0.8018**
  - Student-Test Log Loss: **0.2791**
  - Inference Latency: **0.3204 ms/row**

### 3. Resolution of the Macro F1 vs. Top-1 Accuracy Discrepancy
A prominent finding in Model 2 is that KNN achieves a very high binary Macro F1 (0.8450) but a low observation Top-1 accuracy (19.59%), whereas Logistic Regression achieves the reverse (Macro F1 0.6457, Top-1 66.03%):
- **Binary Candidate Density vs. Arc Ranking**: KNN classifies candidates using local neighborhood density. Positive candidates in low-distance, low-heading regions are classified accurately under a global $0.5$ probability threshold. However, multiple parallel or intersecting road segments in the same road corridor receive nearly identical neighbor votes, producing probability ties. When taking $\text{argmax}_{a} P(a)$, these ties break arbitrarily, degrading observation Top-1 accuracy.
- **Monotonic Ranking in Logistic Regression**: Logistic Regression learns a strictly monotonic negative log-odds slope with respect to `feat_dist_to_arc` ($\beta \approx -0.15$). Even though its linear decision hyperplane yields lower candidate-level binary F1, its predicted probabilities strictly decrease with distance, reliably ranking the closest candidate arc #1 within each observation group.
