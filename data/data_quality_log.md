# PathSnap: Data Quality Log & Assumptions (Person 1)

This log records the empirical observations, coordinate system justifications, sampling characteristics, and architectural decisions made during exploratory data analysis (EDA) for **Person 1** in PathSnap.

---

## 1. Dataset Overview & Inventory

- **Partitions**:
  - `train`: 80 trips intended exclusively for feature table generation, model training, and 5-fold trip-grouped cross-validation.
  - `student_test`: 10 trips held out for student model comparison, HMM parameter tuning, and evaluation benchmarks.
  - `evaluator_test`: 10 held-out trips reserved for final assessment on Laptop 2 (never touched during development).
- **Files per Trip Directory**:
  - `<trip_id>.track`: GPS observations `(longitude, latitude, timestamp)` at ~1 Hz.
  - `<trip_id>.nodes`: Road network junction coordinates `(longitude, latitude)`. 0-based row index = `node_id`.
  - `<trip_id>.arcs`: Directed road network arcs `(from_node_id, to_node_id)`. 0-based row index = `arc_id`.
  - `<trip_id>.route`: Chronological sequence of ground-truth `arc_id` values traversed.
- **Empirical Statistics (90 trips)**:
  - Total GPS observations: 227,577 (mean 2,528.6 per trip; range 262 to 19,313).
  - Total road nodes: 2,024,174 (median 9,305 per trip; range 299 to 155,003).
  - Total road arcs: 4,061,945 (median 18,761 per trip; range 649 to 307,392).
  - Reference route arcs: 34,813 (median 272.5 per trip; range 59 to 2,266).
  - Complete inventory saved to: `experiments/results/dataset_inventory.csv`.

---

## 2. Coordinate System & Geodesic Projection Justification

- **Observation**:
  Raw coordinates across trips range from longitude $-93^\circ$ (North America) to $+106^\circ$ (Southeast Asia), with latitudes between $15^\circ\text{ N}$ and $56^\circ\text{ N}$.
- **Mathematical Limitation of Degree Euclidean Distance**:
  On an ellipsoidal/spherical earth of radius $R = 6,371,000\text{ m}$, the physical distance per degree of longitude shrinks with latitude:
  $$\Delta x = R \cdot \Delta\lambda \cdot \frac{\pi}{180} \cdot \cos(\phi)$$
  $$\Delta y = R \cdot \Delta\phi \cdot \frac{\pi}{180}$$
  At latitude $\phi = 56^\circ\text{ N}$, $\cos(56^\circ) \approx 0.5592$. Thus, $1^\circ$ longitude spans only ~62.3 km, while $1^\circ$ latitude spans ~111.2 km. Naive degree-based Euclidean distance causes a **44% spatial distortion** along the longitudinal axis, skewing nearest-arc candidate queries.
- **Implementation Decision**:
  All spatial calculations, KDTree indexing, and candidate feature engineering project angular coordinates into local Cartesian coordinates $(x, y)$ in meters centered on each trip's median origin $(\lambda_{\text{ref}}, \phi_{\text{ref}})$:
  $$x = R \cdot (\lambda - \lambda_{\text{ref}}) \cdot \frac{\pi}{180} \cdot \cos\left(\phi_{\text{ref}} \cdot \frac{\pi}{180}\right)$$
  $$y = R \cdot (\phi - \phi_{\text{ref}}) \cdot \frac{\pi}{180}$$

---

## 3. Sampling Rates, Temporal Dropouts & Blackouts

- **Nominal Sampling Rate**:
  Across 227,487 consecutive transitions, **96.68% (219,934 steps)** are exactly 1.0 second apart ($\Delta t = 1.0\text{ s}$).
- **Dropouts & Blackouts**:
  - Gaps with $\Delta t > 5.0\text{ s}$ occur in **47 out of 90 trips** (total 1,260 occurrences, 0.55% of transitions).
  - Gaps with $\Delta t > 10.0\text{ s}$ account for 617 occurrences (0.27% of transitions).
  - The maximum gap observed is 153,064 seconds (~42 hours, trip `00000090`), indicating multi-day pauses or sensor log restarts.
- **Handling Strategy in Pipeline**:
  - During GPS gaps ($\Delta t > 5.0\text{ s}$), velocity calculation is capped or marked with a gap flag (`is_large_gap`).
  - For Viterbi transition modeling (Person 3), transition penalties are reset or transitioned via shortest network path when $\Delta t > 10.0\text{ s}$ to prevent penalizing disconnected points across dropouts.

---

## 4. Ground-Truth `.route` Mapping Architecture

- **Critical Architectural Discovery**:
  Initial templates assumed a 1:1 row correspondence between `.track` and `.route`. **This is invalid:**
  - `len(.track) > len(.route)` across all 90 trips.
  - The track-to-route ratio ranges from $1.26\times$ to $56.98\times$ (median $5.25\times$).
  - `.route` represents the **topological sequence of road arcs** traversed by the vehicle, while `.track` represents 1 Hz GPS samples. Multiple consecutive GPS points reside along the same road segment.
- **Labeling Solution**:
  To generate binary labels for candidate pairs `(obs_idx, arc_id)`:
  1. Each GPS observation $(gx_i, gy_i)$ is projected onto the route arc segments.
  2. The observation is assigned to the nearest ground-truth arc in the traversed sequence.
  3. A candidate arc receives `label = 1` if it matches this assigned ground-truth arc, and `label = 0` otherwise.
- **Empirical GPS Measurement Noise**:
  - Median perpendicular distance from GPS point to ground-truth road segment is **4.7 meters**.
  - 95th percentile distance is **18.2 meters**.
  - Over 99% of GPS points fall within 45 meters of the true road centerline.
  - A candidate search radius of **60 meters** captures over 97% of true road segments while keeping candidate counts compact (average 3.9–12 candidates per observation).

---

## 5. Road Network Topology & Integrity

- **Bidirectional Arcs**:
  On average, **92.2%** of road segments are bidirectional (paired directed arcs $(u, v)$ and $(v, u)$).
- **Segment Lengths**:
  Median arc length is **41.2 meters** (mean 58.7 meters).
- **Self-Loops & Anomalies**:
  Zero self-loops (`from_node == to_node`) exist across all 90 networks.
- **Route Continuity in Ground Truth**:
  In 79 of 90 trips, the reference route is 100% topologically continuous (`arcs[route[i]].to_node == arcs[route[i+1]].from_node`).
  In 11 trips, minor 1–2 segment gaps exist in the reference route (connectivity $\ge 98.1\%$), caused by complex multi-lane junctions or sensor blackouts.

---

## 6. Generated EDA Artifacts

All generated artifacts are saved in the project repository:
1. `experiments/results/dataset_inventory.csv`: Full 90-trip tabular metadata.
2. `docs/figures/eda_sampling_intervals.png`: GPS interval distribution and max gap boxplot.
3. `docs/figures/eda_trajectory_sizes.png`: Distribution of trajectory lengths and track/route ratios.
4. `docs/figures/eda_geographic_spread.png`: Spatial distribution of trips across global coordinates.
5. `docs/figures/eda_road_network_properties.png`: Road segment length distributions.
6. `docs/figures/eda_gps_to_route_error.png`: Perpendicular GPS noise distribution.
7. `maps/html/eda_trip_00000000_map.html`: Interactive multi-layer Folium map (GPS points, road network, reference route).
8. `notebooks/01_exploratory_data_analysis.ipynb`: Comprehensive runnable Jupyter Notebook.
