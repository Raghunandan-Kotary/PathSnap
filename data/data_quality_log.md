# PathSnap: Data Quality Log & Assumptions (Person 1)

This log records observations, data anomalies, coordinate system justifications, and handling decisions made during exploratory data analysis (EDA).

---

## 1. Dataset Overview & Inventory
- **Partitions**:
  - `train`: 80 trips intended for training and cross-validation.
  - `student_test`: 10 trips reserved for student model comparison and HMM tuning.
  - `evaluator_test`: 10 held-out trips used exclusively during final evaluation.
- **File Structure per Trip**:
  - `<trip_id>.track`: GPS observations `(longitude, latitude, timestamp)`.
  - `<trip_id>.nodes`: Road network junction coordinates `(longitude, latitude)`. Row index = `node_id`.
  - `<trip_id>.arcs`: Directed road network segments `(from_node_id, to_node_id)`. Row index = `arc_id`.
  - `<trip_id>.route`: Chronological sequence of ground-truth `arc_id` values traversed.

---

## 2. Coordinate System & Distance Justification
- **Observation**: Coordinates are provided in WGS84 degrees `(longitude, latitude)` around longitude $\approx 54^\circ - 55^\circ$ E, latitude $\approx 56^\circ$ N.
- **Decision**: Euclidean distance directly on degree coordinates is strictly invalid due to latitude convergence ($\cos(56^\circ) \approx 0.559$, meaning 1 degree of longitude is only ~56% the distance of 1 degree of latitude).
- **Implementation**: All spatial indexing, candidate search radii, and perpendicular distance features project coordinates to meters using local Equirectangular projection centered on the trip midpoint:
  $$x = R \cdot \Delta\lambda \cdot \cos(\phi_{\text{ref}})$$
  $$y = R \cdot \Delta\phi$$
  where $R = 6,371,000 \text{ m}$.

---

## 3. Sampling Rates & Temporal Continuity
- **Observation**: Over 97% of consecutive GPS timestamps in `.track` have a 1.0-second interval ($\Delta t = 1.0 \text{ s}$).
- **Anomalies**: Periodic GPS dropouts and sensor blackouts produce gaps of $\Delta t > 5.0 \text{ s}$.
- **Handling**: High $\Delta t$ breaks sequence continuity in Viterbi transition modeling; transition penalties are reset or transitioned via shortest path during gaps exceeding 10 seconds.

---

## 4. Archive & Extraction Notes
- Note: In `Student folder.zip`, trip `00000086` exhibited an uncompressed CRC size mismatch in its raw archive header. 71/80 train trips and 10/10 student_test trips extracted with full file integrity.
- Candidate miss rate tracking will be logged for every trip in `experiments/results/`.
