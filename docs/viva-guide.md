# PathSnap Sprint 1 viva guide

## Problem statement

**What problem are we solving?** PathSnap takes timestamped GPS fixes and a directed road graph and aims to recover the sequence of road segments a vehicle traversed. GPS drift can place a fix closer to a parallel street, and a sampling gap can hide a turn or several traversed segments. The Sprint 1 baseline picks the closest segment for each fix independently. The planned sequence decoder uses road direction, connectivity, and elapsed time to choose a coherent route. [Problem and architecture](problem_understanding.md)

## Data preprocessing

**What does a trip contain?** A `.track` contains longitude, latitude, and seconds. A `.nodes` file defines road vertices, and `.arcs` connects pairs of vertices in travel direction. A `.route`, when supplied, contains an ordered list of reference arc IDs. The parser preserves zero-based source indices and rejects malformed rows, invalid coordinates, decreasing timestamps, and dangling references. It preserves timed gaps, duplicate timestamps, and duplicate arcs so observations and segments retain their source IDs. A local equirectangular projection converts GPS and road coordinates to a common metric plane, scaling longitude by the cosine of the network's mid-latitude. This approximation supports local distance comparisons. [Parser contract](parser_baseline_contract.md)

**What did the inventory find?** All 80 training and 10 student-test trips parsed successfully. The training partition had 2,937 consecutive GPS gaps over two seconds, and 79 training trips contained multiple weakly connected components. The 10 reserved evaluator trips were kept outside development. [Dataset inventory](dataset.md)

## Feature engineering

**What does one training candidate look like?** The inference table has one row per `(trip_id, obs_id, arc_id)`. The extractor keeps up to eight segments within 100 m of a GPS fix and orders them by distance and arc ID.

| Feature | Why it matters |
| --- | --- |
| Point-to-segment distance (m) | Measures how closely a finite road segment fits the GPS fix. |
| Wrapped heading difference (degrees) | Compares incoming GPS motion with directed road bearing, on a 0–180° scale. |
| Speed (m/s) and time gap (s) | Describe motion and how much travel could have occurred between samples. |
| Predecessor connectivity | Checks whether any previous candidate ends where the current candidate starts. |

The first fix and nonpositive time deltas leave motion features empty. Sub-metre displacement has zero speed and undefined GPS heading. Connectivity is empty when the previous fix has no candidate set. Feature extraction uses the same GPS and network inputs at training and inference. The provisional alignment uses a separate monotonic dynamic program that associates fixes with reference route steps and flags ambiguity. Trip `00000005` has 1,045 fixes and 105 reference steps, so joining the two files by row would create false labels. [Feature contract and alignment review](features.md)

## Model selection

**Why compare these five algorithms?** Logistic regression gives a fast linear baseline. Distance-weighted KNN tests local similarity and has a higher scoring cost. A decision tree exposes interpretable thresholds. A random forest averages trees to reduce variance. XGBoost tests whether boosting captures nonlinear interactions that simpler models miss. Each family can produce a candidate score for ranking and eventual HMM emissions. [Model experiment plan](model_plan.md)

**How will we choose?** Use five trip-grouped folds on the training partition and fit any imputation, scaling, or calibration using training-side data. Choose by mean per-trip observation Top-1 accuracy and compare frozen fits on student-test trips. Also report Macro F1, precision and recall, log loss, candidate miss rate, route continuity, and latency with their denominators. Validated observation labels are required before supervised fits or accuracy claims. [Evaluation protocol](model_plan.md#trip-grouped-protocol)

## Progress report and demo

| Sprint 1 work | Evidence |
| --- | --- |
| Problem and architecture, inventory, parser and graph | All 90 permitted development trips pass parsing. [Inventory](dataset.md) |
| Geometric candidates and baseline | Trip `00000005` yields 5,225 Top-5 candidate rows and 1,045 nearest matches within 250 m. That percentage measures distance-cutoff coverage. [Baseline report](../reports/baseline/predictions.csv) |
| Candidate features and alignment prototype | The 100 m feature search produces 4,025 rows. The alignment flags 91 ambiguous fixes and sends 954 proposals for manual review. [Alignment report](../outputs/alignment_00000005.json) |
| Visualization and integration | The map shows blue GPS, gray roads, red nearest segments, and a separately selectable green reference route. [Sample map](../outputs/trip_00000005.html) |
| Five-model comparison | The experiment design is ready. Supervised training awaits validated labels. [Results template](model_plan.md#results-template) |

Open the [demo commands](../README.md#run-the-sprint-1-demo), run the sample trip, then toggle the reference and nearest segments independently. Explain why nearby roads can yield similar distances even when only one is reachable in the direction of travel.

Sprint 2 starts with label validation across training trips, then moves to five supervised comparisons, Bayesian tuning, HMM/Viterbi decoding, and the FastAPI and two-laptop evaluator UI. The final handoff also includes the consolidated technical report. [Sprint status](sprint-1.md)
