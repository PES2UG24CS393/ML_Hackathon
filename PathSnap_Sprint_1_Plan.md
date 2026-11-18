# PathSnap — Sprint 1 Plan (First 60 Minutes)

**Project:** PathSnap: Distributed Map-Matching & Route Reconstruction  
**Course:** UE24CS352A — Machine Learning, PES University  
**Sprint:** 1 — First-phase evaluation readiness  
**Duration:** 60 minutes of team work, with parallel tasks  
**Created:** 2026-11-18  
**Status:** PLANNED — no issue or implementation is considered complete merely because it appears in this document.  
**Purpose:** Persistent project reference for future team members, Codex runs, and ChatGPT conversations.

> **Important distinction:** The supplied hackathon materials specify a first-phase evaluation before lunch but do not provide a separate first-phase grading rubric. This document is a **team-proposed sprint plan**, not an official replacement for the full hackathon requirements. All features deferred here remain required for final evaluation unless instructors say otherwise.

## 1. Problem understanding

PathSnap is a distributed machine-learning application that reconstructs realistic vehicle trajectories from noisy GPS observations by matching them to a road network. GPS readings can drift, jump between streets, or contain temporal gaps; assigning each reading to its geographically nearest road may produce an impossible, discontinuous route.

**Intended full pipeline:**

```text
Raw trip files (.track, .nodes, .arcs, .route where supplied)
        ↓
Validate, parse, clean and explore GPS + road-network data
        ↓
Generate candidate road segments for each GPS observation
        ↓
Engineer candidate/observation/sequence features
        ↓
ML candidate scoring/classification (compare ≥ 5 trained models)
        ↓
Bayesian hyperparameter optimization of chosen model
        ↓
HMM + Viterbi decoding for coherent road-segment sequence
        ↓
FastAPI inference and evaluation on Laptop 1
        ↕ network requests
Evaluator UI on Laptop 2; interactive map and metrics
```

**Sprint 1 objective:** Make a *real, demonstrable slice* of the early pipeline, from dataset files to an exploratory Folium map and—if feasible—a geometry-based nearest-road-segment baseline, while documenting candidate feature engineering and the five-model experimental plan.

**Definition of an initial baseline:** A non-ML geometric nearest-road-segment algorithm. It is a baseline for future comparison, **not** one of the five trained ML models required by the full assignment.

## 2. Dataset contract and restrictions

| File | Columns (no header, whitespace-separated) | Meaning |
|---|---|---|
| `.track` | `longitude latitude timestamp` | GPS samples; timestamp in seconds; nominally ~1 Hz |
| `.nodes` | `longitude latitude` | Road-network points; node ID is zero-based row index |
| `.arcs` | `from_node_id to_node_id` | Directed segment endpoints; arc ID is zero-based row index |
| `.route` | `arc_id` | Chronological reference sequence of traversed arcs, where supplied |

**Critical non-assumption:** A row of `.track` does **not** correspond directly to a row of `.arcs` or `.route`. A reference route is not automatically a per-observation classification label. Explicitly investigate how candidate labels can be derived, and record uncertainty rather than fabricating training labels.

| Partition | Trips | Permitted usage |
|---|---:|---|
| Training | 80 | Development, model fitting, group-aware CV and optimization |
| Student testing | 10 | Development-stage comparison/evaluation |
| Evaluator testing | 10 | Reserved final evaluation only; no model, feature, threshold or hyperparameter selection |

**Leakage prevention:** Do not split observations of the same trip between train/validation folds. Fit learned preprocessing on allowed training data only. Never hardcode evaluator outputs, train with evaluator labels, or expose evaluator ground truth to the prediction model or UI.

**Geographical math:** Longitude/latitude are angular coordinates; do not treat raw degree differences as metres. Use an appropriate coordinate projection or geodesic method and justify it. Verify coordinate order and arc endpoint validity.

## 3. First-hour deliverables and definition of done

**Minimum credible first-evaluation demo:**

1. **Dataset inspection:** Run a real file reader on at least one representative trip; print observation/node/arc counts, timestamps, missing-value/invalid-coordinate checks, and indicate whether a reference route is present.
2. **Road graph:** Construct and verify geometric arc endpoints from `.nodes` and `.arcs` with correct zero-based IDs.
3. **Folium visualization:** Produce a reproducible HTML map with actual GPS points and road-network geometry; add reference route where available and correctly decoded.
4. **Documented plan:** Record the task's inputs/outputs, candidate/label questions, initial feature list, and rationale for five proposed ML models.
5. **Reproducibility:** Commit code, documentation and relevant outputs; include basic setup/run instructions.

**Stretch goal for the first hour:** Produce nearest-segment predictions for one trip and overlay them on the Folium map. This stretch goal depends on valid coordinates, arcs and an appropriate distance function.

**Do not claim these are done without evidence:** Accuracy scores, valid supervised ground-truth association, five trained models, Bayesian tuning, HMM results, working FastAPI, cross-laptop networking, or complete final technical report.

## 4. GitHub Issues to create

Create the following **eight issues**, using the titles below. Assign actual teammates only after confirming GitHub identities; role labels are suggestions. Each issue should contain its task checklist, acceptance criteria, dependency information and links to relevant code/docs. Mark issues as `sprint-1` and add domain labels if available.

### Issue 1 — `[Sprint 1] Understand map-matching problem, data relations and architecture`
- **Priority:** P0
- **Suggested owner:** Problem/architecture lead
- **Effort:** 15–20 min
- **Dependencies:** None
- **Tasks:** Explain noisy GPS to road-segment map matching; identify inputs/outputs; specify candidate-vs-reference-label ambiguity; identify noisy observations, GPS gaps and disconnected roads; diagram ML → HMM/Viterbi → API/UI stages.
- **Acceptance:** `docs/problem_understanding.md` (or equivalent) explains the problem, trip-file semantics, anticipated inference output and unresolved assumptions. Document what is **planned** versus implemented.

### Issue 2 — `[Sprint 1] Inspect dataset and run first-pass EDA`
- **Priority:** P0
- **Suggested owner:** Data/EDA lead
- **Effort:** 20–30 min
- **Dependencies:** Access to actual dataset
- **Tasks:** Verify directory and partition structure; inspect representative raw files; count observations, nodes, arcs and available routes; assess missing/invalid coordinates, time ordering and gaps; note disconnected components where feasible; record summary and anomalies.
- **Acceptance:** Reproducible script/notebook and actual observed dataset summary; anomalies and caveats in `docs/eda_findings.md` or output table. Do not invent unverified counts.

### Issue 3 — `[Sprint 1] Implement reusable trip parsers and road-network construction`
- **Priority:** P0
- **Suggested owner:** Data-processing lead
- **Effort:** 25–35 min
- **Dependencies:** Coordinate format expectations from Issue 2
- **Tasks:** Parse whitespace-delimited `.track`, `.nodes`, `.arcs` and optional `.route`; apply column names/dtypes; validate arc endpoint bounds; preserve zero-based node/arc IDs; build indexed segment geometry; select defensible distance/projection method.
- **Acceptance:** Importable parser/functions or runnable script successfully parse one real trip, report validation checks, and output consistent segment geometry; clear error handling for missing/invalid inputs.

### Issue 4 — `[Sprint 1] Generate road-segment candidates and geometric baseline`
- **Priority:** P0 (stretch for first-hour demo if blocked)
- **Suggested owner:** Baseline lead
- **Effort:** 25–35 min
- **Dependencies:** Issue 3 geometry; Issue 2 sample trip
- **Tasks:** Enumerate nearby candidate segments; calculate point-to-segment distances in a suitable spatial metric; preserve `(trip_id, observation_index, arc_id)` correspondence; predict nearest arc for each eligible point; log no-candidate cases and runtime.
- **Acceptance:** Working candidate generator and nearest-road baseline on one real trip, with identifiable arc IDs and no reference-label leakage. No unverified accuracy claims.

### Issue 5 — `[Sprint 1] Define and prototype candidate feature engineering`
- **Priority:** P0
- **Suggested owner:** ML/features lead
- **Effort:** 20–30 min
- **Dependencies:** Issue 1 conceptual contract; Issue 3 for executable feature prototype
- **Tasks:** Justify candidate distance, temporal gap, speed, direction/heading difference and road-network connectivity features; define units, missing-value handling, train/inference parity; investigate valid training-label association with reference arc sequences.
- **Acceptance:** A feature schema/document explaining meaning and calculation. If Issues 3–4 make it possible, create a sample candidate-level feature table with IDs; flag any features/labels that remain unimplemented or unvalidated.

### Issue 6 — `[Sprint 1] Select five ML experiments and design evaluation protocol`
- **Priority:** P0
- **Suggested owner:** ML/model-selection lead
- **Effort:** 15–20 min
- **Dependencies:** Issue 1; coordinate with Issue 5
- **Tasks:** Choose and justify five *distinct* algorithms from several families; possible initial set: Logistic Regression, KNN, Decision Tree, Random Forest and XGBoost (alternatives allowed); document baseline configuration, probability-support/adaptation, training efficiency, strengths and limitations; plan group-aware CV by trip and student-test comparisons.
- **Metrics to plan:** CV Macro F1, student-test Macro F1, observation-level Top-1 Accuracy, Log Loss where applicable, Candidate Miss Rate, route continuity and inference time. Also record precision/recall and other relevant metrics.
- **Acceptance:** `docs/model_experiment_plan.md` with five proposed models, choice rationales, required data/labels, validation methodology and a table template for *future actual results*. Do not report models as trained if not run.

### Issue 7 — `[Sprint 1] Visualize real GPS, network and route layers in Folium`
- **Priority:** P0
- **Suggested owner:** Visualization lead
- **Effort:** 25–35 min
- **Dependencies:** Issues 2–3; baseline layer optionally depends on Issue 4
- **Tasks:** Draw GPS samples in blue; network arcs in neutral color; route reference (if reliably decoded) in green; baseline predictions (if ready) in red; add legend/layer controls/appropriate map zoom; save actual interactive HTML.
- **Acceptance:** Real trip Folium HTML plus generation code. Layers have distinguishable styling and no manual fabrication. If a layer is blocked, note why and still show available real layers.

### Issue 8 — `[Sprint 1] Integrate code, prepare repository and first-phase demonstration`
- **Priority:** P0
- **Suggested owner:** Integration/demo lead
- **Effort:** 20–30 min spread across sprint, final 10 min mandatory
- **Dependencies:** Integrates available outputs from Issues 1–7
- **Tasks:** Organize repository; establish install/run instructions; collect code and evidence; run happy-path demo; record unfinished work, risks and evening handoff; verify commits pushed; prepare verbal walk-through.
- **Acceptance:** Public repository (when permitted) with source and README, reproducible demo commands, clear sprint status, and no fabricated accomplishments or embedded secrets.

### Label and dependency guidance

- Useful labels: `sprint-1`, `P0`, `data`, `preprocessing`, `ml`, `feature-engineering`, `visualization`, `documentation`, `integration`.
- Do not create duplicate issues if matching issues already exist.
- Use issue numbers/links assigned by GitHub; **do not invent** issue URLs or statuses.
- Suggested critical path: **2 → 3 → 4 → 7 → 8**. Issues **1** and **6** can begin in parallel; **5** can start as a feature specification immediately and implement after data/geometry is available.

## 5. Parallel 60-minute schedule

| Time | Critical-path tasks | Parallel tasks / coordination | Minimum checkpoint |
|---|---|---|---|
| **0–10 min** | Locate real data; agree on directory layout and parsing contract | Problem framing (1); EDA begins (2); model shortlist (6); integration sets up repo (8) | Dataset paths and file semantics verified |
| **10–25 min** | Parsers and road geometry (3); EDA (2) | Feature schema (5); Folium setup (7); architecture doc (1) | One real trip parsed, GPS/network visible or near ready |
| **25–40 min** | Candidate generation and nearest-segment baseline (4) | Folium finalizes GPS/network layers (7); model plan (6); README (8) | At least a functioning raw-data map; baseline if feasible |
| **40–50 min** | Integrate baseline overlay and optional reference route (7) | Review feature contract and model comparison plan (5, 6) | Stable demo path, no invented results |
| **50–60 min** | **Freeze scope, run demo, push GitHub (8)** | Rehearse speaking points; log blockers and evening tasks | Repository, map/output and explanation ready |

**Time estimates are individual person-minutes, not a sequence of 8 tasks within 60 elapsed minutes.** Assign tasks in parallel. If there are fewer teammates, concentrate on Issues 2, 3, 7 and 8; Issue 4 becomes a stretch goal. Maintain at least a documented plan for Issues 1, 5 and 6.

## 6. Files and code organization (suggested, adapt to actual repo)

```text
PathSnap/
├── README.md
├── requirements.txt
├── docs/
│   ├── problem_understanding.md
│   ├── eda_findings.md
│   ├── feature_spec.md
│   ├── model_experiment_plan.md
│   └── sprint_1_status.md
├── src/
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── candidates.py
│   ├── features.py
│   └── visualization.py
├── notebooks/
│   └── initial_eda.ipynb
├── scripts/
│   └── demo_trip.py
└── outputs/
    └── sample_trip_map.html
```

Store original data according to repository constraints and dataset permissions; do not indiscriminately push large/private raw datasets. Keep generated outputs reproducible from code. Exact structure may be adjusted after inspecting existing repo conventions.

## 7. Demo script for the first evaluation (only claim what is working)

> “PathSnap addresses GPS map matching: noisy GPS points need to be associated with plausible road segments and ultimately reconstructed into a coherent route. We inspected the trip files (`.track`, `.nodes`, `.arcs`, and reference `.route` when present), checked initial data quality, and created a geographic visualization of a real trip. [If true: We also implemented a nearest-road-segment geometric baseline and displayed its predictions.] We know that nearest-segment matching can fail on parallel or disconnected roads; our next stage is candidate-level feature engineering, training and comparing at least five models, Bayesian tuning, and HMM/Viterbi decoding. The completed system will be served by FastAPI on Laptop 1 and demonstrated from a UI on Laptop 2.”

**Have ready:** one executed notebook/script, one HTML map, actual summary statistics, a GitHub repo, and the issue board. Explain known limitations—especially the `.route` versus per-GPS-label mismatch—rather than hiding them.

## 8. Deferred Sprint 2 / final-evaluation work

The following remain **mandatory for the overall hackathon**, even though Sprint 1 may only plan them:

1. Robust candidate generation/feature-label association and full data-quality handling across trips.
2. **Train, evaluate and document at least five distinct ML models** with genuine results; use training trips for group-aware CV and student-testing trips for development comparisons.
3. Select a model using evidence (not training accuracy alone); run **Bayesian hyperparameter optimization** and compare tuned versus untuned results.
4. Implement HMM/Viterbi with documented hidden states, emission/transition probabilities, disconnected-road and GPS-gap handling. Compare classifier-only versus classifier-plus-HMM pointwise and sequence metrics.
5. Produce Folium views with raw GPS, road network, reference route where available and predicted route; export a representative view for the report.
6. Deploy **FastAPI on Laptop 1** with model, road network and server-side reference labels; UI on **Laptop 2** lists all 10 reserved evaluation trips, calls the API by trip ID and displays output. Show API logs on receiving laptop. System must work for every evaluator trip; keep ground truth out of the model and UI.
7. Submit a **public GitHub repository** with full code, saved artifacts/configurations, logs, experiments, README and deployment instructions, plus **one consolidated PDF technical report** with results, graphs, maps, UI screenshots and server-side API response evidence.

## 9. Open questions / decisions to record

The following are **not established facts** until the team checks them:

- Where are the actual dataset files and which `.route` files are available in each partition?
- What are the network size, connectivity, GPS-coordinate ranges and observed data-quality issues?
- How can the arc-sequence reference be associated with GPS observations for supervised candidate labels, and what uncertainty remains?
- Which projection/distance metric and candidate-search radius or top-K policy will be used?
- Which five models are feasible given dataset size, label construction, computational cost and probability needs?
- What are the exact source paths/entrypoints, repo URL, issue numbers, owners and live progress?
- Have the evaluators announced any additional first-phase criteria beyond the written documents?

Record decisions in `docs/sprint_1_status.md`; avoid retroactively treating assumptions as established experimental findings.

## 10. Cross-chat continuity and update convention

When this file is supplied in a future chat, treat it as the *Sprint 1 proposed plan*. Do not assume checklist items are completed. Ask for or inspect the current GitHub issues, commits and `docs/sprint_1_status.md` to determine actual progress. Keep the original official documents authoritative for mandatory requirements. Amend the plan with dated change notes rather than rewriting history.

**Suggested status format for each issue:**

```markdown
- Issue: #<real GitHub issue number> — <title>
- Status: Not started | In progress | Blocked | Complete
- Owner: <confirmed teammate>
- Evidence: <real file path, commit or artifact>
- Blocker/next action: <specific detail>
```

### Source documents (provided with the project)

1. **Main Guidelines.pdf** — `PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION`, problem and two-laptop architecture, expected stages.
2. **Dataset Exploration.pdf** — raw trip-file specifications, EDA expectations, 80/10/10 partitions, leakage and geographical considerations.
3. **Models and Training Guidelines.pdf** — minimum five trained models, CV/metrics, Bayesian tuning, HMM/Viterbi, final evaluation behavior.
4. **Deliverables.pdf** — GitHub repository and technical report requirements, report sections, visuals, API/UI evidence and first-phase mention.

*This plan translates the official requirements into a proposed one-hour sprint. All specific issue titles, owners, time allocations, folder names and milestones here are planning choices, not requirements stated verbatim in the source PDFs.*

### Repository layout update (2026-11-18)

The developer entry point is `python main.py`. Development inputs live in `data/`. The assignment PDFs live in `docs/guidelines/`, representative evidence lives in `docs/evidence/`, and regenerated files go to `outputs/`. See the [current README](README.md) for commands and [Sprint 1 status](docs/sprint-1.md) for acceptance evidence.
