# Candidate classifier comparison and validation plan

This document specifies planned experiments for [issue #6](https://github.com/PES2UG24CS393/ML_Hackathon/issues/6). The parser, nearest-segment baseline, and [inference-time candidate table](features.md) are available. The provisional observation-to-arc alignment still needs validation across training trips before the first supervised fit. An ordered `.route` supplies traversal references without assigning each GPS observation an arc.

## Training data and baselines

- Identify each candidate by `(trip_id, obs_id, arc_id)` using source IDs from the [parser contract](architecture.md#shared-parser-output). Generate candidates from the GPS fixes and directed network. Use the same candidate generator, settings, feature definitions and fold assignments for all five classifiers and the nearest-segment ranking baseline. Record the distance cutoff, maximum candidate count and tie-breaking rule with each run.
- For an observation with one accepted reference arc, label that arc as positive and the remaining candidates as negative. Keep validated observations whose reference arc is outside the candidate set in observation-level evaluation, where they count as misses. Exclude them from classifier fit and conditional candidate-row metrics so every included training group has one positive. Retain empty sets and ambiguous or unavailable alignments in coverage counts.
- Start with point-to-segment distance in meters, wrapped heading difference, speed, timestamp gap and any inference-time connectivity feature validated in #5. Record missing-value rules. Use IDs as join keys and keep reference-derived quantities out of model inputs. Verify one row per candidate and at most one positive per observation. Review the alignment against representative parallel roads, repeated arcs and gaps before accepting it for supervision.

## Five model families

All models produce a score for the positive candidate class. Record raw and calibrated probability quality on held-out trips. Compare fitted model size, measured inference time and the separate cost of candidate generation.

| Classifier | Initial settings and suitability | Trade-offs and inference cost | Probabilities and decoder use |
| --- | --- | --- | --- |
| Logistic regression | L2 penalty, `C=1`, standardized numeric inputs. A fast baseline for additive effects such as distance and heading agreement. | Limited ability to capture nonlinear effects and feature interactions. One dot product per candidate, with cost linear in feature count. | Sigmoid probabilities give smooth scores. Check held-out log loss and fit a grouped calibrator if supported. Feed the positive-class probability into the common emission transform. |
| Distance-weighted KNN | Start with `k=15` and Euclidean distance on scaled inputs. Can capture local nonlinear patterns without an explicit decision boundary. | Sensitive to feature scales, sparse neighborhoods and large classes. Stores training rows, with prediction cost driven by neighbor search. | Weighted neighbor fractions can be discrete or extreme. Check calibration and clip extremes before computing log emissions. |
| Decision tree | Start with maximum depth 8 and minimum leaf size 20. Gives interpretable threshold interactions between geometry and motion features. | Small changes in the data can change the splits. One root-to-leaf traversal per candidate. | Leaf class fractions can be coarse or extreme. Use minimum leaf support and check grouped calibration before the common emission transform. |
| Random forest | Start with 200 trees and minimum leaf size 10. Averaging captures nonlinear interactions and reduces the variance of one tree. | More memory and inference work than one tree. Per-candidate cost grows with tree count and traversal depth. | Mean leaf probabilities usually give finer scores than one tree. Check held-out reliability before use as emissions. |
| XGBoost | Use binary logistic loss, 200 rounds, depth 4 and learning rate 0.05. Boosted trees can model interactions and residual errors across candidates. | Sensitive to depth, learning rate and round count. Inference traverses each tree, so constrain ensemble size and measure latency. | The logistic objective returns positive-class probabilities. Check for overconfidence on held-out trips and calibrate before the common emission transform if supported. |

Use a common fold-local median imputer with missingness indicators as applicable. Standardize continuous inputs for logistic regression and KNN inside their pipelines. Trees use the same imputed features without scaling. Keep natural candidate class frequencies for the initial comparison and record the positive fraction in every split.

## Trip-grouped protocol

1. Create and save a deterministic five-fold split across the 80 `train` trips with seed 42. A trip belongs wholly to one fold, including all its observations and candidates. Check trip disjointness, label coverage and representation of both classes. Freeze the fold assignments for the five model families and the geometric baseline.
2. Fit preprocessing and each classifier on the four training folds. Generate candidate scores on the held-out fold. Any data-dependent feature transformation, weighting choice, score calibration or threshold selection must use only training-side trips. Use inner grouped splits for those choices. For calibration, generate inner out-of-fold scores with separate preprocessing per inner split, then fit a sigmoid calibrator to those scores and refit the classifier on the outer training trips. Compare raw and calibrated log loss using training-side validation. Record when limited label or class support leads to using raw scores.
3. Rank each observation's candidates by positive-class score, breaking ties by smaller arc ID. Use the same rule for every classifier. For binary confusion matrices and F1, use one frozen decision rule across families initially, with a probability cutoff of 0.5. Log predictions and label masks per fold and report fold variability alongside pooled or trip-averaged metrics as defined below.
4. Select the leading family by **training CV observation Top-1 accuracy averaged equally across eligible trips**. This measures the actual arc selection task and prevents long trips from dominating. For ties, use lower CV log loss, then lower measured median scoring latency. Compare all five frozen fits separately on the 10 `student_test` trips after refitting on the full training partition. Freeze settings before scoring student-test and keep its results separate from CV. Keep evaluator trips and references outside development entirely.
5. After the initial comparison, run a Bayesian search on the leading family's training data using the same trip groups and primary objective. Choose model-specific ranges around the starting settings, such as log-scale `C` for logistic regression or depth, learning rate and tree count for XGBoost. Record the search space, optimizer, trial budget, seeds, parameter values, fold scores, runtimes and selected configuration. Treat the search CV score as a tuning result. Refit on all training trips and report tuned versus untuned on the same student-test trips, with probability and runtime comparisons.

## Metrics to report

| Metric | Calculation and aggregation | Validated observation labels |
| --- | --- | --- |
| CV Macro F1 | Compute binary candidate F1 for both classes and average the two equally in each held-out fold. Use candidate rows for observations with a validated reference present, count undefined class F1 as zero, and report the mean and standard deviation over five folds. Include the per-class precision, recall, support and confusion matrix. | Required |
| Observation Top-1 accuracy | For each trip, divide the number of observations whose highest-scoring arc equals the validated reference by all observations with a validated reference. Count missed candidates and unmatched outputs as incorrect. Average over trips with at least one validated observation and report the pooled correct/validated count alongside. Apply separately to CV out-of-fold and student-test predictions. | Required |
| Student-test Macro F1 | Pool candidate rows across the 10 student-test trips and use the same class averaging, eligibility and decision rule as CV. Report class support and the number of contributing trips. | Required |
| Log loss | Average negative binary log likelihood over eligible candidate rows after clipping probabilities to `[1e-6, 1-1e-6]`. Use pooled out-of-fold rows for CV and pooled student-test rows for the separate report. Report raw and calibrated values and row counts. | Required |
| Candidate miss rate | Divide the number of validated observations whose reference arc is absent from the candidate set by all validated observations. Count empty candidate sets as misses when a reference is available. Report pooled numerator and denominator and per-trip rates. This is common across models with the same candidate generator. | Required |
| Route continuity | Over adjacent input observations in each trip, count a pair as continuous if both predictions resolve to arcs and the later projected position can be reached from the earlier one in travel direction. Use directed connectivity between the first arc's end and the second arc's start, accounting for ordered movement within the same arc and directed loops if needed. Count unmatched endpoints and explicit route breaks as discontinuities. Pool continuous pairs over all adjacent pairs, and report discontinuity reasons and gaps separately. | No |
| Inference time | On the same machine and thread settings, warm up once, then run five measured passes over the same trips. Report median and p95 per-trip wall-clock time for feature-row preprocessing, scoring and ranking, plus end-to-end time with candidate generation and decoding when applicable. Report candidates and observations per trip, hardware and dependency versions. Record fit and calibration time separately. | No |

Report label coverage as validated observations divided by all observations, by split and trip. Separately report unresolved alignments, missing references and empty candidate sets. A metric with a zero denominator is unavailable and should carry its support count. Once route reconstruction is available, compare ordered reference and predicted arc sequences using edit distance divided by reference arc count per trip, averaging across trips with nonempty references. Preserve revisits and document how adjacent repeated observation matches are collapsed into a traversal. This sequence metric needs reference routes and can run without point-level alignment.

## HMM handoff

Use the candidate key and carry forward the positive-class probability `p` for each arc. Clip it using the log-loss bounds and use `log(p)` as the emission score in the initial decoder. The binary probabilities describe candidate fit independently. If normalized scores are needed, divide the clipped probabilities by their sum within an observation before taking logs. This adds a constant across candidate states at that observation, so it preserves the Viterbi choice for paths that cover the same observations. Specify a separate score for any future unmatched state.

Use directed network reachability and movement in meters over elapsed seconds for transitions. Downweight implausible gaps and split the sequence or mark observations unmatched when no feasible transition or candidate is available. Compare classifier-only and classifier-plus-HMM on the same observations using Top-1 and continuity. Report sequence results when ordered reconstructed routes are available.

Each run should save its actual command, commit, feature and label versions, split fingerprint, seeds, candidate settings, parameters, prediction keys, metric supports and timed results. The current blocker for the first supervised run is validation of the provisional observation-to-arc labels.

## Results template

Fill one row per method and evaluation split after running the experiments. Add rows for tuned and decoded variants. Record metric supports and attach the per-fold results and run metadata. A dash marks an unfilled result. Use “N/A” when a metric cannot be computed for a method.

| Method | Split | Macro F1 | Macro precision / recall | Trip-mean Top-1 | Log loss (raw / calibrated) | Candidate miss rate | Continuity | Inference ms (median / p95) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nearest segment | CV / student test | — | — | — | N/A | — | — | — |
| Logistic regression | CV / student test | — | — | — | — | — | — | — |
| Distance-weighted KNN | CV / student test | — | — | — | — | — | — | — |
| Decision tree | CV / student test | — | — | — | — | — | — | — |
| Random forest | CV / student test | — | — | — | — | — | — | — |
| XGBoost | CV / student test | — | — | — | — | — | — | — |
