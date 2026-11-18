# PathSnap Sprint 2 data readiness

**Supervised training is blocked.** The development export generated with seed
2026 contains zero independently validated observations in both partitions.
All 227,577 observation rows remain unreviewed. The student-test Top-1,
Macro F1 and Log Loss metrics are unavailable.

## Development snapshot

| Partition | Trips | Observations | Candidate rows | Observations with candidates | Empty candidate sets | Proposals whose arc is absent | Ambiguous proposals | Directed breaks (trips affected) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 80 | 208,417 | 1,359,214 | 207,074 (99.36%) | 1,343 | 23,191 (11.13%) | 149,704 | 13 (10) |
| Student-test | 10 | 19,160 | 130,963 | 19,160 (100%) | 0 | 71 (0.37%) | 17,139 | 1 (1) |
| Total | 90 | 227,577 | 1,490,177 | 226,234 | 1,343 | 23,262 | 166,843 | 14 (11) |

Every observation has an alignment proposal. A proposal whose arc is absent
can still have candidates on other arcs. Ambiguity counts come from the aligner
status and may undercount ambiguity on rows classified first as misses. The
proposal-presence percentages are mechanical estimates. Validated-reference
coverage currently has zero support.

All 450 sampled checks agreed with independently recomputed geometry and
monotonic route-step order. The [break report](../artifacts/data/reference_breaks.csv)
records directed endpoint mismatches. The builder retains each discontinuity
as review evidence.

## Review progress

Each training fold has 16 disjoint trips. Reviewed, accepted and rejected counts
are zero for every trip and fold, so the observation count is also the unreviewed
count.

| Fold | Trips | Observations | Ambiguous | Empty candidates | Proposals whose arc is absent | Validated | Fit positive / negative rows |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 24,080 | 17,684 | 2 | 89 | 0 | 0 / 0 |
| 1 | 16 | 57,210 | 46,804 | 1 | 362 | 0 | 0 / 0 |
| 2 | 16 | 58,464 | 35,461 | 1,059 | 17,608 | 0 | 0 / 0 |
| 3 | 16 | 26,055 | 16,816 | 10 | 161 | 0 | 0 / 0 |
| 4 | 16 | 42,608 | 32,939 | 271 | 4,971 | 0 | 0 / 0 |

See the [trip summary](../artifacts/data/review_summary.csv) and
[fold summary](../artifacts/data/review_summary_by_fold.csv) for complete counts.
Folds 2 and 4 account for 1,330 of the 1,343 training fixes with empty candidate
sets. All folds need independent review, including these lower-coverage folds.
The training queue has 55 quick picks, 173,107 diagnostic rows and 69 remaining
anchors. The student-test queue has 7, 17,215 and 10 respectively. The
[queue sample](evidence/review_queue_sample.csv) illustrates the reason tags.
Selecting quick picks alone would omit many ambiguous and low-coverage
conditions. Quartile anchors and diagnostic strata support a broader review,
but a queue-derived sample is still conditional on those selection rules.

## Complete independent review

Regenerate the working queue and open the review sheet for a selected observation.

```bash
python -m data_prep.build_dataset --data data --out artifacts/data --seed 2026
python -m data_prep.review_trip train 00000005 0
python -m data_prep.review_trip student_test 00000000 2104
```

Open the generated HTML under `outputs/review/`. Have an independent reviewer
inspect GPS and directed road geometry, adjacent observations and route
occurrences. For junctions and parallel roads, record why the chosen arc fits
the sequence better than competing candidates. For a repeated arc, check the
route occurrence. For a discontinuity, inspect both sides separately. For a
candidate miss, verify the reference geometry and record the cause if it can
be established. Leave unresolved rows pending. Store the reviewer account,
UTC date and evidence in a review note and link it from the CSV. Copy completed
stubs into the corresponding decision file with the original keys, proposal
IDs and trip hash. See the [review workflow](../data_prep/README.md).

```bash
python -m data_prep.build_dataset --data data --out artifacts/data --seed 2026 \
  --review-decisions artifacts/data/review_decisions.csv \
  --student-test-review-decisions artifacts/data/student_test_review_decisions.csv
python -m data_prep.check_dataset --manifest artifacts/data/manifest.json
python -m data_prep.check_dataset --require-training-ready
```

Pass each decision flag once its file is ready. All 80 training trips are below
the minimum of two candidate-present validated labels. None of the five folds
has both classes supported by at least two trips. Continue review across both
trajectory positions and challenging conditions after reaching the structural
minimum. A rejected proposal supplies no replacement label. Training candidate
misses remain outside the reviewed training label set under the current policy.
Student-test reviewers may accept evidence-backed candidate misses for Top-1
support, and conditional candidate metrics use only validated hits.

## Verification and provenance

- CPython 3.14.10 on Linux x86-64, with CPU dependencies installed in an isolated
  environment. The optional CUDA dependency set also installed and passed
  `pip check`. The NVIDIA driver was unavailable to `nvidia-smi`, so GPU
  execution was not verified.
- The regression run passed 55 tests, including the local HTTP tests. It
  excluded two estimator-fit tests and the Optuna trial fixture.
- Full development export and existing loader checked without fitting a model.
  The feature export for trip `00000005` matched all 4,025 dataset candidate rows,
  including feature values and keys. Feature and label tables and grouped folds
  had identical SHA-256 hashes across the two full verification builds. The feature radius is 100 m with up to eight
  candidates and ascending arc-ID tie-breaks. The geometric candidate and demo
  commands exercise their separate policy.
- Manifest version `sprint2-data-manifest-v1`. Feature contract `sprint2_v1`.
  Label version `sprint2-route-labels-v1`. Review protocol `proposal-bound-v2`.
- Source SHA-256 (`trip-file-sha256-v2`)
  `9c8d2948c74badc18be705e8adf280a9c593362ce5bd464835b6deac15b4962c`.
- Schema SHA-256
  `bdeef9a5bf256715e349021dd561006fe985f4c068e94fa53acd594762dd7afa`.
- Fold SHA-256
  `f84adbbf897d781ac81bf2fe60631f258e1fc0dedcb6c709e9723c5690fb5554`.
  See the [manifest](../artifacts/data/manifest.json) for artifact hashes and the
  [fold assignments](../artifacts/data/folds.json) for trip membership.

The supervised feature matrix uses the five ordered v1 inference columns.
Only rows with `label_status=validated` represent point-level truth.
Training and tuning use training folds, with preprocessing and calibration fit
on fold-local training data. Student-test labels are evaluation-only. Reserved
evaluator data was outside this export and review.
