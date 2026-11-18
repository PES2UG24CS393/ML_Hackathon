# Sprint 2 dataset build and independent review

Run from the repository root. The build and review tools can run on CPU with
Python alone. Use the [pinned environment](../README.md#cpu-setup) to run the
rest of the project.

```bash
python -m data_prep.build_dataset --data data --out artifacts/data --seed 2026
python -m data_prep.check_dataset --manifest artifacts/data/manifest.json
```

The build reads 80 training trips and 10 student-test trips. The raw
`.route` sequence supplies ordered proposals, and `.track` supplies the
observation keys. Candidate features use only the GPS track and directed network.
The two splits each export separate feature and label CSVs. An observation with
zero candidates still has a label/proposal row and appears in coverage totals.
The full feature, label and review-queue files are regenerated locally. The
checked-in manifest and small CSVs document the unreviewed snapshot.

## Inspect and record a decision

1. Open `review_queue.csv` and filter by split, fold, trip, and
   `independent_review=pending`. Start with one `1_quick` per trip where
   available. Cover each fold and use both trip anchors and `2_diagnostic`
   rows with different reason tags. Select examples across the trip rather
   than adjacent fixes. Any original observation ID can be reviewed.
2. Generate an offline review sheet, then open the HTML in a browser.

   ```bash
   python -m data_prep.review_trip train 00000005 0
   ```

   The sheet shows candidate geometry, the proposed segment, neighboring GPS
   and route steps, source node IDs, directed breaks, and timestamp gaps.
   Compare competing parallel roads and junctions using the GPS sequence,
   directed connections, and any independent geographic or sequence evidence
   available. Look beyond the displayed window when the route revisits an
   arc or the trail crosses a gap.
3. Edit the generated CSV stub in `outputs/review/`. Fill in `decision` as
   `accepted` or `rejected`, `reviewer` as the independent reviewer's account,
   and `review_evidence` as a link or path to a retained note. Record the UTC
   date, decision rationale, competing arcs considered and relevant sequence
   or geographic evidence in the note. Leave an uncertain observation pending
   by omitting its incomplete stub from the decision file.
4. Copy completed rows into `artifacts/data/review_decisions.csv` using
   `review_decisions_template.csv` as the header. Keep the original
   `(trip_id, obs_id)`, proposal arc, route occurrence, and input hash intact.
   Preserve leading zeroes. Maintain one row per key. Use the separate
   student-test template and decision file for that split.

```bash
python -m data_prep.build_dataset --data data --out artifacts/data --seed 2026 \
  --review-decisions artifacts/data/review_decisions.csv \
  --student-test-review-decisions artifacts/data/student_test_review_decisions.csv
python -m data_prep.check_dataset --require-training-ready
```

Pass a decision flag only when its file exists. An owner account, missing
evidence, duplicate observation, cross-partition key, stale input hash or stale
proposal produces an error. Earlier five-column CSVs need the proposal arc,
route step and source hash verified by the reviewer before resubmission. An
accepted training proposal requires candidate presence and a distance within
100 m. Student-test review may explicitly validate a candidate miss. With such
a decision the observation counts as incorrect in Top-1, and the builder leaves
its candidate rows outside the conditional Macro F1 and Log Loss support.

Only reviewers can establish the authenticity and sufficiency of evidence.
The builder checks metadata and data consistency. Rejecting an incorrect proposal
does not supply a replacement arc. Editing the raw route to resolve a proposal
changes the input hash and requires fresh review.

## Priority, support and readiness

- `1_quick` is the first unflagged proposal per trip with at least two candidates
  and the proposed arc ranked first. This is a review priority, with no effect on
  label status.
- `2_diagnostic` flags ambiguous parallel roads or junctions, repeated route arcs,
  missing reference or candidate arcs, observations adjacent to reference breaks,
  anchors on discontinuous trips, and previous GPS gaps of at least five seconds
  or zero/negative duration. Five seconds is a triage threshold relative to the
  nominal 1 Hz cadence.
- `3_anchor` contains remaining quartile anchors and rows with recorded decisions.
  These strata intentionally oversample known issues and favorable matches. The
  queue does not provide representative accuracy estimates. Compare it with the
  all-observation denominators in `review_summary.csv` and
  `review_summary_by_fold.csv`, and sample independently across trajectory
  positions and conditions when estimating review coverage.

The grouped-support gate requires each training trip to supply at least two
validated candidate-present observations, keeping each trip represented and
avoiding a one-fix trip score. Each of the five held-out folds must contain
both candidate classes across at least two trips so a single trip cannot
provide all negative examples. These are minimum structural checks for a
grouped comparison. Review enough separated observations and sample diverse
conditions before drawing statistical conclusions. The gate also checks the
frozen contract, review provenance, split isolation, geometry and route-step
order. The manifest records excluded and unresolved counts, validated
reference misses and conditional class support. Failure leaves
`training_ready: false` and triggers exit status 1 with the strict checker.

Student-test labels live only in `labels_student_test.csv`. Their review
decisions do not affect train features, train labels, grouped folds, or the
training gate. Use that partition for frozen evaluation after model selection
and tuning. Grouped CV, preprocessing fit, calibration and optimisation use
training trips exclusively. Reserved evaluator data is outside both workflows.

The feature and label interfaces retain Sprint 2 contract v1. The manifest
adds `proposal-bound-v2` review provenance, `trip-file-sha256-v2` source
fingerprinting and SHA-256 for every generated file. Run the checker against the
same development root from which the export was created. The checker verifies
both source and artifact hashes, grouped assignment and loader row counts. If a
source or output differs, regenerate from the original data and matching
review decisions.
