# Candidate features and route-label alignment

## Inference feature contract

`pathsnap.features.feature_rows(trip, candidate_radius_m=100, max_candidates=8)` uses only the parsed GPS observations and road network. It never reads `trip.route`. It returns one row for each retained `(trip_id, obs_id, arc_id)` candidate, sorted by observation, then distance, then arc ID. Candidate segments are those within 100 m, capped to the 8 closest candidates per observation. An observation with no segment in the radius has no candidate row and is recorded as a candidate miss by the CLI.

| Field | Type | Unit / meaning | Missing-value rule |
|---|---|---|---|
| `trip_id` | string | Trip directory ID | Never null |
| `obs_id` | integer | Zero-based GPS observation row | Never null |
| `arc_id` | integer | Zero-based directed arc row | Never null |
| `distance_m` | float | Point-to-segment distance in parser's local metric projection | Never null for a candidate |
| `heading_diff_deg` | float | Absolute wrapped difference, 0–180°, of incoming GPS bearing and directed arc bearing | Empty for first observation, stationary movement (<1 m), or nonpositive time delta |
| `speed_mps` | float | Displacement from previous GPS observation divided by positive timestamp delta | Empty for first observation or invalid/nonpositive delta; stationary movement is 0 |
| `time_gap_s` | float | Current timestamp minus previous timestamp | Empty for first observation or invalid/nonpositive delta |
| `predecessor_connected` | boolean | True if any retained previous-observation candidate's directed arc ends at this candidate's start node | Empty at the first observation or when previous observation has no candidates |

The connectivity proxy uses only current and previous GPS-derived candidate sets plus the network. It does not depend on a predicted previous state, route labels, or decoder output. CSV nulls are empty cells; booleans are `True`/`False`. Numeric columns are written as numeric text and should be parsed using this schema.

## Route alignment is a separate, provisional label step

The `.route` file is an ordered sequence of arc IDs, not an observation-level label array. The current alignment proposal uses dynamic programming over observation index and route-step index. It minimizes projected point-to-segment distance while requiring the assigned route-step index to be nondecreasing; a small per-step skip penalty allows route progress between sparse GPS samples. It does not affect feature extraction.

The alignment CSV is separate from `features_<trip>.csv`. Each observation is marked `accepted`, `outside_label_radius`, `reference_arc_missing_from_candidates`, `ambiguous_repeated_arc`, or `ambiguous_parallel_or_junction`. Near-tied route occurrences within 2 m are rejected rather than arbitrarily labeled. The radius gate is 100 m. Accepted means only that the heuristic found a locally supported, non-ambiguous match; it does not certify ground truth.

### Current review status

Trip `00000005` is the representative development example: 1,045 GPS observations, 105 reference route steps, 4,025 candidate rows, and zero observations without a candidate in the 100 m search radius. The monotonic alignment accepted 954 rows for review and rejected 91 as `ambiguous_parallel_or_junction`; no route arc ID repeats in this particular reference route, so the repeated-arc rejection branch was not exercised by this trip.

Manual checks sampled observations 0, 261, 522, 783, and 1,044. Their aligned route steps were 0, 26, 48, 62, and 104, with GPS-to-selected-segment distances 1.69 m, 6.62 m, 5.66 m, 7.21 m, and 21.22 m. The first four sampled arcs were also their pointwise nearest candidate. At the final sample the aligned arc 328 was 21.22 m away, while candidate 326 was slightly closer at 20.59 m; the 0.63 m difference correctly triggers ambiguity rejection. These checks support the order and geometry of the sampled alignments but do not validate every label. **Route-derived labels are not ready for model training** until the remaining accepted rows, rejected cases, and more training trips are reviewed. Never use evaluator-test route files to tune the alignment procedure.

## Reproduce

From the repository root:

```bash
python3 -m pathsnap features "data/train/00000005"
```

See the [sample feature table](evidence/features_00000005.csv). The command writes an inference-only candidate table, a separate provisional alignment CSV, and a JSON report under `outputs/`.
