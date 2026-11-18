# Sprint 1 evaluation rehearsal

## Smoke test

Verified on 2026-11-18 using Python 3.14.10 and the bundled development training trip. Run both commands from the repository root.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/demo_trip.py "Student folder/train/00000005" --output outputs/trip_00000005
```

All six parser tests passed, including the test across the bundled training and student-test trips. The CLI reproduced the checked-in HTML and JSON byte for byte. The terminal prints absolute paths for both artifacts.

In headless Helium, the repaired legend, gray roads, red predictions, and blue GPS rendered. Both layer toggles and zoom worked without JavaScript errors. OpenStreetMap tiles did not load within the browser smoke test timeout. Check the basemap on the presentation network.

```text
Trip 00000005: 1045 GPS observations, 898 nodes, 1579 arcs
Matched within 250 m: 1045/1045 observations
```

The [JSON](../outputs/trip_00000005.json) contains 1,045 prediction rows, 117 distinct selected arcs, and a separate reference sequence of 105 steps. The reference steps describe an ordered route. Their indices have no established correspondence to GPS observation indices.

## Demo checklist

1. **Inputs, 30 seconds.** Show the sample directory. Explain that the track supplies GPS observations, nodes and directed arcs define the road network, and the optional route supplies a reference arc sequence.
2. **Run, 60 seconds.** Execute the CLI above and point out the observation, network, and match counts. If the live command is interrupted, open the checked-in sample [HTML](../outputs/trip_00000005.html).
3. **Map, 60 seconds.** Show gray roads, red selected arcs, and blue GPS. Zoom into an intersection and toggle Raw GPS off and back on. The road and prediction overlay share a layer control. Check Leaflet and tile access at the venue before presenting.
4. **Interpretation, 30 seconds.** Explain that 1,045/1,045 is coverage at 250 m. A nearby parallel road or a crossing can win on distance, and the selected arcs need not form a connected route. Ground-truth alignment is still required to measure accuracy.
5. **Handoff, 30 seconds.** Show the issue board and identify label validation as the prerequisite for model comparisons.

## Owners and handoffs

GitHub snapshot from 2026-11-18. Times are planned elapsed minutes from the sprint kickoff. GitHub close times are recorded in IST where available.

| Issue | Owner | Planned handoff | Status and evidence |
| --- | --- | --- | --- |
| [#1 Approach and contracts](https://github.com/PES2UG24CS393/ML_Hackathon/issues/1) | Rahul B | By minute 25 | Closed at 10:55 IST. [Problem and architecture](problem_understanding.md) describes file semantics, reference alignment, and inference. |
| [#2 Dataset inventory](https://github.com/PES2UG24CS393/ML_Hackathon/issues/2) | @Rajath-N | By minute 25 | Open. The parser test covers bundled trips. Dataset-wide EDA findings are pending. |
| [#3 Parser and network](https://github.com/PES2UG24CS393/ML_Hackathon/issues/3) | @RiteshReddyy | By minute 25 | Closed at 10:40 IST. [PR #9](https://github.com/PES2UG24CS393/ML_Hackathon/pull/9) and the parser tests cover the handoff. |
| [#4 Candidates and baseline](https://github.com/PES2UG24CS393/ML_Hackathon/issues/4) | @Rajath-N | By minute 40 | Open. The nearest-segment baseline ran in the demo. Reusable ranked candidate output and broader acceptance checks remain pending. |
| [#5 Features and reference alignment](https://github.com/PES2UG24CS393/ML_Hackathon/issues/5) | R Agilesh | By minute 50 | Open. [The data contract](parser_baseline_contract.md) separates the reference sequence from observations. Candidate labels and alignment validation are pending. |
| [#6 Model comparison plan](https://github.com/PES2UG24CS393/ML_Hackathon/issues/6) | @RiteshReddyy | By minute 50 | Open. The [sprint plan](../PathSnap_Sprint_1_Plan.md) proposes five candidates and trip-aware metrics. Experiment results are pending. |
| [#7 Interactive map](https://github.com/PES2UG24CS393/ML_Hackathon/issues/7) | R Agilesh | By minute 50 | Closed at 10:56 IST. [PR #11](https://github.com/PES2UG24CS393/ML_Hackathon/pull/11) supplies the Leaflet demo. A separate reference layer is optional for this development rehearsal. |
| [#8 Integration and rehearsal](https://github.com/PES2UG24CS393/ML_Hackathon/issues/8) | Rahul B | By minute 60 | Open for team review. The README, smoke test, and demo checklist document the integration evidence. |

## Evening priorities

1. **Validate labels and candidates.** Finish the dataset inventory. Align reference arc sequences to observations on training trips, manually inspect ambiguous cases, and measure how often the candidate set contains the validated arc.
2. **Compare five trained models.** Start with logistic regression, KNN, a decision tree, random forest, and XGBoost. Keep each trip within one fold and compare against the geometric baseline using shared candidate and label rules. Use training trips for grouped cross-validation and student-test trips for development comparisons.
3. **Run Bayesian tuning.** Tune the selected model with trip-grouped validation and compare tuned and untuned metrics and timing.
4. **Decode routes.** Use classifier scores as HMM emissions and network connectivity and travel feasibility for transitions. Use Viterbi to compare continuity and match quality against independent predictions.
5. **Serve and rehearse on two laptops.** Run FastAPI on Laptop 1, retaining evaluator references server-side. From Laptop 2, exercise the UI across all ten reserved evaluator trip IDs, confirm predictions render, and show the matching API logs on Laptop 1.
