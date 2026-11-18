# Sprint 1 evaluation rehearsal

## Smoke test

Verified on 2026-11-18 with the bundled training trip. Run from the repository root.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
python3 -m pathsnap inspect
python3 -m pathsnap candidates "data/train/00000005" --top-k 5
python3 -m pathsnap features "data/train/00000005"
python3 -m pathsnap demo "data/train/00000005" --output outputs/trip_00000005
```

All 14 tests passed, including parsing all 90 development trips. The checked-in inventory, feature table, and alignment summary are under [evidence](evidence/). The commands regenerate those files along with candidate tables, provisional labels, and demo JSON. Folium generates new internal element IDs on each render. In headless Helium, all four vector layers toggled independently, OpenStreetMap tiles loaded, and no uncaught JavaScript errors occurred. The demo also produced GPS and network layers on a copy of the trip with no reference file and a 0 m cutoff that left every fix unmatched.

```text
Trip 00000005: 1045 GPS observations, 898 nodes, 1579 arcs
Matched within 250 m: 1045/1045 observations
Timestamps: 1 to 1045 s
Reference route: 105 steps
```

The generated `outputs/trip_00000005.json` contains 1,045 prediction rows, 117 distinct selected arcs, and a separate reference sequence of 105 steps. The reference steps describe an ordered route. Their indices have no established correspondence to GPS observation indices.

## Demo checklist

1. **Inputs, 30 seconds.** Show the sample directory. Explain that the track supplies GPS observations, nodes and directed arcs define the road network, and the optional route supplies a reference arc sequence.
2. **Run, 60 seconds.** Execute the CLI above and point out the observation, network, and match counts. If the live command is interrupted, open the checked-in sample [HTML](evidence/trip_00000005.html).
3. **Map, 60 seconds.** Show gray roads, red nearest segments, and blue GPS. Turn on the green reference route and toggle the nearest segments off and back on. Zoom into an intersection to show where the layers diverge.
4. **Interpretation, 30 seconds.** Explain that 1,045/1,045 is coverage at 250 m. A nearby parallel road or a crossing can win on distance, and the selected arcs need not form a connected route. Ground-truth alignment is still required to measure accuracy.
5. **Handoff, 30 seconds.** Show the issue board and identify label validation as the prerequisite for model comparisons.

## Owners and handoffs

All eight Sprint 1 issues were closed and carried the `sprint-1` label when checked on 2026-11-18. Times are planned elapsed minutes from the sprint kickoff. GitHub close times are recorded in IST where available.

| Issue | Owner | Planned handoff | Status and evidence |
| --- | --- | --- | --- |
| [#1 Approach and contracts](https://github.com/PES2UG24CS393/ML_Hackathon/issues/1) | Rahul B | By minute 25 | Closed at 10:55 IST. [Problem and architecture](architecture.md) describes file semantics, reference alignment, and inference. |
| [#2 Dataset inventory](https://github.com/PES2UG24CS393/ML_Hackathon/issues/2) | @Rajath-N | By minute 25 | The [dataset report](dataset.md) records the development inventory and validation findings from PR #12. |
| [#3 Parser and network](https://github.com/PES2UG24CS393/ML_Hackathon/issues/3) | @RiteshReddyy | By minute 25 | Closed at 10:40 IST. [PR #9](https://github.com/PES2UG24CS393/ML_Hackathon/pull/9) and the parser tests cover the handoff. |
| [#4 Candidates and baseline](https://github.com/PES2UG24CS393/ML_Hackathon/issues/4) | @Rajath-N | By minute 40 | PR #12 adds ranked candidate output and geometric baseline checks. See the [dataset report](dataset.md). |
| [#5 Features and reference alignment](https://github.com/PES2UG24CS393/ML_Hackathon/issues/5) | R Agilesh | By minute 50 | [PR #14](https://github.com/PES2UG24CS393/ML_Hackathon/pull/14) adds [candidate features and provisional alignment](features.md). Training labels still require validation. |
| [#6 Model comparison plan](https://github.com/PES2UG24CS393/ML_Hackathon/issues/6) | @RiteshReddyy | By minute 50 | [PR #13](https://github.com/PES2UG24CS393/ML_Hackathon/pull/13) adds the [five-model plan](model_plan.md). The comparison protocol and results template are complete. Training awaits validated labels. |
| [#7 Interactive map](https://github.com/PES2UG24CS393/ML_Hackathon/issues/7) | R Agilesh | By minute 50 | Closed at 10:56 IST. [PR #11](https://github.com/PES2UG24CS393/ML_Hackathon/pull/11) supplied the original interactive demo. The current [Folium map](evidence/trip_00000005.html) includes a separate reference layer. |
| [#8 Integration and rehearsal](https://github.com/PES2UG24CS393/ML_Hackathon/issues/8) | Rahul B | By minute 60 | Merged in [PR #15](https://github.com/PES2UG24CS393/ML_Hackathon/pull/15). The README, smoke test, and [viva guide](viva-guide.md) cover the integrated handoff. |

## Evening priorities

1. **Validate labels and candidates.** Align reference arc sequences to observations on training trips, manually inspect ambiguous cases, and measure how often the candidate set contains the validated arc.
2. **Compare five trained models.** Start with logistic regression, KNN, a decision tree, random forest, and XGBoost. Keep each trip within one fold and compare against the geometric baseline using shared candidate and label rules. Use training trips for grouped cross-validation and student-test trips for development comparisons.
3. **Run Bayesian tuning.** Tune the selected model with trip-grouped validation and compare tuned and untuned metrics and timing.
4. **Decode routes.** Use classifier scores as HMM emissions and network connectivity and travel feasibility for transitions. Use Viterbi to compare continuity and match quality against independent predictions.
5. **Serve and rehearse on two laptops.** Run FastAPI on Laptop 1, retaining evaluator references server-side. From Laptop 2, exercise the UI across all ten reserved evaluator trip IDs, confirm predictions render, and show the matching API logs on Laptop 1. Collect that evidence and the model comparisons in the consolidated PDF report.
