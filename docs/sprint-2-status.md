# Sprint 2 release verification

The final report and release depend on the artifacts and checks below. Record links to the actual evidence together with the command, commit, manifest, seed and denominator where applicable. Follow the merge order in #16.

| Checkpoint | Required evidence |
| --- | --- |
| C0 | Contract, synthetic examples, ownership approval from four owners |
| C1 | Fixture passes for candidates, runner, decoder and API |
| C2 | Reviewed labels and versioned manifests with five disjoint training folds |
| C3 | Five real experiments with fold and student-test measures, plots and timings. Record the KNN CPU runtime and peak memory |
| C4 | Measured Bayesian trials, tuned/untuned comparison, portable model bundle and independent loading check |
| C5 | Directed HMM comparison, reproducible Folium maps, UI and real inference from Laptop 2 with receiver logs from Laptop 1 |
| C6 | Fresh public checkout, pinned install, regressions, ten student-test runs, reserved-trip evidence when supplied, PDF and four sign-offs |

See the [technical report outline](technical_report/outline.md) for the required sections and their evidence sources. Record official reserved-data availability before final verification and keep reserved routes and credentials server-side.

## Verification to date

- The shared interface contract and fixtures merged in PR #29.
- The CPU KNN factory in PR #30 cloned and fit a two-row synthetic fixture and returned the expected positive-class probabilities.
- The API integration in PR #31 exercised the scoring and directed decoder from PR #33 on all ten student-test inputs using a saved synthetic estimator. GPS and match lengths agreed and no demo metrics were returned. A separate development-only check exercised server-side aggregate serialization.
- Loopback HTTP fixture checks passed for health, trip listing, prediction, malformed input and unknown trip IDs. Receiver logs recorded the successful prediction and HTTP statuses.

These checks support implementation and interface verification. C2 requires the independently reviewed dataset and frozen folds from #18, plus the runner in #20 before grouped experiments can start. Imputation and scaling must fit inside each fold. The measured model bundle from #22 and merged decoder are required for real API verification. C5 and C6 remain open pending the real model, evaluator UI, two physical laptops, reserved data, report evidence and team sign-off.
