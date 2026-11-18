# PathSnap technical report outline

Use this outline to assemble the single `PathSnap_Final_Report.pdf` once the measured results and deployment evidence are available. Follow the [submission guidelines](../guidelines/Deliverables.pdf). Keep the final analysis grounded in the team's experiments and record applicable AI assistance under the event's disclosure policy.

## 2.2 Dataset understanding and exploratory data analysis

Start from the [Sprint 1 inventory](../dataset.md) and [feature definitions](../features.md). For each analyzed partition, attach the data manifest, missing-value counts, GPS and road-network plots, and a representative Folium view. Describe feature rationale, preprocessing, label review, candidate misses and the eligible-observation denominator. Cite the exact commands and commits that generated each figure.

## 3 Model experimentation and comparison

Use the [trip-grouped protocol](../model_plan.md) and the shared comparison artifacts from #22. Include one subsection per family.

- Logistic regression and random forest from #21
- Decision tree and XGBoost from #23
- Distance-weighted KNN from #26

For every family, state suitability, strengths, limits, hyperparameters and the fitting or voting objective. Report five-fold Macro F1, student-test Top-1 and Macro F1, log loss, route continuity and measured scoring latency with their supports. Include actual cross-model plots and the evidence for any fit diagnosis. Use epoch-loss plots only for runs that logged epoch losses. Explain metrics with unavailable or inapplicable support.

## 4 Bayesian hyperparameter optimisation

Use the search space, objective, trial history and winning parameters from #22. Compare tuned and untuned runs on identical folds and student-test trips. Attach the seed, manifest version and model bundle provenance.

## 5 HMM and Viterbi route reconstruction

Use #24 for hidden states, emissions, directed transitions, missing candidates and break behavior. Compare the geometric baseline, classifier and HMM on the same eligible observations. Include Top-1, continuity, trajectory examples and sequence metrics where supported. Explain examples where decoding helped and where it failed.

## 6 Folium geographical visualisation

Use #25 for a representative export showing GPS, network, predicted parts and permitted reference layers. Add the screenshot to the PDF and link the interactive HTML or its reproduction command. Verify that route breaks remain separate and that legends and geographic bounds are readable.

## 7 API, UI and deployment

Use #19 and #27 for the working UI and Laptop 1 receiver screenshots. Record actual HTTP requests from Laptop 2, response status and timing, the model/decoder asset versions, and reproducible CLI/API/UI commands. Protect evaluator paths and references when collecting screenshots and publishing logs.

## Reproducibility and team sign-off

Record exact Git commits, versions, dependency pins, seeds, commands and per-partition supports. Document unavailable official reserved inputs and the resulting limits. Record the clean-clone check, ten student-test and ten reserved-trip runs when supplied, applicable disclosure, independent analysis and four team sign-offs.
