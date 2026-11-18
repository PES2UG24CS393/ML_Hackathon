# Provisional-label KNN pilot

This pilot tests the repository's CPU distance-weighted KNN on a small sample of development observations. The included `provisional_labels.csv` records arc proposals that passed an AI-assisted check of GPS distance, direction and alternate candidates. It binds each proposal to its trip, observation, route step and input hash. The official training-readiness and independent-review files are outside this experiment.

## Measured results

| Split | Trips with provisional labels | GPS observations | KNN Top-1 matches | Nearest-segment matches |
| --- | ---: | ---: | ---: | ---: |
| Five-fold training CV | 53 | 124 | 124/124 | 124/124 |
| Student-test | 7 | 16 | 16/16 | 16/16 |

The final refit used 927 candidate rows from the 124 training fixes. Candidate-row macro F1 was 0.955 in pooled CV and 1.000 in student-test. The proposal filter favored close, directionally consistent, separated candidates. Both methods matched all selected proposals, so this pilot found no improvement over nearest-segment matching. Ground-truth accuracy remains unmeasured. The student-test trips were held out from fitting. Reserved evaluator data was excluded.

## Reproduce and load

Use the repository CPU environment. From the repository root, build the development export and point the pilot at its manifest.

```bash
python -m data_prep.build_dataset --data data --out ../provisional-knn-data --seed 2026
python -m experiments.provisional_knn.run --manifest ../provisional-knn-data/manifest.json --out ../provisional-knn-rerun
```

The runner checks feature and label file hashes and matches each provisional label to the raw trip's input hash and current route proposal. It uses the frozen trip folds, retains their disjoint train and validation membership, and reloads the exported model to check probability parity. The run uses 15 neighbors, distance-weighted voting, median imputation and standardization fitted inside each fold.

Use [metrics.json](results/metrics.json) for per-fold and per-trip support, dependency versions and model checksum. Load the trusted [pilot model](results/knn-provisional.joblib) with `joblib.load` from an environment with this repository on its import path. Use the five ordered input features listed in the report. This artifact is for provisional experiments. The official model selection and release require independently reviewed labels.
