# PathSnap

PathSnap matches timestamped GPS fixes to a directed road network. Sprint 1 runs the development-data inventory, geometric candidate ranking, nearest-segment demo, and candidate feature extraction. The [viva guide](docs/viva-guide.md) explains those stages and the remaining work.

## CPU setup

The pinned Sprint 2 environment is verified on CPython 3.14 (3.14.10) on Linux x86-64.
Use that environment for reproducible results. The CPU XGBoost package supplies
the existing `xgboost` import. NumPy, SciPy (scikit-learn and XGBoost dependency),
model/runtime libraries, the Folium renderer, and HTTP test dependencies are pinned.
`constraints-cpu.txt` records the resolved transitive versions.

From the repository root, create a virtual environment and install the requirements.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python main.py demo data/train/00000005 --output outputs/trip_00000005
```

Windows and macOS installs have not been verified. On Linux, use the documented
CPython version if pip cannot find a binary wheel. Install packages with the
virtual environment's `python -m pip` if command imports fail.

The inventory, features, dataset build, readiness checker, and offline review
sheet use Python's standard library. The Folium demo requires the pinned mapping
dependencies and loads Leaflet and map tiles in the browser.

Open `outputs/trip_00000005.html` or the [checked-in map](docs/evidence/trip_00000005.html) in a browser with internet access. Toggle gray roads, blue GPS observations, red nearest segments, and the green reference route independently. The reference starts hidden. The generated JSON contains one prediction per observation and a separate development reference sequence. The demo matches all 1,045 fixes within 250 m on this trip. This is distance-cutoff coverage. Prediction accuracy requires validated observation labels.

## Inspect data, candidates, and features

```bash
python main.py inspect
python main.py candidates data/train/00000005 --top-k 5
python main.py features data/train/00000005
python -m unittest discover -s tests -v
```

Run `python main.py --help` for the command list and add `--help` after any command for its options. Reports and demo files are generated in the ignored `outputs/` directory. The inventory scans only `data/train` and `data/student_test`. The ranked candidate command uses an uncapped search for Top-K. The map baseline uses a 250 m cutoff, and the feature extractor defaults to 100 m and up to eight candidates. The feature command exports the inference table separately from the provisional alignment.

The full suite includes local HTTP tests that bind loopback and tiny synthetic
model fit and Optuna tests. The fixture tests use synthetic inputs. A socket
permission failure in the UI tests means the environment needs loopback access.

## Dataset export and review

```bash
python -m data_prep.build_dataset --data data --out artifacts/data --seed 2026
python -m data_prep.check_dataset --manifest artifacts/data/manifest.json
python -m data_prep.check_dataset --require-training-ready
```

The builder writes candidate features, per-observation proposals, trip and fold
coverage, review queues, grouped folds, and a hash-bearing manifest. The strict
readiness command exits 1 while independent training review is insufficient.
See the [review instructions](data_prep/README.md) and
[current readiness report](docs/data_readiness_report.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `main.py` and `__init__.py` | CLI entry point and root package initializer |
| `map_matching/` | Trip parser, candidate ranking, baseline, features, inventory, and Folium rendering |
| `data/` | Original development files organised by partition and trip |
| `docs/` | Architecture, viva guide, experiment plan, official guidelines, and representative evidence |
| `tests/` | Parser, candidate, and inventory tests |

- [Sprint 1 plan](PathSnap_Sprint_1_Plan.md) and [verification status](docs/sprint-1.md)
- [Architecture and parser contract](docs/architecture.md), [dataset findings](docs/dataset.md), and [feature schema](docs/features.md)
- [Five-model experiment plan](docs/model_plan.md) and [official assignment guidelines](docs/guidelines/)

## Optional CUDA environment

For CUDA-capable NVIDIA Linux x86-64 hosts, install the GPU dependency set in a separate
environment to keep the two XGBoost distributions from sharing an import directory.

```bash
python3.14 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements-gpu.txt
python -m pip check
nvidia-smi
```

The logistic and random forest configs currently request CUDA. For their CPU
backend, record a separate config with `params.device: cpu`. The KNN and standalone
XGBoost factory use CPU. See the [deployment guide](deployment/README.md) for API setup.
