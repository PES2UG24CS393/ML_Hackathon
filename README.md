# PathSnap

PathSnap matches timestamped GPS fixes to a directed road network. Sprint 1 runs the development-data inventory, geometric candidate ranking, nearest-segment demo, and candidate feature extraction. The [viva guide](docs/viva-guide.md) explains those stages and the remaining work.

## Run the Sprint 1 demo

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the map dependency.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py demo data/train/00000005 --output outputs/trip_00000005
```

On Windows PowerShell, use `py -3 -m venv .venv` and `.venv\Scripts\Activate.ps1`, then run the same install and demo commands.

Open `outputs/trip_00000005.html` or the [checked-in map](docs/evidence/trip_00000005.html) in a browser with internet access. Toggle gray roads, blue GPS observations, red nearest segments, and the green reference route independently. The reference starts hidden. The generated JSON contains one prediction per observation and a separate development reference sequence. The demo matches all 1,045 fixes within 250 m on this trip. This is distance-cutoff coverage. Prediction accuracy requires validated observation labels.

## Inspect data, candidates, and features

```bash
python main.py inspect
python main.py candidates data/train/00000005 --top-k 5
python main.py features data/train/00000005
python -m unittest discover -s tests -v
```

Run `python main.py --help` for the command list and add `--help` after any command for its options. Reports and demo files are generated in the ignored `outputs/` directory. The inventory scans only `data/train` and `data/student_test`. The ranked candidate command uses an uncapped search for Top-K. The map baseline uses a 250 m cutoff, and the feature extractor defaults to 100 m and up to eight candidates. The feature command exports the inference table separately from the provisional alignment.

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
