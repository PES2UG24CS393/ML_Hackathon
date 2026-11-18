# PathSnap problem and architecture

PathSnap reconstructs a vehicle's route through a directed road network from a time-ordered series of GPS observations. GPS drift can place a sample closer to a parallel road than the road the vehicle used. Individually plausible matches can produce an impossible route when the selected arcs have no directed path between them. A gap in sampling can also hide several traversed arcs, so consecutive GPS samples may match roads that do not share an endpoint.

## Trip inputs

Each trip has whitespace-delimited files without headers. Node and arc IDs are implicit zero-based row indices within that trip's network.

| File | Columns | Role |
| --- | --- | --- |
| `.track` | Longitude, latitude, timestamp in seconds | GPS observations in trip order. Preserve the source row index as the observation ID when cleaning or filtering. |
| `.nodes` | Longitude, latitude | Location of each road-network node. |
| `.arcs` | From-node ID, to-node ID | Directed road segment between two node rows. Validate both endpoint IDs before building segment geometry. |
| `.route` | Arc ID | Optional chronological reference sequence. Resolve each ID against the trip's arc rows and preserve sequence order, including repeated arcs. |

Prediction takes the track and road network as input. Reference routes support offline label development and evaluation where authorized. Evaluator references stay on the server for evaluation and outside prediction and UI payloads.

## Candidate and reference alignment

A candidate is a directed arc that might explain one GPS observation, identified by `(trip_id, observation_index, arc_id)`. A reference route records which arcs the vehicle traversed in sequence. Multiple samples can fall on one arc, some arcs can be traversed between samples, and a route can revisit an arc. Row offsets and route membership alone therefore cannot establish the correct candidate label for an observation.

Investigate a monotonic alignment of GPS observations to positions along the reference traversal. Check alignment quality on representative training trips before using the inferred associations as labels. Record ambiguous associations and distinguish them from candidate misses, where an accepted reference association points to an arc outside the generated candidate set. Keep all observations from the same training trip together in validation splits and reserve evaluator trips for final evaluation.

## Open decisions

- Determine reference availability by split and examine coordinate validity, timestamp ordering, sampling gaps and directed network connectivity.
- Validate the alignment method and decide how to handle uncertain or unavailable labels and quantify candidate misses.
- Choose a metric projection or geodesic distance method, then validate candidate radius or top-K selection on development data. Longitude and latitude differences in degrees cannot be used directly as distances in metres.
- Decide when to bridge a gap over a plausible directed path, split the route into matched portions, or mark an observation unmatched. Define separate policies for missing candidates and unreachable transitions.

## Planned inference outputs

Return one result per GPS observation in input order, keyed by its source index. Each result contains the selected arc ID or an explicit unmatched status. Return the reconstructed route as ordered directed arc IDs, including inferred connecting arcs where the decoder can bridge a gap. Represent breaks as separate route portions so the UI can draw them without implying a connection. Adjacent observations may select the same arc, while a traversal that revisits an arc can include that ID again.

## Planned architecture

```mermaid
flowchart LR
    Track["GPS track"] --> Candidates["Generate candidate arcs"]
    Network["Directed road network"] --> Candidates
    Candidates --> Features["Compute candidate features"]
    Features --> ML["Score candidates with trained ML model"]
    ML --> Decoder["Decode sequence with HMM / Viterbi"]
    Network --> Decoder
    Decoder --> API["Return matches and route via FastAPI on Laptop 1"]
    UI["Evaluator UI on Laptop 2"] -->|Request by trip ID| API
    API -->|Matches and route geometry| UI
```

Candidate generation narrows the search using distances in metres. Candidate features describe observation-to-arc fit, such as offset and heading agreement. The selected classifier provides emission scores for candidate arcs. The HMM uses those scores and transition scores based on directed network paths, elapsed time and plausible travel distance. Viterbi selects a sequence across observations, and connecting paths turn the decoded states into route geometry where feasible.

FastAPI runs inference on Laptop 1 and supplies the outputs needed by the map. The UI on Laptop 2 requests an evaluator trip by ID and draws the returned matches and route portions. Where reference data is available for authorized evaluation, the server computes metrics separately from prediction.

## Implementation status

This issue delivers the problem definition, data contract and proposed architecture. Dataset-wide EDA, parsers, candidate generation, label alignment, model experiments, decoding, API and UI are planned work. See the [Sprint 1 plan](../PathSnap_Sprint_1_Plan.md) for acceptance criteria and dependencies.
