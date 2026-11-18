"""Create a local, offline review sheet for one GPS observation and its context."""

import argparse
import csv
import html
from pathlib import Path

from map_matching.features import feature_rows
from map_matching.label_quality import label_rows, reference_break_rows
from map_matching.preprocess import load_trip

from .build_dataset import REVIEW_DECISION_FIELDS, SPLITS, hash_trip_inputs


def review_trip(data, split, trip_id, obs_id, output):
    trip_dir = data / split / trip_id
    fingerprint = hash_trip_inputs(trip_dir)
    trip = load_trip(trip_dir)
    candidates = feature_rows(trip)
    labels = label_rows(trip, candidates)
    indices = {obs.obs_id: i for i, obs in enumerate(trip.observations)}
    if obs_id not in indices:
        raise ValueError(f"Unknown observation ID: {obs_id}")
    index = indices[obs_id]
    label = labels[index]
    neighbors = range(max(0, index - 3), min(len(labels), index + 4))
    route = trip.route or ()
    breaks = reference_break_rows(trip, split)
    break_steps = {row["break_before_route_step_id"] for row in breaks}
    step_id = label.route_step_id
    nearby_steps = range(max(0, step_id - 4), min(len(route), step_id + 5)) if step_id is not None else ()
    selected = [row for row in candidates if row.obs_id == obs_id]
    segments = {segment.arc_id: segment for segment in trip.network.segments}
    center_x, center_y = trip.observation_xy[index]
    extent = 120

    def point(xy):
        return f"{xy[0] - center_x:.2f},{center_y - xy[1]:.2f}"

    def line(arc_id, color, description):
        segment = segments[arc_id]
        points = f"{point(segment.start_xy)} {point(segment.end_xy)}"
        return (f'<polyline points="{points}" stroke="{color}" stroke-width="1.7" '
                f'marker-end="url(#arrow)"><title>{description}</title></polyline>')

    drawing = [
        line(route[j].arc_id, "#a56600", f"Route step {j}, arc {route[j].arc_id}")
        for j in nearby_steps if j != step_id
    ]
    drawing += [line(row.arc_id, "#637485", f"Candidate {row.arc_id}, {row.distance_m:.2f} m")
                for row in selected if row.arc_id != label.reference_arc_id]
    if label.reference_arc_id is not None:
        drawing.append(line(label.reference_arc_id, "#792ab5", f"Proposed arc {label.reference_arc_id}"))
    for i in neighbors:
        x, y = point(trip.observation_xy[i]).split(",")
        obs = trip.observations[i]
        drawing.append(f'<circle cx="{x}" cy="{y}" r="2" fill="#0076bc">'
                       f'<title>GPS {obs.obs_id}, time {obs.timestamp_s} s</title></circle>'
                       f'<text x="{x}" y="{y}" dy="-3" font-size="4">{obs.obs_id}</text>')

    def table(headers, rows):
        header = "".join(f"<th>{cell}</th>" for cell in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
                       + "</tr>" for row in rows)
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    observation_table = table(
        ("Obs ID", "Time (s)", "Previous gap (s)", "Longitude", "Latitude", "Proposed step", "Arc", "Status"),
        [(obs.obs_id, obs.timestamp_s, obs.timestamp_s - trip.observations[i - 1].timestamp_s if i else "—",
          obs.lon, obs.lat, labels[i].route_step_id, labels[i].reference_arc_id, labels[i].alignment_status)
         for i in neighbors for obs in (trip.observations[i],)],
    )
    route_table = table(
        ("Step", "Arc", "From node", "To node", "Break before"),
        [(j, arc.arc_id, arc.from_node_id, arc.to_node_id, j in break_steps)
         for j in nearby_steps for arc in (trip.network.arcs[route[j].arc_id],)],
    )
    candidate_table = table(
        ("Rank", "Arc", "Distance (m)", "From node", "To node"),
        [(i, row.arc_id, f"{row.distance_m:.3f}", arc.from_node_id, arc.to_node_id)
         for i, row in enumerate(selected, 1) for arc in (trip.network.arcs[row.arc_id],)],
    )
    break_table = table(
        ("Break before step", "Previous arc", "Current arc", "Previous end", "Current start"),
        [(row["break_before_route_step_id"], row["previous_arc_id"], row["current_arc_id"],
          row["previous_to_node_id"], row["current_from_node_id"]) for row in breaks],
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"{split}_{trip_id}_{obs_id}"
    sheet = stem.with_suffix(".html")
    sheet.write_text(
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>PathSnap review {trip_id}/{obs_id}</title>
<style>body{{font:16px system-ui;margin:2rem;max-width:60rem}}th,td{{padding:.35rem .65rem;
text-align:left;border-bottom:1px solid #ddd}}svg{{border:1px solid #ddd;max-width:100%}}</style>
<h1>{split} trip {trip_id}, observation {obs_id}</h1>
<p>Projected meters centered on the selected GPS fix. Gray marks retained candidates,
purple marks the proposal, orange marks nearby reference steps, and blue marks GPS fixes.
Arrows show directed segment endpoints when visible. Long segments and distant fixes can be clipped.</p>
<svg width="640" height="640" viewBox="-{extent} -{extent} {2 * extent} {2 * extent}"
     aria-label="GPS fixes and directed road segments" fill="none">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4"
 markerHeight="4" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>
<circle r="100" stroke="#ccc" stroke-dasharray="2 2"/>
<g fill="#202020">{"".join(drawing)}</g></svg>
<h2>Observation sequence</h2>{observation_table}
<h2>Retained candidates</h2><p>{len(selected)} candidates. Proposed arc present: {label.candidate_arc_present}.</p>
{candidate_table}
<h2>Nearby reference steps</h2>{route_table}
<h2>Reference breaks across this trip</h2>{break_table}
<p>Input SHA-256 <code>{fingerprint}</code></p>
<p>Record the independent reviewer account, UTC date and observations supporting the verdict in a
review note. Include competing arcs and sequence or geographic evidence where relevant. Link that
note in the decision CSV. Leave uncertain proposals pending.</p></html>""",
        encoding="utf-8",
    )
    decision = stem.with_suffix(".csv")
    # Regenerating context must preserve a reviewer's edited decision.
    try:
        with decision.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_DECISION_FIELDS)
            writer.writeheader()
            writer.writerow(dict(trip_id=trip_id, obs_id=obs_id, reference_arc_id=label.reference_arc_id,
                                 route_step_id=step_id, trip_input_sha256=fingerprint))
    except FileExistsError:
        pass
    return sheet, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("trip_id")
    parser.add_argument("obs_id", type=int)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("outputs/review"))
    args = parser.parse_args()
    if len(args.trip_id) != 8 or not args.trip_id.isascii() or not args.trip_id.isdecimal():
        parser.error("trip_id must contain eight ASCII digits")
    for path in review_trip(args.data, args.split, args.trip_id, args.obs_id, args.out):
        print(path)


if __name__ == "__main__":
    main()
