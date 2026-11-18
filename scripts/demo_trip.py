"""Generate a real-trip Leaflet map and JSON from the parser + baseline."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import load_trip, nearest_segment_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trip_dir", type=Path, help="Directory containing <trip_id>.* files")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"), help="Output filename prefix")
    parser.add_argument("--max-distance-m", type=float, default=250.0)
    args = parser.parse_args()

    trip = load_trip(args.trip_dir)
    predictions = nearest_segment_predictions(trip, args.max_distance_m)
    payload = {
        "trip_id": trip.trip_id,
        "track": trip.track,
        "nodes": trip.nodes,
        "arcs": trip.arcs,
        "predictions": predictions,
        # Reference route stays in this development artifact only; it is never used to predict.
        "reference_route": trip.route,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output.with_suffix(".json")
    html_path = args.output.with_suffix(".html")
    json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathSnap baseline — TRIP</title><link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{height:100%;margin:0} .legend{background:white;padding:8px 12px;line-height:1.7;font:14px sans-serif}</style></head><body><div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const d=DATA; const map=L.map('map').setView([d.track[0].latitude,d.track[0].longitude],14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}).addTo(map);
const roads=L.layerGroup(), gps=L.layerGroup(), matched=L.layerGroup(); const used=new Set(d.predictions.map(p=>p.arc_id).filter(x=>x!==null));
d.arcs.forEach(a=>{const n1=d.nodes[a.from_node_id],n2=d.nodes[a.to_node_id]; const line=[[n1.latitude,n1.longitude],[n2.latitude,n2.longitude]];
 L.polyline(line,{color:used.has(a.arc_id)?'#df3b32':'#89939d',weight:used.has(a.arc_id)?4:1,opacity:used.has(a.arc_id)?0.85:0.35}).addTo(roads);});
d.track.forEach(p=>L.circleMarker([p.latitude,p.longitude],{radius:3,color:'#1769d2',fillOpacity:.8}).addTo(gps));
L.polyline(d.track.map(p=>[p.latitude,p.longitude]),{color:'#1769d2',weight:2,dashArray:'4 5',opacity:.65}).addTo(gps);
roads.addTo(map);gps.addTo(map); L.control.layers(null,{'Road network / predicted arcs':roads,'Raw GPS':gps}).addTo(map);
map.fitBounds(L.latLngBounds(d.track.map(p=>[p.latitude,p.longitude])).pad(.08));
L.control({position:'bottomleft'}).onAdd=function(){const e=L.DomUtil.create('div','legend');e.innerHTML='<b>PathSnap geometric baseline</b><br>Gray: road network<br>Red: selected nearest arcs<br>Blue: observed GPS';return e}.addTo(map);
</script></body></html>""".replace("TRIP", trip.trip_id).replace("DATA", data)
    html_path.write_text(html, encoding="utf-8")
    matched = sum(p["arc_id"] is not None for p in predictions)
    print(f"Trip {trip.trip_id}: {len(trip.track)} GPS observations, {len(trip.nodes)} nodes, {len(trip.arcs)} arcs")
    print(f"Matched within {args.max_distance_m:g} m: {matched}/{len(predictions)} observations")
    print(f"Parser contract JSON: {json_path.resolve()}")
    print(f"Interactive map: {html_path.resolve()}")


if __name__ == "__main__":
    main()
