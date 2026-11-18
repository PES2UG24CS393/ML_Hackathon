"""Small stdlib-only evaluator UI proxy for the PathSnap prediction API."""

from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MOCK_RESULT = {
    "trip_id": "00000005",
    "gps": [[8.0, 49.0], [8.001, 49.0005], [8.01, 49.01]],
    "matches": [{"obs_id": 0, "arc_id": 1}, {"obs_id": 1, "arc_id": None},
                {"obs_id": 2, "arc_id": 7}],
    "route_arc_ids": [1, 7],
    "roads": [{"arc_id": 1, "coordinates": [[8.0, 49.0], [8.001, 49.0]]}],
    "route_parts": [{"part_id": 0, "arc_ids": [1],
                      "coordinates": [[8.0, 49.0], [8.001, 49.0]]},
                    {"part_id": 1, "arc_ids": [7],
                     "coordinates": [[8.01, 49.01], [8.011, 49.01]]}],
    "metrics": {"matched_observations": 2, "observation_count": 3},
    "breaks": [{"obs_id": 1, "reason": "no_candidates"}],
    "elapsed_ms": 2.4,
    "status": "mock prediction",
}


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def upstream_request(path: str, *, method: str = "GET", body: object = None,
                     api_url: str | None = None, timeout: float = 15.0) -> object:
    """Forward one API request; prediction body contains only selected trip_id."""
    base = (api_url if api_url is not None else os.environ.get("PATHSNAP_API_URL", "")).rstrip("/")
    if not base:
        raise RuntimeError("PATHSNAP_API_URL is not configured")
    payload = None if body is None else _json_bytes(body)
    request = Request(base + path, data=payload, method=method,
                      headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"prediction API returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"prediction API request failed: {exc}") from exc


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PathSnap Evaluator</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;color:#172033;background:#f4f6fa}header,main{max-width:1100px;margin:auto;padding:16px}header{background:#14233b;color:white;max-width:none}header h1{max-width:1068px;margin:0 auto}main{display:grid;grid-template-columns:260px 1fr;gap:14px}.panel{background:white;padding:14px;border:1px solid #d5dbe6;border-radius:8px}label{display:block;margin:10px 0 5px}select,button{font:inherit;padding:8px;width:100%}button{background:#2259a8;color:white;border:0;border-radius:5px;cursor:pointer;margin-top:9px}button:disabled{opacity:.55}canvas{width:100%;height:520px;background:#edf3f8;border:1px solid #d5dbe6;border-radius:5px}.checks label{display:flex;gap:7px;align-items:center}.checks input{width:auto}.legend span{margin-right:12px}.dot{color:#e34838}.road{color:#708090}.route{color:#1267ce}.break{color:#d17800}.status{min-height:1.5em;white-space:pre-wrap}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{background:#eef3fa;padding:8px;border-radius:5px}.error{color:#a22121}.muted{color:#637083;font-size:.9em}@media(max-width:760px){main{grid-template-columns:1fr}canvas{height:400px}}
</style><header><h1>PathSnap Evaluator</h1></header><main>
<section class="panel"><h2>Trip</h2><label for="trip">Available API trips</label><select id="trip"></select><button id="run">Generate prediction</button><p id="status" class="status" role="status">Loading trips…</p><p class="muted">Prediction requests contain only the selected trip ID. The UI does not load training data or reference routes.</p><h3>Layers</h3><div class="checks"><label><input type="checkbox" id="gps" checked> GPS observations</label><label><input type="checkbox" id="roads" checked> Road geometry (when supplied)</label><label><input type="checkbox" id="predicted" checked> Predicted route</label></div><h3>Legend</h3><div class="legend"><span class="dot">● GPS</span><span class="road">━ Roads</span><span class="route">━ Predicted</span><span class="break">✕ Break</span></div><h3>Metrics</h3><div id="metrics" class="metrics">No prediction yet</div><h3>Route breaks</h3><div id="breaks" class="muted">No prediction yet</div></section>
<section class="panel"><canvas id="map" aria-label="Trip map"></canvas><p class="muted">Local coordinate view; route parts are drawn separately across breaks. Bounds fit the visible trip data.</p></section></main>
<script>
const tripSelect=document.querySelector('#trip'), statusEl=document.querySelector('#status'), metricsEl=document.querySelector('#metrics'), canvas=document.querySelector('#map'), ctx=canvas.getContext('2d'); let result=null;
function status(text,error=false){statusEl.textContent=text;statusEl.className='status'+(error?' error':'')}
async function api(path,options={}){const r=await fetch(path,options);let data;try{data=await r.json()}catch{throw Error('UI server returned invalid JSON')}if(!r.ok){const e=data.error;throw Error((e&&typeof e==='object'?e.message:e)||('HTTP '+r.status))}return data}
async function loadTrips(){try{const data=await api('/api/trips');const trips=Array.isArray(data)?data:(data.trip_ids||data.trips||[]);tripSelect.replaceChildren(...trips.map(x=>{const id=typeof x==='string'?x:(x.trip_id||x.id);const o=document.createElement('option');o.value=id;o.textContent=id;return o}));status(trips.length?'Select one trip and generate a prediction.':'The API returned no trips.')}catch(e){status('Could not load trips: '+e.message,true)}}
function coordsOf(item){return item.coordinates||item.geometry?.coordinates||[]}
function draw(){const dpr=window.devicePixelRatio||1, box=canvas.getBoundingClientRect();canvas.width=box.width*dpr;canvas.height=box.height*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,box.width,box.height);if(!result)return;
const gps=result.gps||result.observations||result.gps_points||[], roads=result.roads||result.road_segments||[], parts=result.route_parts||result.predicted_route_parts||[];const point=p=>Array.isArray(p)?p:[p.longitude??p.lon,p.latitude??p.lat];let sets=[];if(document.querySelector('#gps').checked)sets.push(gps.map(point));if(document.querySelector('#roads').checked)sets.push(...roads.map(coordsOf));if(document.querySelector('#predicted').checked)sets.push(...parts.map(coordsOf));const breakRows=result.breaks||[];if(document.querySelector('#gps').checked)sets.push(breakRows.map(b=>{const i=(result.matches||[]).findIndex(m=>m.obs_id===b.obs_id);return i>=0?point(gps[i]):null}));const pts=sets.flat().filter(p=>Array.isArray(p)&&Number.isFinite(+p[0])&&Number.isFinite(+p[1]));if(!pts.length){ctx.fillStyle='#49566a';ctx.fillText('No coordinates in this prediction',20,30);return}let xs=pts.map(p=>+p[0]),ys=pts.map(p=>+p[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),pad=24;const spanX=maxX-minX||1e-6,spanY=maxY-minY||1e-6,s=Math.min((box.width-2*pad)/spanX,(box.height-2*pad)/spanY);const xy=p=>[pad+(+p[0]-minX)*s+(box.width-2*pad-spanX*s)/2,box.height-pad-(+p[1]-minY)*s-(box.height-2*pad-spanY*s)/2];
function lines(items,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;for(const item of items){const c=coordsOf(item);if(c.length<2)continue;ctx.beginPath();c.forEach((p,i)=>{const q=xy(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke()}}
lines(document.querySelector('#roads').checked?roads:[],'#8793a4',2);lines(document.querySelector('#predicted').checked?parts:[],'#1267ce',4);if(document.querySelector('#gps').checked){ctx.fillStyle='#e34838';for(const p of gps){const q=xy(point(p));ctx.beginPath();ctx.arc(q[0],q[1],3.5,0,Math.PI*2);ctx.fill()}ctx.strokeStyle='#d17800';ctx.lineWidth=2;for(const b of breakRows){const i=(result.matches||[]).findIndex(m=>m.obs_id===b.obs_id);if(i<0||!gps[i])continue;const q=xy(point(gps[i]));ctx.beginPath();ctx.moveTo(q[0]-5,q[1]-5);ctx.lineTo(q[0]+5,q[1]+5);ctx.moveTo(q[0]+5,q[1]-5);ctx.lineTo(q[0]-5,q[1]+5);ctx.stroke()}}
}
function showMetrics(m={},elapsed=null){const metrics=m&&typeof m==='object'?Object.entries(m):[];if(elapsed!==null)metrics.push(['elapsed_ms',elapsed]);metricsEl.replaceChildren(...(metrics.length?metrics.map(([k,v])=>{const d=document.createElement('div');d.className='metric';d.textContent=k.replaceAll('_',' ')+': '+(typeof v==='object'?JSON.stringify(v):v);return d}):[Object.assign(document.createElement('span'),{textContent:'No aggregate metrics supplied'})]));const bs=result?.breaks||[];document.querySelector('#breaks').textContent=bs.length?bs.map(b=>'obs '+b.obs_id+': '+(b.reason||'route discontinuity')).join(' · '):'No route breaks reported.'}
document.querySelector('#run').onclick=async()=>{const trip_id=tripSelect.value;if(!trip_id){status('Select a trip first.',true);return}const b=document.querySelector('#run');b.disabled=true;status('Requesting prediction…');try{result=await api('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trip_id})});showMetrics(result.metrics,result.elapsed_ms);draw();status(result.status||'Prediction complete.')}catch(e){result=null;showMetrics(null);draw();status('Prediction failed: '+e.message,true)}finally{b.disabled=false}};
for(const id of ['gps','roads','predicted'])document.querySelector('#'+id).onchange=draw;window.addEventListener('resize',draw);loadTrips();
</script></html>'''


class EvaluatorHandler(BaseHTTPRequestHandler):
    mock = False
    api_url: str | None = None

    def _send(self, code: int, value: object, content_type: str = "application/json") -> None:
        body = value.encode("utf-8") if isinstance(value, str) else _json_bytes(value)
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send(200, HTML, "text/html")
        elif self.path == "/api/trips":
            try:
                value = {"trip_ids": ["00000005"]} if self.mock else upstream_request(
                    "/trips", api_url=self.api_url)
                self._send(200, value)
            except (RuntimeError, ValueError) as exc:
                self._send(502, {"error": str(exc)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("request body must be 1 to 4096 bytes")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict) or set(payload) != {"trip_id"}:
                raise ValueError("prediction request must contain exactly trip_id")
            trip_id = payload["trip_id"]
            if not isinstance(trip_id, str) or re.fullmatch(r"[0-9]{8}", trip_id) is None:
                raise ValueError("trip_id must contain exactly eight ASCII digits")
            value = ({**MOCK_RESULT, "trip_id": trip_id} if self.mock else upstream_request(
                "/predict", method="POST", body={"trip_id": trip_id}, api_url=self.api_url))
            self._send(200, value)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            code = 400 if isinstance(exc, (ValueError, json.JSONDecodeError)) else 502
            self._send(code, {"error": str(exc)})

    def log_message(self, fmt: str, *args: object) -> None:
        pass


def serve(host: str = "127.0.0.1", port: int = 8765, *, mock: bool = False,
          api_url: str | None = None) -> None:
    handler = type("ConfiguredEvaluatorHandler", (EvaluatorHandler,),
                   {"mock": mock, "api_url": api_url})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"PathSnap Evaluator: http://{host}:{port} (mock={mock})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mock", action="store_true", help="serve in-memory mock API responses")
    parser.add_argument("--api-url", help="override PATHSNAP_API_URL")
    args = parser.parse_args()
    serve(args.host, args.port, mock=args.mock, api_url=args.api_url)


if __name__ == "__main__":
    main()
