import os
import asyncio
import contextlib
import json
import time
import uuid
from typing import List, Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from sqlalchemy import select, func

from .db import SessionLocal
from .models import Message
from .config import settings
from .nlp import nlp_pipeline
from .repo import save_result, query_messages

# quick early-language hint (optional)
from langdetect import detect


app = FastAPI(title="Real-time NLP Triage", version="0.4.0")


# ---------- Pydantic models ----------
class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class Sentiment(BaseModel):
    label: str
    score: float
    latency_ms: float


class NER(BaseModel):
    entities: List[Entity]
    latency_ms: float


class Toxicity(BaseModel):
    is_toxic: bool
    scores: Dict[str, float]
    threshold: float
    latency_ms: float


class Intent(BaseModel):
    label: str
    scores: Dict[str, float]
    method: str
    latency_ms: float


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=1, description="Message to analyze")


class AnalyzeOut(BaseModel):
    ok: bool
    language: str
    sentiment: Sentiment
    ner: NER
    toxicity: Toxicity
    intent: Intent
    total_ms: float
    model_init_ms: float


# ---------- Basic routes ----------
@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Simple in-browser client (JSON POST to /analyze) ----------
@app.get("/client", response_class=HTMLResponse, include_in_schema=False)
def client_page():
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>NLP Client</title>
      <style>
        body{font-family:system-ui, Arial, sans-serif; max-width: 720px; margin: 40px auto;}
        textarea{width:100%; height:120px; padding:10px; font-size:16px;}
        button{padding:10px 16px; font-size:16px; cursor:pointer}
        pre{background:#f7f7f7; padding:12px; overflow:auto}
      </style>
    </head>
    <body>
      <h1>Send a message</h1>
      <textarea id="msg" placeholder="Type your message..."></textarea><br/>
      <button onclick="send()">Analyze</button>
      <h3>Response</h3>
      <pre id="out"></pre>
      <script>
        async function send(){
          const txt = document.getElementById('msg').value.trim();
          const out = document.getElementById('out');
          if (!txt) { out.textContent = 'Please type something.'; return; }
          out.textContent = 'Sending...';
          try{
            const r = await fetch('/analyze', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text: txt })
            });
            const j = await r.json();
            out.textContent = JSON.stringify(j, null, 2);
          }catch(e){
            out.textContent = 'Error: ' + e;
          }
        }
      </script>
    </body>
    </html>
    """


@app.post("/analyze", response_model=AnalyzeOut)
def analyze(inp: AnalyzeIn):
    t0 = time.perf_counter()
    result = nlp_pipeline.analyze(inp.text)
    save_result(inp.text, result)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "ok": True,
        **result,
        "total_ms": total_ms,
        "model_init_ms": getattr(nlp_pipeline, "init_ms", -1.0),
    }


# ---------- WebSocket processing ----------
MIN_INTERVAL_MS = 100   # simple per-connection rate limit
QUEUE_MAXSIZE = 10      # backpressure: drop with error if full


async def _consumer_loop(websocket: WebSocket, queue: asyncio.Queue):
    """
    Consumes queued requests, runs NLP in a threadpool (non-blocking),
    and streams progressive + final results back to the client.
    """
    while True:
        item = await queue.get()
        cid = item["id"]
        text = item["text"]
        t0 = time.perf_counter()

        # Early language hint (fast)
        try:
            lang_hint = detect(text)
        except Exception:
            lang_hint = "unknown"
        await websocket.send_json({"type": "update", "id": cid, "stage": "language", "language": lang_hint})

        # Full analysis (blocking → run in threadpool)
        result: Dict[str, Any] = await run_in_threadpool(nlp_pipeline.analyze, text)
        elapsed = (time.perf_counter() - t0) * 1000.0
        await run_in_threadpool(save_result, text, result, cid)
        await websocket.send_json({
            "type": "result",
            "id": cid,
            **result,
            "elapsed_ms": elapsed
        })
        queue.task_done()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint with simple rate-limit + backpressure.
    Accepts text or JSON {id?, text}; streams back ack/update/result frames.
    """
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    consumer_task = asyncio.create_task(_consumer_loop(websocket, queue))

    last_ts = 0.0
    try:
        while True:
            raw = await websocket.receive_text()

            # Rate-limit per connection
            now = time.perf_counter()
            if (now - last_ts) * 1000.0 < MIN_INTERVAL_MS:
                await websocket.send_json({"type": "error", "reason": "rate_limited", "min_interval_ms": MIN_INTERVAL_MS})
                continue
            last_ts = now

            # Parse payload
            try:
                data = json.loads(raw)
                text = data.get("text", "")
                cid = data.get("id") or str(uuid.uuid4())
            except Exception:
                text = raw
                cid = str(uuid.uuid4())

            if not isinstance(text, str) or not text.strip():
                await websocket.send_json({"type": "error", "reason": "empty_text"})
                continue

            # Backpressure: bounded queue
            if queue.full():
                await websocket.send_json({"type": "error", "reason": "backpressure_queue_full", "maxsize": queue.maxsize})
                continue

            await queue.put({"id": cid, "text": text})
            await websocket.send_json({"type": "ack", "id": cid})

    except WebSocketDisconnect:
        pass
    finally:
        consumer_task.cancel()
        with contextlib.suppress(Exception):
            await consumer_task


# ---------- Minimal in-browser WebSocket client (moved to /client-ws) ----------
@app.get("/client-ws", response_class=HTMLResponse, include_in_schema=False)
def client_ws():
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Realtime NLP Client (WebSocket)</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:20px;}
    #log{white-space:pre-wrap;border:1px solid #ccc;padding:10px;height:340px;overflow:auto;}
    input,button{font-size:16px;}
    .msg{margin:6px 0;}
    .ack{color:#888}
    .update{color:#0a6}
    .result{color:#06c}
    .error{color:#c00}
  </style>
</head>
<body>
  <h2>Realtime NLP Client (WebSocket)</h2>
  <div>
    <input id="text" size="70" placeholder="Type a message..." />
    <button id="send">Send</button>
    <button id="spam">Send 5 quickly (rate/bp demo)</button>
  </div>
  <div id="log"></div>

<script>
  const log = (obj, cls="msg") => {
    const el = document.createElement("div");
    el.className = cls;
    el.textContent = typeof obj === "string" ? obj : JSON.stringify(obj);
    const box = document.getElementById("log");
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
  };

  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => log("WebSocket connected");
  ws.onclose = () => log("WebSocket closed");
  ws.onerror = () => log("WebSocket error", "error");
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      const cls = data.type || "msg";
      log(data, cls);
    } catch {
      log(event.data);
    }
  };

  const send = () => {
    const text = document.getElementById("text").value;
    if (!text.trim()) return;
    ws.send(JSON.stringify({ text }));
    document.getElementById("text").value = "";
  };

  document.getElementById("send").onclick = send;
  document.getElementById("text").addEventListener("keydown", e => {
    if (e.key === "Enter") send();
  });

  // quick demo of rate-limit/backpressure behavior
  document.getElementById("spam").onclick = () => {
    for (let i = 0; i < 5; i++) {
      ws.send(JSON.stringify({ text: `quick-${i}: Order #12345 never arrived and I want a refund.` }));
    }
  };
</script>
</body>
</html>
""")


# ---------- Query and metrics ----------
@app.get("/search")
def search(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sentiment: Optional[str] = None,
    intent: Optional[str] = None,
    toxic: Optional[bool] = None,
    language: Optional[str] = None,
    q: Optional[str] = None,
):
    """
    Query stored messages. Example:
    /search?intent=refund_request&toxic=true&limit=20
    /search?q=order%20never%20arrived&sentiment=negative
    """
    rows = query_messages(
        limit=limit, offset=offset, sentiment=sentiment,
        intent=intent, toxic=toxic, language=language, q=q
    )
    return {"count": len(rows), "items": rows}


@app.get("/metrics")
def metrics():
    with SessionLocal() as db:
        total = db.execute(select(func.count(Message.id))).scalar_one()

        # sentiment counts
        sent_counts = db.execute(
            select(Message.sentiment_label, func.count())
            .group_by(Message.sentiment_label)
        ).all()
        sentiment_counts = {k or "unknown": int(v) for k, v in sent_counts}

        # intent counts
        intent_counts_raw = db.execute(
            select(Message.intent_label, func.count())
            .group_by(Message.intent_label)
        ).all()
        intent_counts = {k or "unknown": int(v) for k, v in intent_counts_raw}

        # toxic count
        toxic_count = db.execute(
            select(func.count()).where(Message.toxicity_is_toxic == True)
        ).scalar_one()

        return {
            "total_messages": int(total),
            "sentiment_counts": sentiment_counts,
            "intent_counts": intent_counts,
            "toxic_count": int(toxic_count),
        }


@app.get("/debug/dbinfo")
def dbinfo():
    url = settings.database_url
    path = None
    size = None
    if url.startswith("sqlite:///./"):
        path = os.path.abspath(url.replace("sqlite:///./", ""))
        size = os.path.getsize(path) if os.path.exists(path) else None
    elif url.startswith("sqlite:///"):
        # handle sqlite:///data.db (no './')
        path = os.path.abspath(url.replace("sqlite:///", ""))
        size = os.path.getsize(path) if os.path.exists(path) else None
    return {
        "database_url": url,
        "resolved_path": path,
        "exists": os.path.exists(path) if path else None,
        "size_bytes": size,
    }
