# backend/live_router.py
from __future__ import annotations
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from .live_manager import LiveManager


live_router = APIRouter()
live = LiveManager()  # MockProvider por defecto

@live_router.get("/state")
async def live_state(match_id: str | None = None):
    if not match_id:
        match_id = "Sexy FC:Romance United"  # demo por defecto
    ev = await live.provider.get_state(match_id)
    return JSONResponse(ev.model_dump())

@live_router.websocket("/ws")
async def live_ws(websocket: WebSocket):
    # Ejemplo de conexión: ws://HOST:PUERTO/live/ws?match_id=Sexy FC:Romance United
    await websocket.accept()

    # Leer querystring
    query_string = websocket.scope.get("query_string", b"")
    if query_string:
        params = dict(pair.split(b"=", 1) if b"=" in pair else (pair, b"")
                      for pair in query_string.split(b"&"))
    else:
        params = {}
    match_id = params.get(b"match_id", b"Sexy FC:Romance United").decode("utf-8")

    q = await live.subscribe(match_id)
    try:
        while True:
            item = await q.get()
            if item.get("_close"):
                await websocket.send_text(json.dumps({"type": "close", "reason": "FT"}))
                break
            await websocket.send_text(json.dumps({"type": "event", "data": item}))
            # (Opcional) escuchar algo del cliente, sin bloquear
            try:
                _ = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        live.unsubscribe(match_id, q)
        try:
            await websocket.close()
        except Exception:
            pass
