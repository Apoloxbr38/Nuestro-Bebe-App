# backend/live_router.py
from __future__ import annotations

import json
import asyncio
from typing import Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse

from .live_manager import LiveManager
from .live_provider import LiveEvent

live_router = APIRouter()
live = LiveManager()


@live_router.get("/state")
async def live_state(match_id: str | None = Query(None, description="Puede ser id, 'Equipo', o 'EquipoA:EquipoB'")):
    if not match_id:
        return JSONResponse({"error": "falta match_id (id, 'Equipo', o 'EquipoA:EquipoB')"}, status_code=400)
    ev = await live.provider.get_state(match_id)
    return JSONResponse(ev.model_dump())


@live_router.get("/search")
async def live_search(q: str = Query(..., description="Texto para buscar en partidos en vivo")):
    """
    Devuelve un listado simple de partidos que están en vivo y matchean el texto.
    Útil para autocompletar.
    """
    items = []
    try:
        # usamos método privado del provider real; si es mock, devolver vacío
        fetch_live = getattr(live.provider, "_fetch_live", None)
        if not callable(fetch_live):
            return {"results": []}
        for fx in fetch_live():
            home = fx["teams"]["home"]["name"]
            away = fx["teams"]["away"]["name"]
            if q.lower() in home.lower() or q.lower() in away.lower():
                items.append({
                    "fixture_id": fx["fixture"]["id"],
                    "home": home,
                    "away": away,
                    "status": fx["fixture"]["status"]["short"],
                    "minute": fx["fixture"]["status"].get("elapsed") or 0,
                })
    except Exception:
        pass
    return {"results": items}


@live_router.websocket("/ws")
async def live_ws(websocket: WebSocket):
    # Conexión: ws://HOST:PUERTO/live/ws?match_id=Barcelona o EquipoA:EquipoB o 1126024
    await websocket.accept()

    # Leer querystring manualmente (WebSocket no trae Query de FastAPI)
    query_string = websocket.scope.get("query_string", b"")
    if query_string:
        params = dict(pair.split(b"=", 1) if b"=" in pair else (pair, b"")
                      for pair in query_string.split(b"&"))
    else:
        params = {}
    match_id = params.get(b"match_id", b"").decode("utf-8").strip()

    if not match_id:
        await websocket.send_text(json.dumps({"type": "error", "reason": "falta match_id"}))
        await websocket.close()
        return

    q = await live.subscribe(match_id)
    try:
        while True:
            item = await q.get()
            if isinstance(item, dict) and item.get("_close"):
                await websocket.send_text(json.dumps({"type": "close", "reason": "FT"}))
                break

            # Normalizamos salida
            if isinstance(item, dict) and "home" in item and "away" in item:
                await websocket.send_text(json.dumps({"type": "event", "data": item}))
            else:
                # item puede ser LiveEvent o dict de error
                if hasattr(item, "model_dump"):
                    await websocket.send_text(json.dumps({"type": "event", "data": item.model_dump()}))
                else:
                    await websocket.send_text(json.dumps(item))

            # (Opcional) leer algo del cliente (no bloqueante)
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
