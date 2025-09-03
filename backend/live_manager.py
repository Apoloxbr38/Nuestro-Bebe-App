# backend/live_manager.py
from __future__ import annotations

import asyncio
from typing import Dict, List
import os

from .live_provider import (
    LiveProvider,
    MockProvider,
    ApiFootballProvider,
    LiveEvent,
)

APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY")


class LiveManager:
    """
    Orquesta suscripciones a streams en vivo.
    Crea una tarea por 'match_id' y reparte eventos a las colas de los suscriptores.
    """
    def __init__(self) -> None:
        # Elegimos provider: real si hay key, si no mock
        if APIFOOTBALL_KEY:
            self.provider: LiveProvider = ApiFootballProvider(poll_seconds=3.0)
        else:
            self.provider = MockProvider()

        self._subs: Dict[str, List[asyncio.Queue]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, match_id: str) -> asyncio.Queue:
        """
        match_id puede ser nombre/“A:B”/id. El provider real lo resolverá por dentro.
        Devuelve una cola de la que el router leerá eventos y los enviará por WS.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subs.setdefault(match_id, []).append(q)
            if match_id not in self._tasks:
                self._tasks[match_id] = asyncio.create_task(self._run_stream(match_id))
        return q

    def unsubscribe(self, match_id: str, q: asyncio.Queue) -> None:
        """
        Quita la cola; si no quedan subs, cancela la tarea de stream.
        """
        lst = self._subs.get(match_id)
        if not lst:
            return
        if q in lst:
            lst.remove(q)
        if not lst:
            # sin subs => cancelar stream
            task = self._tasks.pop(match_id, None)
            if task and not task.done():
                task.cancel()
            self._subs.pop(match_id, None)

    async def _broadcast(self, match_id: str, payload) -> None:
        lst = self._subs.get(match_id, [])
        dead: List[asyncio.Queue] = []
        for q in lst:
            try:
                await q.put(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            self.unsubscribe(match_id, q)

    async def _run_stream(self, match_id: str) -> None:
        """
        Hilo/tarea que consume el provider.stream y reparte a suscriptores.
        """
        try:
            async for item in self.provider.stream(match_id):
                # item puede ser LiveEvent o dict {"_close": True}
                if isinstance(item, LiveEvent):
                    payload = item.model_dump()
                else:
                    payload = item
                await self._broadcast(match_id, payload)

                # Cerrar si el provider avisa cierre
                if isinstance(payload, dict) and payload.get("_close"):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self._broadcast(match_id, {"type": "error", "reason": str(e)})
        finally:
            # limpiar al terminar
            self._tasks.pop(match_id, None)
