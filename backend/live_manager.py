# backend/live_manager.py
from __future__ import annotations
import os
import asyncio
from typing import Dict, Set, Any
from .live_provider import ApiFootballProvider, MockProvider, LiveEvent, LiveProvider

class LiveManager:
    def __init__(self) -> None:
        mode = (os.getenv("LIVE_PROVIDER") or "api").lower()
        if mode == "mock":
            self.provider: LiveProvider = MockProvider()
        else:
            # por defecto: real API-Football
            self.provider = ApiFootballProvider()
        self._queues: Dict[str, Set[asyncio.Queue]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    async def subscribe(self, match_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(match_id, set()).add(q)
        if match_id not in self._tasks:
            self._tasks[match_id] = asyncio.create_task(self._pump(match_id))
        return q

    def unsubscribe(self, match_id: str, q: asyncio.Queue) -> None:
        if match_id in self._queues:
            self._queues[match_id].discard(q)
            if not self._queues[match_id]:
                self._queues.pop(match_id, None)
        # si ya no hay subs, cancelamos el task
        if match_id not in self._queues and match_id in self._tasks:
            self._tasks[match_id].cancel()
            self._tasks.pop(match_id, None)

    async def _pump(self, match_id: str):
        try:
            async for ev in self.provider.stream(match_id):
                payload = ev.model_dump()
                for q in list(self._queues.get(match_id, [])):
                    await q.put(payload)
                if ev.status == "FT":
                    # avisamos cierre
                    for q in list(self._queues.get(match_id, [])):
                        await q.put({"_close": True})
                    break
        except Exception as e:
            for q in list(self._queues.get(match_id, [])):
                await q.put({"type": "error", "message": str(e)})
        finally:
            # limpieza
            self._queues.pop(match_id, None)
            self._tasks.pop(match_id, None)
