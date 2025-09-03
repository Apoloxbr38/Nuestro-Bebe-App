# backend/live_manager.py
from __future__ import annotations
import asyncio
from typing import Dict, Set
from .live_provider import LiveProvider, MockProvider, LiveEvent


class LiveManager:
    def __init__(self, provider: LiveProvider | None = None) -> None:
        self.provider = provider or MockProvider()
        self.rooms: Dict[str, Set[asyncio.Queue]] = {}   # match_id -> suscriptores
        self.running_tasks: Dict[str, asyncio.Task] = {} # match_id -> task de stream

    async def subscribe(self, match_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.rooms.setdefault(match_id, set()).add(q)
        if match_id not in self.running_tasks:
            self.running_tasks[match_id] = asyncio.create_task(self._run_stream(match_id))
        return q

    def unsubscribe(self, match_id: str, q: asyncio.Queue) -> None:
        if match_id in self.rooms:
            self.rooms[match_id].discard(q)

    async def _broadcast(self, match_id: str, event: LiveEvent) -> None:
        if match_id not in self.rooms:
            return
        for q in list(self.rooms[match_id]):
            await q.put(event.model_dump())

    async def _run_stream(self, match_id: str) -> None:
        try:
            async for ev in self.provider.stream(match_id):
                await self._broadcast(match_id, ev)
                if ev.status == "FT":
                    break
        finally:
            for q in self.rooms.get(match_id, set()):
                await q.put({"_close": True})
            self.rooms.pop(match_id, None)
            self.running_tasks.pop(match_id, None)
