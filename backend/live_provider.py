# backend/live_provider.py
from __future__ import annotations
import asyncio
import random
from typing import AsyncIterator, Dict, Any, Optional
from pydantic import BaseModel


class LiveEvent(BaseModel):
    match_id: str
    minute: int
    status: str  # "NS", "1H", "HT", "2H", "FT"
    home: str
    away: str
    score_home: int
    score_away: int
    incident: Optional[str] = None  # "GOAL_HOME", "GOAL_AWAY", "YELLOW_HOME", etc.


class LiveProvider:
    """Interfaz para proveedores en vivo."""
    async def get_state(self, match_id: str) -> LiveEvent:
        raise NotImplementedError

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        raise NotImplementedError


class MockProvider(LiveProvider):
    """
    Simula un partido de ~95' con eventos aleatorios.
    Útil para probar frontend/WebSocket.
    """
    def __init__(self) -> None:
        self._cache: Dict[str, LiveEvent] = {}

    def _ensure_match(self, match_id: str) -> LiveEvent:
        if match_id not in self._cache:
            home, away = ("Sexy FC", "Romance United") if ":" not in match_id else match_id.split(":")
            self._cache[match_id] = LiveEvent(
                match_id=match_id,
                minute=0, status="NS",
                home=home, away=away,
                score_home=0, score_away=0,
            )
        return self._cache[match_id]

    async def get_state(self, match_id: str) -> LiveEvent:
        return self._ensure_match(match_id)

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        state = self._ensure_match(match_id)
        # Inicio 1er tiempo
        state.status = "1H"
        for minute in range(1, 47):
            await asyncio.sleep(1.0)  # 1s = 1'
            state.minute = minute
            # Eventos aleatorios
            if random.random() < 0.18:
                if random.random() < 0.6:
                    state.score_home += 1
                    state.incident = "GOAL_HOME"
                else:
                    state.score_away += 1
                    state.incident = "GOAL_AWAY"
            elif random.random() < 0.08:
                state.incident = random.choice(["YELLOW_HOME", "YELLOW_AWAY", "RED_HOME", "RED_AWAY"])
            else:
                state.incident = None
            yield state

        # Descanso
        state.status = "HT"
        state.incident = None
        state.minute = 45
        yield state
        await asyncio.sleep(2.0)

        # Segundo tiempo
        state.status = "2H"
        for minute in range(46, 96):
            await asyncio.sleep(1.0)
            state.minute = minute
            if random.random() < 0.16:
                if random.random() < 0.5:
                    state.score_home += 1
                    state.incident = "GOAL_HOME"
                else:
                    state.score_away += 1
                    state.incident = "GOAL_AWAY"
            elif random.random() < 0.06:
                state.incident = random.choice(["YELLOW_HOME", "YELLOW_AWAY", "RED_HOME", "RED_AWAY"])
            else:
                state.incident = None
            yield state

        state.status = "FT"
        state.incident = None
        yield state
