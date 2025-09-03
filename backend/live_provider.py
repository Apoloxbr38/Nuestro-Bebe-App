# backend/live_provider.py
import os
import asyncio
import random
from typing import AsyncIterator, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# Cargar API key desde .env
load_dotenv()
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")


class LiveEvent(BaseModel):
    match_id: str
    minute: int
    status: str  # "NS", "1H", "HT", "2H", "FT"
    home: str
    away: str
    score_home: int
    score_away: int
    incident: Optional[str] = None


class LiveProvider:
    """Interfaz para proveedores en vivo."""
    async def get_state(self, match_id: str) -> LiveEvent:
        raise NotImplementedError

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        raise NotImplementedError


# ===========================
# 🔹 Proveedor REAL con API-Football
# ===========================
class ApiFootballProvider(LiveProvider):
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3/fixtures"

    def __init__(self):
        if not RAPIDAPI_KEY:
            raise RuntimeError("⚠️ RAPIDAPI_KEY no configurada en .env")
        self.headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }

    async def get_state(self, match_id: str) -> LiveEvent:
        """
        Obtiene estado actual de un partido en vivo según su fixture_id (API-Football).
        """
        url = f"{self.BASE_URL}?id={match_id}"
        resp = requests.get(url, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("response"):
            raise ValueError(f"No se encontró info para match_id={match_id}")

        fixture = data["response"][0]["fixture"]
        teams = data["response"][0]["teams"]
        goals = data["response"][0]["goals"]

        return LiveEvent(
            match_id=str(fixture["id"]),
            minute=fixture["status"]["elapsed"] or 0,
            status=fixture["status"]["short"],  # ej "1H", "HT", "2H"
            home=teams["home"]["name"],
            away=teams["away"]["name"],
            score_home=goals["home"] or 0,
            score_away=goals["away"] or 0,
            incident=None
        )

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        """
        Stream en vivo (polling cada 20s).
        """
        while True:
            try:
                state = await self.get_state(match_id)
                yield state
                if state.status == "FT":
                    break
            except Exception as e:
                print("Error en stream:", e)
                break
            await asyncio.sleep(20)  # cada 20s consultar de nuevo


# ===========================
# 🔹 Proveedor SIMULADO
# ===========================
class MockProvider(LiveProvider):
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
        state.status = "1H"
        for minute in range(1, 47):
            await asyncio.sleep(1.0)
            state.minute = minute
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

        state.status = "HT"; state.minute = 45; state.incident = None
        yield state
        await asyncio.sleep(2.0)

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

        state.status = "FT"; state.incident = None
        yield state
