# backend/live_provider.py
from __future__ import annotations

import os
import re
import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional

import requests
from pydantic import BaseModel
from dotenv import load_dotenv

# Cargar .env de la raíz del repo
load_dotenv()
APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY")

API_BASE = "https://v3.football.api-sports.io"


class LiveEvent(BaseModel):
    match_id: str
    minute: int
    status: str  # "NS", "1H", "HT", "2H", "FT", ...
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


# ----------------------------- MOCK (demo) -----------------------------
class MockProvider(LiveProvider):
    """
    Simula un partido de ~95' con eventos aleatorios.
    Útil para probar frontend/WebSocket sin API real.
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
        import random
        state = self._ensure_match(match_id)
        # Inicio 1er tiempo
        state.status = "1H"
        for minute in range(1, 47):
            await asyncio.sleep(1.0)
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


# ------------------------ API-FOOTBALL (real) -------------------------
class ApiFootballProvider(LiveProvider):
    """
    Proveedor real usando API-FOOTBALL.
    Permite 'match_id' como:
      - fixture id numérico ("1126024")
      - nombre de un equipo ("Barcelona")
      - "EquipoA:EquipoB" (prioriza este emparejamiento)
    """
    def __init__(self, poll_seconds: float = 3.0) -> None:
        if not APIFOOTBALL_KEY:
            raise RuntimeError("Falta APIFOOTBALL_KEY en el entorno (.env)")
        self._headers = {
            "x-apisports-key": APIFOOTBALL_KEY,
        }
        self._poll = max(1.0, float(poll_seconds))
        self._cache: Dict[str, LiveEvent] = {}

    # --------------------------- helpers ---------------------------
    def _norm(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    def _fetch_live(self) -> List[dict]:
        # fixtures?live=all
        r = requests.get(
            f"{API_BASE}/fixtures",
            params={"live": "all"},
            headers=self._headers,
            timeout=15,
        )
        r.raise_for_status()
        j = r.json()
        return j.get("response", [])

    def _to_event(self, fx: dict) -> LiveEvent:
        home = fx["teams"]["home"]["name"]
        away = fx["teams"]["away"]["name"]
        status = fx["fixture"]["status"]["short"] or "NS"
        minute = fx["fixture"]["status"].get("elapsed") or 0
        sh = fx["goals"]["home"] if fx.get("goals") else 0
        sa = fx["goals"]["away"] if fx.get("goals") else 0
        return LiveEvent(
            match_id=str(fx["fixture"]["id"]),
            minute=int(minute or 0),
            status=status,
            home=home,
            away=away,
            score_home=int(sh or 0),
            score_away=int(sa or 0),
        )

    # ------------------------ resolución humana ------------------------
    def resolve_match_id(self, query: str) -> Optional[str]:
        """
        Devuelve fixture id (str) o None si no encuentra.
        Acepta:
          - '123456' (ya es id)
          - 'Barcelona'
          - 'Barcelona:Real Madrid'
        """
        if not query:
            return None
        q = query.strip()
        # Si ya es número
        if q.isdigit():
            return q

        parts = [p.strip() for p in q.split(":", 1)]
        want_home = want_away = None
        if len(parts) == 2:
            want_home, want_away = self._norm(parts[0]), self._norm(parts[1])
        else:
            want = self._norm(q)

        live = self._fetch_live()

        # Emparejamiento A:B (mismo orden)
        if want_home and want_away:
            for fx in live:
                h = self._norm(fx["teams"]["home"]["name"])
                a = self._norm(fx["teams"]["away"]["name"])
                if want_home in h and want_away in a:
                    return str(fx["fixture"]["id"])
            # Fallback: invertido
            for fx in live:
                h = self._norm(fx["teams"]["home"]["name"])
                a = self._norm(fx["teams"]["away"]["name"])
                if want_home in a and want_away in h:
                    return str(fx["fixture"]["id"])
            return None

        # Un solo nombre
        for fx in live:
            h = self._norm(fx["teams"]["home"]["name"])
            a = self._norm(fx["teams"]["away"]["name"])
            if want in h or want in a:
                return str(fx["fixture"]["id"])
        return None

    # ----------------------------- API -----------------------------
    async def get_state(self, match_id: str) -> LiveEvent:
        # Resolver si viene en formato humano
        real_id = self.resolve_match_id(match_id) or match_id
        if not real_id.isdigit():
            # No se pudo resolver
            return LiveEvent(
                match_id=match_id, minute=0, status="NS",
                home="—", away="—", score_home=0, score_away=0
            )

        # Buscarlo en live
        for fx in self._fetch_live():
            if str(fx["fixture"]["id"]) == real_id:
                ev = self._to_event(fx)
                self._cache[real_id] = ev
                return ev

        # No está en vivo: regresar cache o stub
        if real_id in self._cache:
            return self._cache[real_id]

        return LiveEvent(
            match_id=real_id, minute=0, status="NS",
            home="—", away="—", score_home=0, score_away=0
        )

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        """
        Pull cada N segundos (self._poll) y emite cambios.
        Cierra cuando el status sea FT/AET/PEN.
        """
        real_id = self.resolve_match_id(match_id) or match_id
        seen: Optional[LiveEvent] = None

        while True:
            current: Optional[LiveEvent] = None
            try:
                for fx in self._fetch_live():
                    if str(fx["fixture"]["id"]) == str(real_id):
                        current = self._to_event(fx)
                        break
            except Exception:
                # En caso de error temporal de red/api, esperamos y seguimos
                await asyncio.sleep(self._poll)
                continue

            if current is not None:
                # Emitir si cambió algo visible
                if (
                    seen is None
                    or current.minute != seen.minute
                    or current.status != seen.status
                    or current.score_home != seen.score_home
                    or current.score_away != seen.score_away
                ):
                    seen = current
                    yield current

                # Terminado el partido
                if current.status in ("FT", "AET", "PEN"):
                    yield {"_close": True}  # señal de cierre
                    return
            else:
                # Si nunca lo vimos y no aparece (puede no estar live)
                if seen and seen.status in ("FT", "AET", "PEN"):
                    yield {"_close": True}
                    return

            await asyncio.sleep(self._poll)
