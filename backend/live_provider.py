from __future__ import annotations
import os
import asyncio
from typing import AsyncIterator, Dict, Any, Optional
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

# ===== Clave desde .env (raíz del repo) =====
load_dotenv()
APIFOOTBALL_KEY = os.getenv("APIFOOTBALL_KEY")
if not APIFOOTBALL_KEY:
    raise RuntimeError("Falta APIFOOTBALL_KEY en el entorno (.env)")

API_BASE = "https://v3.football.api-sports.io"

# ===== Modelo de evento que consume el frontend =====
class LiveEvent(BaseModel):
    match_id: str
    minute: int
    status: str  # "NS", "1H", "HT", "2H", "ET", "PEN", "FT"
    home: str
    away: str
    score_home: int
    score_away: int
    incident: Optional[str] = None  # "GOAL_HOME", "GOAL_AWAY", "YELLOW_HOME", etc.

# ===== Interfaz =====
class LiveProvider:
    async def get_state(self, match_id: str) -> LiveEvent:
        raise NotImplementedError

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        raise NotImplementedError

# ====== PROVEEDOR REAL: API-FOOTBALL ======
class ApiFootballProvider(LiveProvider):
    """
    Lee partidos en vivo desde API-FOOTBALL.
    match_id puede ser:
      - Un número de fixture (p.ej. "1126024")
      - "EquipoLocal:EquipoVisita" (búsqueda por nombres en los fixtures live)
    """

    def __init__(self) -> None:
        self._headers = {"x-apisports-key": APIFOOTBALL_KEY}
        # cache de últimos contadores para detectar incidentes
        self._last: Dict[str, Dict[str, Any]] = {}

    # ---------- helpers HTTP ----------
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{API_BASE}{path}", params=params or {}, headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def _list_live(self) -> list[dict]:
        data = await self._get("/fixtures", {"live": "all"})
        return data.get("response", []) or []

    async def _get_fixture_by_id(self, fixture_id: int) -> Optional[dict]:
        data = await self._get("/fixtures", {"id": fixture_id})
        arr = data.get("response", []) or []
        return arr[0] if arr else None

    @staticmethod
    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    async def _find_fixture(self, match_id: str) -> Optional[dict]:
        """
        Si match_id es dígito -> busca por fixture id.
        Si tiene ":" -> intenta por nombres (Local:Visita) en fixtures live.
        Si viene vacío -> toma el primer live (si hay).
        """
        mid = (match_id or "").strip()
        if mid.isdigit():
            return await self._get_fixture_by_id(int(mid))

        # Buscar en live por nombres
        live = await self._list_live()
        if not mid:
            return live[0] if live else None

        if ":" in mid:
            left, right = [self._norm(x) for x in mid.split(":", 1)]
            for fx in live:
                h = self._norm(fx["teams"]["home"]["name"])
                a = self._norm(fx["teams"]["away"]["name"])
                # match flexible: exacto o contiene
                if (left in h or h in left) and (right in a or a in right):
                    return fx
        else:
            # si te pasan un solo nombre, intenta hacer match con cualquiera de los dos
            q = self._norm(mid)
            for fx in live:
                h = self._norm(fx["teams"]["home"]["name"])
                a = self._norm(fx["teams"]["away"]["name"])
                if q in h or q in a:
                    return fx
        return None

    # ---------- mapping a LiveEvent ----------
    @staticmethod
    def _status_short(fx: dict) -> str:
        s = fx.get("fixture", {}).get("status", {}).get("short") or "NS"
        # normalizamos a lo que usa el frontend
        mapping = {
            "NS": "NS",
            "1H": "1H",
            "HT": "HT",
            "2H": "2H",
            "ET": "ET",
            "P": "PEN",
            "FT": "FT",
            "AET": "ET",
            "PEN": "PEN",
            "SUSP": "SUSP",
            "INT": "HT",
            "BT": "HT",
        }
        return mapping.get(s, s)

    @staticmethod
    def _elapsed(fx: dict) -> int:
        el = fx.get("fixture", {}).get("status", {}).get("elapsed")
        try:
            return int(el or 0)
        except Exception:
            return 0

    @staticmethod
    def _scores(fx: dict) -> tuple[int, int]:
        g = fx.get("goals", {}) or {}
        return int(g.get("home") or 0), int(g.get("away") or 0)

    @staticmethod
    def _names(fx: dict) -> tuple[str, str]:
        t = fx.get("teams", {}) or {}
        return (t.get("home", {}).get("name") or "Home",
                t.get("away", {}).get("name") or "Away")

    @staticmethod
    def _fixture_key(fx: dict) -> str:
        fid = fx.get("fixture", {}).get("id")
        return str(fid) if fid is not None else "unknown"

    def _to_event(self, fx: dict, incident: Optional[str] = None) -> LiveEvent:
        h, a = self._names(fx)
        sh, sa = self._scores(fx)
        st = self._status_short(fx)
        fid = self._fixture_key(fx)
        return LiveEvent(
            match_id=fid,
            minute=self._elapsed(fx),
            status=st,
            home=h,
            away=a,
            score_home=sh,
            score_away=sa,
            incident=incident
        )

    # ---------- incident detection ----------
    def _diff_incident(self, key: str, fx: dict) -> Optional[str]:
        """Detecta cambio de goles/tarjetas entre tick y tick para emitir incident."""
        prev = self._last.get(key, {})
        # Goles
        sh, sa = self._scores(fx)
        psh, psa = prev.get("score_home", 0), prev.get("score_away", 0)
        if sh > psh:
            return "GOAL_HOME"
        if sa > psa:
            return "GOAL_AWAY"

        # Tarjetas (si vienen eventos en la respuesta del fixture)
        events = fx.get("events") or []
        # Chequeo simple: si aumentó la cantidad total de tarjetas para algún lado
        # Nota: muchos endpoints live no traen todas las cards; esto es "best-effort".
        prev_yh = prev.get("yh", 0); prev_ya = prev.get("ya", 0)
        prev_rh = prev.get("rh", 0); prev_ra = prev.get("ra", 0)
        yh = ya = rh = ra = 0
        for ev in events:
            if ev.get("type") == "Card":
                detail = (ev.get("detail") or "").lower()
                is_y = "yellow" in detail
                is_r = "red" in detail
                team = (ev.get("team", {}) or {}).get("name", "")
                # heurística por nombre (home/away)
                home_name, away_name = self._names(fx)
                if is_y:
                    if team == home_name: yh += 1
                    elif team == away_name: ya += 1
                if is_r:
                    if team == home_name: rh += 1
                    elif team == away_name: ra += 1
        if yh > prev_yh:
            return "YELLOW_HOME"
        if ya > prev_ya:
            return "YELLOW_AWAY"
        if rh > prev_rh:
            return "RED_HOME"
        if ra > prev_ra:
            return "RED_AWAY"

        return None

    def _remember(self, key: str, fx: dict) -> None:
        sh, sa = self._scores(fx)
        # contamos tarjetas según eventos
        events = fx.get("events") or []
        yh = ya = rh = ra = 0
        home_name, away_name = self._names(fx)
        for ev in events:
            if ev.get("type") == "Card":
                detail = (ev.get("detail") or "").lower()
                is_y = "yellow" in detail
                is_r = "red" in detail
                team = (ev.get("team", {}) or {}).get("name", "")
                if is_y:
                    if team == home_name: yh += 1
                    elif team == away_name: ya += 1
                if is_r:
                    if team == home_name: rh += 1
                    elif team == away_name: ra += 1

        self._last[key] = {
            "score_home": sh, "score_away": sa,
            "yh": yh, "ya": ya, "rh": rh, "ra": ra,
        }

    # ---------- API pública ----------
    async def get_state(self, match_id: str) -> LiveEvent:
        fx = await self._find_fixture(match_id)
        if not fx:
            # Si no hay partido, devolvemos placeholder NS
            # (el frontend lo mostrará como sin comenzar)
            return LiveEvent(
                match_id=match_id or "no-match",
                minute=0, status="NS",
                home="—", away="—",
                score_home=0, score_away=0,
            )
        key = self._fixture_key(fx)
        self._remember(key, fx)
        return self._to_event(fx)

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        """
        Pooling simple cada ~2s. Emite estado + incidentes hasta FT.
        """
        # Primera localización
        fx = await self._find_fixture(match_id)
        if not fx:
            # Emitimos un único NS y cerramos
            yield LiveEvent(
                match_id=match_id or "no-match",
                minute=0, status="NS",
                home="—", away="—",
                score_home=0, score_away=0,
            )
            return

        key = self._fixture_key(fx)
        self._remember(key, fx)
        yield self._to_event(fx)

        # loop de actualización
        while True:
            await asyncio.sleep(2.0)
            # refrescamos por id directo (más fiable)
            fid = fx.get("fixture", {}).get("id")
            if fid:
                fx = await self._get_fixture_by_id(int(fid))
            else:
                fx = await self._find_fixture(key)  # fallback

            if not fx:
                # si desaparece, cerramos
                yield LiveEvent(
                    match_id=key, minute=0, status="FT",
                    home="—", away="—", score_home=0, score_away=0,
                )
                break

            st = self._status_short(fx)
            inc = self._diff_incident(key, fx)
            self._remember(key, fx)
            yield self._to_event(fx, incident=inc)

            if st in {"FT", "AET", "PEN"}:
                # avisamos cierre con un FT
                break

# ====== (Opcional) Proveedor Mock para pruebas locales ======
class MockProvider(LiveProvider):
    """Simula un partido para pruebas sin API."""
    def __init__(self) -> None:
        self._cache: Dict[str, LiveEvent] = {}

    def _ensure(self, match_id: str) -> LiveEvent:
        if match_id not in self._cache:
            home, away = ("Sexy FC", "Romance United") if ":" not in match_id else match_id.split(":")
            self._cache[match_id] = LiveEvent(
                match_id=match_id, minute=0, status="NS",
                home=home, away=away, score_home=0, score_away=0
            )
        return self._cache[match_id]

    async def get_state(self, match_id: str) -> LiveEvent:
        return self._ensure(match_id)

    async def stream(self, match_id: str) -> AsyncIterator[LiveEvent]:
        st = self._ensure(match_id)
        st.status = "1H"
        for m in range(1, 10):
            await asyncio.sleep(1.0)
            st.minute = m
            if m in (3, 7):
                st.score_home += 1
                st.incident = "GOAL_HOME"
            else:
                st.incident = None
            yield st
        st.status = "FT"
        st.incident = None
        yield st

