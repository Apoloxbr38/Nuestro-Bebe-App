import numpy as np
import pandas as pd

class PoissonModel:
    def __init__(self, max_goals: int = 10):
        self.max_goals = max_goals
        self.teams = set()
        self.avg_home = 1.4
        self.avg_away = 1.1
        self.attack_home = {}
        self.defense_home = {}
        self.attack_away = {}
        self.defense_away = {}
        self._trained = False

    def fit(self, df: pd.DataFrame):
        cols = {c.lower(): c for c in df.columns}
        hcol = cols.get("hometeam") or "HomeTeam"
        acol = cols.get("awayteam") or "AwayTeam"
        hg = cols.get("fthg") or "FTHG"
        ag = cols.get("ftag") or "FTAG"

        df = df[[hcol, acol, hg, ag]].copy()
        df.columns = ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        self.teams = set(df["HomeTeam"]).union(set(df["AwayTeam"]))

        self.avg_home = df["FTHG"].mean()
        self.avg_away = df["FTAG"].mean()

        home_stats = df.groupby("HomeTeam")["FTHG"].mean().rename("GF_H").to_frame()
        away_stats = df.groupby("AwayTeam")["FTAG"].mean().rename("GF_A").to_frame()
        home_conc = df.groupby("HomeTeam")["FTAG"].mean().rename("GA_H").to_frame()
        away_conc = df.groupby("AwayTeam")["FTHG"].mean().rename("GA_A").to_frame()

        teams = sorted(list(self.teams))
        stats = pd.DataFrame(index=teams)
        stats = (stats.join(home_stats, how="left")
                      .join(away_stats, how="left")
                      .join(home_conc, how="left")
                      .join(away_conc, how="left"))

        stats = stats.fillna({
            "GF_H": self.avg_home,
            "GF_A": self.avg_away,
            "GA_H": self.avg_away,
            "GA_A": self.avg_home,
        })

        self.attack_home = (stats["GF_H"] / self.avg_home).to_dict()
        self.attack_away = (stats["GF_A"] / self.avg_away).to_dict()
        self.defense_home = (stats["GA_H"] / self.avg_away).to_dict()
        self.defense_away = (stats["GA_A"] / self.avg_home).to_dict()
        self._trained = True

    def _exp_goals(self, home: str, away: str):
        ah = self.attack_home.get(home, 1.0)
        da = self.defense_away.get(away, 1.0)
        aa = self.attack_away.get(away, 1.0)
        dh = self.defense_home.get(home, 1.0)
        lam_home = max(0.05, self.avg_home * ah * da)
        lam_away = max(0.05, self.avg_away * aa * dh)
        return lam_home, lam_away

    @staticmethod
    def _poisson_pmf(lmbda: float, k: int) -> float:
        p0 = np.exp(-lmbda)
        if k == 0:
            return float(p0)
        p = p0
        for i in range(1, k + 1):
            p *= lmbda / i
        return float(p)

    def predict(self, home: str, away: str):
        if not self._trained:
            raise RuntimeError("Modelo no entrenado aún.")
        if home == away:
            raise ValueError("Los equipos deben ser distintos.")
        if home not in self.teams or away not in self.teams:
            raise ValueError("Equipo no encontrado en el dataset.")

        lam_h, lam_a = self._exp_goals(home, away)
        maxg = self.max_goals

        p_h = p_d = p_a = 0.0
        score_probs = []
        for h in range(0, maxg + 1):
            ph = self._poisson_pmf(lam_h, h)
            for a in range(0, maxg + 1):
                pa = self._poisson_pmf(lam_a, a)
                p = ph * pa
                score_probs.append(((h, a), p))
                if h > a:   p_h += p
                elif h==a:  p_d += p
                else:       p_a += p

        total = p_h + p_d + p_a
        if total > 0:
            p_h /= total; p_d /= total; p_a /= total

        score_probs.sort(key=lambda x: x[1], reverse=True)
        top5 = [{"score": f"{s[0]}-{s[1]}", "prob": round(p, 4)} for s, p in score_probs[:5]]

        return {
            "home": home, "away": away,
            "p_home": round(p_h, 4),
            "p_draw": round(p_d, 4),
            "p_away": round(p_a, 4),
            "exp_goals_home": round(lam_h, 3),
            "exp_goals_away": round(lam_a, 3),
            "top_scorelines": top5,
        }
