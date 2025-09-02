from collections import defaultdict
import math

class Elo:
    def __init__(self, k=20, home_adv=60, base=1500):
        self.k = k
        self.home_adv = home_adv
        self.base = base
        self.rating = defaultdict(lambda: base)

    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_match(self, home, away, home_goals, away_goals):
        ra = self.rating[home] + self.home_adv
        rb = self.rating[away]
        ea = self.expected(ra, rb)
        eb = 1 - ea
        if home_goals > away_goals:
            sa, sb = 1.0, 0.0
        elif home_goals == away_goals:
            sa, sb = 0.5, 0.5
        else:
            sa, sb = 0.0, 1.0
        self.rating[home] += self.k * (sa - ea)
        self.rating[away] += self.k * (sb - eb)
