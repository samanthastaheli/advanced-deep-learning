"""Farm simulator v0: a tiny farming game as a Gymnasium environment.

The agent has one season (max_days) to earn as much gold as possible by
tilling soil, planting, watering, and harvesting. Everything here is a knob you
can extend later: seasons, a fluctuating market, animals, more crops, etc.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# (name, days to grow, seed cost, sell price)
CROPS = [
    ("radish", 4, 8, 20),
    ("pumpkin", 8, 25, 80),
]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


class FarmEnv(gym.Env):
    """Actions: 0-3 move, 4 till, 5.. plant crop i, then water, harvest, sleep."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, size=8, max_days=28, energy_per_day=40, rain_prob=0.25,
                 obstacle_prob=0.1, show_forecast=True, render_mode=None):
        super().__init__()
        self.size, self.max_days = size, max_days
        self.energy_per_day, self.rain_prob = energy_per_day, rain_prob
        self.obstacle_prob, self.show_forecast = obstacle_prob, show_forecast
        self.render_mode = render_mode
        k = len(CROPS)
        self.A_TILL, self.A_PLANT = 4, 5
        self.A_WATER, self.A_HARVEST, self.A_SLEEP = 5 + k, 6 + k, 7 + k
        self.action_space = spaces.Discrete(8 + k)
        # grid channels: tilled, watered, growth fraction, agent, obstacle, one per crop
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(0.0, 1.0, (5 + k, size, size), np.float32),
            "vec": spaces.Box(0.0, 1.0, (4,), np.float32),  # day, energy, gold, rain forecast
        })

    # ------------------------------------------------------------------ API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = self.size
        self.tilled = np.zeros((n, n), bool)
        self.crop = -np.ones((n, n), int)  # -1 means no crop
        self.growth = np.zeros((n, n), int)
        self.watered = np.zeros((n, n), bool)
        self.blocked = self.np_random.random((n, n)) < self.obstacle_prob
        self.pos = [n // 2, n // 2]
        self.blocked[self.pos[0], self.pos[1]] = False
        self.day, self.energy, self.gold = 0, self.energy_per_day, 100
        self.rain_tonight = self.np_random.random() < self.rain_prob
        return self._obs(), self._info()

    def step(self, action):
        gold_before = self.gold
        valid = self._apply(int(action))
        reward = (self.gold - gold_before) / 100.0 - (0.0 if valid else 0.01)
        terminated = self.day >= self.max_days
        return self._obs(), reward, terminated, False, self._info()

    # ------------------------------------------------------------- dynamics
    def _apply(self, a):
        n, (x, y) = self.size, self.pos
        if a < 4:
            nx, ny = x + MOVES[a][0], y + MOVES[a][1]
            if not (0 <= nx < n and 0 <= ny < n) or self.blocked[nx, ny]:
                return False
            self.pos = [nx, ny]
            self._spend(1)
        elif a == self.A_TILL:
            if self.tilled[x, y]:
                return False
            self.tilled[x, y] = True
            self._spend(2)
        elif self.A_PLANT <= a < self.A_WATER:
            i = a - self.A_PLANT
            cost = CROPS[i][2]
            if not self.tilled[x, y] or self.crop[x, y] >= 0 or self.gold < cost:
                return False
            self.gold -= cost
            self.crop[x, y], self.growth[x, y] = i, 0
            self._spend(2)
        elif a == self.A_WATER:
            if self.crop[x, y] < 0 or self.watered[x, y]:
                return False
            self.watered[x, y] = True
            self._spend(2)
        elif a == self.A_HARVEST:
            i = self.crop[x, y]
            if i < 0 or self.growth[x, y] < CROPS[i][1]:
                return False
            self.gold += CROPS[i][3]
            self.crop[x, y], self.growth[x, y] = -1, 0
            self._spend(2)
        else:  # sleep
            self._new_day()
        return True

    def _spend(self, cost):
        self.energy -= cost
        if self.energy <= 0:  # passed out: the day ends
            self._new_day()

    def _new_day(self):
        if self.rain_tonight:
            self.watered |= self.crop >= 0
        grow = (self.crop >= 0) & self.watered
        self.growth[grow] += 1
        self.watered[:] = False
        self.day += 1
        self.energy = self.energy_per_day
        self.rain_tonight = self.np_random.random() < self.rain_prob

    # --------------------------------------------------------- observations
    def _obs(self):
        n, k = self.size, len(CROPS)
        grid = np.zeros((5 + k, n, n), np.float32)
        grid[0], grid[1] = self.tilled, self.watered
        need = np.array([c[1] for c in CROPS])
        planted = self.crop >= 0
        grid[2][planted] = np.minimum(self.growth[planted] / need[self.crop[planted]], 1.0)
        grid[3, self.pos[0], self.pos[1]] = 1.0
        grid[4] = self.blocked
        for i in range(k):
            grid[5 + i] = self.crop == i
        vec = np.array([
            self.day / self.max_days,
            max(self.energy, 0) / self.energy_per_day,
            min(self.gold, 1000) / 1000,
            float(self.rain_tonight) if self.show_forecast else 0.0,
        ], np.float32)
        return {"grid": grid, "vec": vec}

    def _info(self):
        return {"gold": self.gold, "day": self.day}

    def render(self):
        rows = []
        for x in range(self.size):
            row = ""
            for y in range(self.size):
                if [x, y] == self.pos:
                    row += "@"
                elif self.blocked[x, y]:
                    row += "#"
                elif self.crop[x, y] >= 0:
                    i = self.crop[x, y]
                    c = CROPS[i][0][0]
                    row += c.upper() if self.growth[x, y] >= CROPS[i][1] else c
                elif self.tilled[x, y]:
                    row += "_"
                else:
                    row += "."
            rows.append(row)
        rows.append(f"day {self.day}  energy {self.energy}  gold {self.gold}")
        return "\n".join(rows)
