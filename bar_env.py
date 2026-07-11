import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BarEnv_v1(gym.Env):
    """
    Кастомная среда "Бар" (v1).

    Изменения относительно v0:
      1. BAC_mind теперь СМЕЩЁН в сторону недооценки истинного BAC, и
         дисперсия шума растёт квадратично с ростом BAC — чем пьянее
         агент, тем сильнее он недооценивает своё состояние и тем менее
         предсказуемо его восприятие.
      2. Награда переработана:
         - пик награды теперь на УМЕРЕННОМ уровне BAC (ниже порога
           штрафа за здоровье), а не на экстремальном 0.5;
         - бонус за пробование РАЗНЫХ напитков за эпизод;
         - штраф за повторение одного и того же напитка подряд
           (стимулирует менять темп/тактику во времени);
         - небольшой бонус за каждый прожитый шаг + бонус за то, что
           агент успел попробовать почти все напитки к финалу.
      Итог: агенту приходится балансировать между hp, настроением и
      уровнем опьянения, а не просто максимизировать BAC.

    Действия:
        0: вода          → mood -1, BAC -0.01
        1: пиво          → mood +1, BAC +0.02
        2: вино          → mood +2, BAC +0.05
        3: водка         → mood +3, BAC +0.12
        4: безалког.     → mood +0.001, BAC без изменений
        5: выйти из бара → терминальное состояние (terminated)
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        m_sigma_BAC=1e-5,
        bias_k=0.6,             # сила недооценки BAC (растёт как BAC^2)
        noise_k=1.2,            # сила роста дисперсии шума (растёт как BAC^2)
        target_bac=0.06,        # "комфортный" целевой уровень BAC
        comfort_sigma=0.05,     # ширина гауссианы комфортной зоны BAC
        comfort_amp=40.0,       # амплитуда награды в комфортной зоне
        mood_weight=0.3,        # вес награды за настроение
        variety_bonus=3.0,      # бонус за первую пробу нового напитка
        streak_penalty_k=0.75,  # штраф за повтор одного напитка подряд
        alive_bonus=0.05,       # бонус за каждый прожитый шаг
        completion_bonus=6.0,   # бонус за пробование почти всех напитков
    ):
        super().__init__()
        self.m_s = m_sigma_BAC
        self.bias_k = bias_k
        self.noise_k = noise_k
        self.target_bac = target_bac
        self.comfort_sigma = comfort_sigma
        self.comfort_amp = comfort_amp
        self.mood_weight = mood_weight
        self.variety_bonus = variety_bonus
        self.streak_penalty_k = streak_penalty_k
        self.alive_bonus = alive_bonus
        self.completion_bonus = completion_bonus

        self.action_space = spaces.Discrete(6)

        low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([100.0, 1.0, 100.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Внутреннее состояние
        self.BAC = 0.0
        self.BAC_mind = 0.0
        self.hp = 100
        self.time = 1
        self.mood = 5
        self.pred_hp = 100

        # Состояние для бонусов за разнообразие/темп
        self.tried_drinks = set()
        self.last_action = None
        self.streak_len = 0

    def _update_BAC_mind(self):
        """
        Искажённое восприятие BAC:
          - mean смещено вниз относительно истинного BAC, смещение растёт
            квадратично с BAC (недооценка растёт с опьянением);
          - std растёт квадратично с BAC (восприятие всё менее стабильно).
        """
        underestimation = self.bias_k * self.BAC ** 2
        mean = max(0.0, self.BAC - underestimation)
        std = self.m_s + self.noise_k * self.BAC ** 2

        sample = np.random.normal(mean, std)
        self.BAC_mind = np.clip(sample, 0.0, 1.0).astype(np.float32)

    def _get_obs(self):
        return np.array([self.time, self.BAC_mind, self.hp, self.mood], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.BAC = 0.0
        self.hp = 100
        self.time = 1
        self.mood = 5
        self.pred_hp = 100

        self.tried_drinks = set()
        self.last_action = None
        self.streak_len = 0

        self._update_BAC_mind()
        return self._get_obs(), {}

    def step(self, action):
        self.pred_hp = self.hp
        self.time += 1

        # Естественное снижение BAC каждые 2 шага
        if self.time % 2 == 0:
            self.BAC -= 0.01

        is_drink_action = action in (0, 1, 2, 3, 4)

        # Обработка действия
        if action == 0:
            self.BAC -= 0.01
            self.mood -= 1
        elif action == 1:
            self.BAC += 0.02
            self.mood += 1
        elif action == 2:
            self.BAC += 0.05
            self.mood += 2
        elif action == 3:
            self.BAC += 0.12
            self.mood += 3
        elif action == 4:
            self.mood += 0.001
        # action == 5 обрабатывается ниже

        self.BAC = max(self.BAC, 0.0)
        self.mood = max(self.mood, 0.0)
        self.mood = min(self.mood, 10.0)

        # Влияние BAC на здоровье
        if self.BAC > 0.15:
            self.hp -= 1
        if self.BAC > 0.3:
            self.hp -= 1
        if self.BAC > 0.5:
            self.hp -= 3

        self._update_BAC_mind()

        # --- Бонус за разнообразие напитков (первая проба за эпизод) ---
        variety_reward = 0.0
        if is_drink_action and action not in self.tried_drinks:
            self.tried_drinks.add(action)
            variety_reward += self.variety_bonus

        # --- Штраф за спам одного и того же напитка подряд ---
        streak_penalty = 0.0
        if is_drink_action and action == self.last_action:
            self.streak_len += 1
            if self.streak_len >= 2:  # штраф начинается с 3-го повторения подряд
                streak_penalty = self.streak_penalty_k * self.streak_len
        else:
            self.streak_len = 0

        if is_drink_action:
            self.last_action = action

        # --- Комфортная зона BAC (умеренный уровень, не экстремальный) ---
        comfort_reward = self.comfort_amp * np.exp(
            -((self.BAC - self.target_bac) ** 2) / (2 * self.comfort_sigma ** 2)
        )

        # --- Награда за настроение ---
        mood_reward = self.mood_weight * self.mood

        # --- Штраф за потерю hp за этот шаг ---
        hp_penalty = max(0, self.pred_hp - self.hp)

        # --- Небольшой бонус за то, что агент активно взаимодействует ---
        alive_reward = self.alive_bonus

        reward = (
            comfort_reward
            + mood_reward
            - hp_penalty
            + variety_reward
            - streak_penalty
            + alive_reward
        )

        # Терминальная награда: бонус за оставшееся здоровье + бонус за
        # то, что агент успел попробовать почти все напитки за эпизод
        completion = self.completion_bonus if len(self.tried_drinks) >= 4 else 0.0
        term_rew = reward + 0.5 * (100 - self.hp) + completion

        terminated = False
        truncated = False

        if self.hp <= 0:
            terminated = True
            reward = -100.0
        elif self.time > 100:
            truncated = True
            reward = term_rew
        elif action == 5:
            terminated = True
            reward = term_rew

        obs = self._get_obs()
        info = {
            "BAC": self.BAC,
            "tried_drinks": len(self.tried_drinks),
        }

        return obs, reward, terminated, truncated, info