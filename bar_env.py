import numpy as np
import gymnasium as gym
from gymnasium import spaces

class BarEnv_v0(gym.Env):
    """
    Кастомная среда "Бар".
    Действия:
        0: вода        → mood -1, BAC -0.01
        1: пиво        → mood +1, BAC +0.02
        2: вино        → mood +2, BAC +0.05
        3: водка       → mood +3, BAC +0.12
        4: безалког.   → mood +0.001, BAC без изменений
        5: выйти из бара → терминальное состояние (terminated)
    """
    metadata = {"render_modes": []}  # без визуализации

    def __init__(self, m_sigma_BAC=1e-5):
        super().__init__()
        self.m_s = m_sigma_BAC

        # Пространство действий: 6 дискретных вариантов (0..5)
        self.action_space = spaces.Discrete(6)

        # Пространство наблюдений: 4 непрерывные переменные
        # time: 0..100, BAC_mind: 0..1, hp: 0..100, mood: 0..10
        low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([100.0, 1.0, 100.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Внутреннее состояние
        self.BAC = 0.0
        self.BAC_mind = 0.0
        self.hp = 100
        self.time = 1
        self.mood = 5
        self.pred_hp = 100  # для расчёта награды

    def _update_BAC_mind(self):
        """Генерирует искажённое восприятие BAC (BAC_mind) с помощью усечённого нормального распределения."""
        mean = self.BAC
        std = np.sqrt(self.BAC) + self.m_s
        # Используем numpy, обрезаем до [0, 1]
        sample = np.random.normal(mean, std)
        self.BAC_mind = np.clip(sample, 0.0, 1.0).astype(np.float32)

    def _get_obs(self):
        """Возвращает наблюдение в виде массива float32."""
        return np.array([self.time, self.BAC_mind, self.hp, self.mood], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Сбрасывает среду в начальное состояние."""
        super().reset(seed=seed)  # инициализирует генератор случайных чисел

        self.BAC = 0.0
        self.hp = 100
        self.time = 1
        self.mood = 5
        self.pred_hp = 100
        self._update_BAC_mind()

        return self._get_obs(), {}  # observation, info

    def step(self, action):
        """Выполняет действие, обновляет состояние и возвращает (obs, reward, terminated, truncated, info)."""
        self.pred_hp = self.hp
        self.time += 1

        # Естественное снижение BAC каждые 10 шагов
        if self.time % 2 == 0:
            self.BAC -= 0.01

        # Обработка действия
        if action == 0:          # вода
            self.BAC -= 0.01
            self.mood -= 1
        elif action == 1:        # пиво
            self.BAC += 0.02
            self.mood += 1
        elif action == 2:        # вино
            self.BAC += 0.05
            self.mood += 2
        elif action == 3:        # водка
            self.BAC += 0.12
            self.mood += 3
        elif action == 4:        # безалкогольное
            self.mood += 0.001
        # action == 5 -> выход, обрабатывается позже

        # Ограничения значений
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

        # Обновляем искажённое восприятие BAC
        self._update_BAC_mind()

        # Расчёт награды
        reward = (80 * np.exp(-100 * (self.BAC - 0.5)**2) +
                  0.2 * self.mood -
                  max(0, self.pred_hp - self.hp))

        # Награда за завершение (используется, если эпизод заканчивается)
        term_rew = reward + 0.5 * (100 - self.hp)

        # Проверка терминальных условий
        terminated = False
        truncated = False

        if self.hp <= 0:
            terminated = True
            reward = -100.0  # штраф за смерть
        elif self.time > 100:
            truncated = True
            reward = term_rew
        elif action == 5:  # выход из бара
            terminated = True
            reward = term_rew

        # В остальных случаях продолжаем
        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info