# BarEnv-v1

A custom [Gymnasium](https://gymnasium.farama.org/) environment that simulates a night out at a bar. An agent chooses drinks over time while managing intoxication (BAC), health, and mood — under a perception of its own drunkenness that is deliberately biased and unreliable, the way a real drinker's is.

This version (`v1`) replaces `v0`'s "maximize BAC" incentive structure with a reward function designed to make the agent balance multiple competing goals, in a way that loosely mirrors how people actually reason about drinking.

## Overview

| | |
|---|---|
| **Environment ID** | `BarEnv-v1` (class `BarEnv_v1`) |
| **Action space** | `Discrete(6)` |
| **Observation space** | `Box(4,)`, `float32` |
| **Episode length** | up to 100 steps (auto-truncated) |
| **Render modes** | none |

## Installation

```bash
pip install gymnasium numpy
```

```python
import gymnasium as gym
from gymnasium.envs.registration import register

register(id="BarEnv-v1", entry_point="bar_env_v1:BarEnv_v1")
env = gym.make("BarEnv-v1")
```

## Action Space

`Discrete(6)`:

| Action | Drink | Effect on BAC | Effect on mood |
|:---:|---|---|---|
| `0` | Water | −0.01 | −1 |
| `1` | Beer | +0.02 | +1 |
| `2` | Wine | +0.05 | +2 |
| `3` | Vodka | +0.12 | +3 |
| `4` | Non-alcoholic drink | 0 | +0.001 |
| `5` | Leave the bar | — | — (ends episode) |

## Observation Space

`Box(low=[0,0,0,0], high=[100,1,100,10], dtype=float32)`:

| Index | Variable | Range | Meaning |
|:---:|---|---|---|
| 0 | `time` | 0–100 | Current step count |
| 1 | `BAC_mind` | 0–1 | The agent's **perceived** BAC — biased and noisy, not the true value |
| 2 | `hp` | 0–100 | Current health points |
| 3 | `mood` | 0–10 | Current mood |

The true `BAC` is tracked internally but **never observed directly** — only through `BAC_mind`.

---

## Why the agent doesn't see its real BAC

A real person doesn't have a readout of their blood alcohol content — they estimate it from how they feel, and that self-estimate gets *less* accurate the more they've had to drink. This is a well-documented effect: alcohol impairs the same self-monitoring and interoceptive faculties that would otherwise let someone judge "I've had enough." People tend to underestimate their own impairment as intoxication increases, which is part of why heavy drinking so often escalates past the point the drinker intended.

`BAC_mind` is built to mirror that:

```python
underestimation = bias_k * BAC**2      # grows quadratically with true BAC
mean = max(0.0, BAC - underestimation) # perceived BAC is biased low
std  = m_s + noise_k * BAC**2          # spread also grows quadratically
BAC_mind = clip(Normal(mean, std), 0, 1)
```

- **At low true BAC**, `mean ≈ BAC` and `std` is tiny — a sober or lightly buzzed agent perceives its state fairly accurately, just like a person after one drink usually still has a reasonably clear head.
- **At high true BAC**, the perceived value is pulled noticeably below the truth, and the noise around it widens a lot — the agent *thinks* it's less drunk than it is, and its read on its own state becomes unreliable. This is the mechanism that makes over-drinking dangerous in the simulation (and in reality): the more you need an accurate signal to stop, the worse a signal you get.

This turns the environment into a genuine partially-observable problem where the cost of misjudging your own state scales with how far you've already gone.

## Reward Function

```python
reward = comfort_reward + mood_reward - hp_penalty + variety_reward - streak_penalty + alive_reward
```

| Term | Formula | What it represents |
|---|---|---|
| `comfort_reward` | `comfort_amp * exp(-(BAC - target_bac)² / (2·comfort_sigma²))` | A Gaussian centered on a **moderate** BAC (default `0.06`), not an extreme one |
| `mood_reward` | `mood_weight * mood` | Alcohol's genuine, real short-term mood/social payoff |
| `hp_penalty` | `max(0, pred_hp - hp)` | Physiological cost of intoxication |
| `variety_reward` | `+variety_bonus` the first time a drink type is tried in an episode | Encourages trying different drinks rather than one repeated strategy |
| `streak_penalty` | `streak_penalty_k * streak_len` after 3+ identical drinks in a row | Discourages "chug the same thing repeatedly" policies |
| `alive_reward` | small constant per step | Encourages using the available time instead of rushing to leave |

At episode end (`truncated`, or the agent chooses to leave):

```python
term_rew = reward + 0.5 * (100 - hp) + completion_bonus
```

where `completion_bonus` is added if the agent tried at least 4 of the 5 drink types during the episode. If `hp <= 0`, the step reward is fixed at `-100.0` regardless of everything else.

### Why the reward is shaped this way — the real-world logic

**1. The comfort zone sits at a moderate BAC, not the maximum.**
In `v0`, reward peaked at `BAC = 0.5` — a level associated with real medical danger (risk of loss of consciousness, respiratory depression). That created an incentive that pointed straight at the worst outcome. In `v1`, the target (`0.06`) sits below where health penalties even begin (`0.15`), roughly in the range many people associate with a mild, sociable "buzz" rather than heavy impairment. The Gaussian shape means reward drops off in **both** directions: staying stone-sober isn't rewarded much either, and pushing past the target starts costing reward well before it starts costing health. This is meant to capture something real — that people don't drink to maximize blood alcohol, they drink for a specific pleasant effect, and that effect has a peak, not a monotonic upward slope.

**2. Health loss is an explicit, direct penalty.**
This is the most literal piece of realism: excessive drinking has real physiological costs (organ strain, poisoning risk, injury risk, blackouts), and the environment prices that in every single step, not just at the end. Because `hp_penalty` and `comfort_reward` push in different directions once BAC climbs, the agent is structurally forced to trade one off against the other — it can't get both by just cranking BAC up.

**3. Mood is rewarded, but only lightly.**
Alcohol does have a real, immediate mood/social payoff — that's why people drink at all — so it's included as a positive term. But it's weighted much lower than `comfort_reward`, so mood alone can't justify heavy drinking; it functions as a secondary consideration, not the dominant driver.

**4. Variety and pacing bonuses model something more behavioral than physiological.**
People at a bar don't usually optimize a single number — they order different drinks, pace themselves over an evening, and treat the *experience* as the goal, not the destination BAC. The `variety_reward` and `streak_penalty` encode that as an incentive: repeating one action over and over is discouraged, sampling different drinks (including non-alcoholic ones) is rewarded. This also happens to be good RL hygiene — it stops the agent from collapsing onto a single-action policy — but the underlying idea (that binge-repeating one drink is a worse "experience" than a varied, paced evening) is a deliberate design choice, not just a training trick.

**5. The alive/completion bonuses reward sticking around and engaging, not rushing to the exit.**
Without them, the safest policy is often "act once, then immediately leave" — technically optimal, but not an interesting evening and not an interesting policy to study. These bonuses nudge the agent toward actually using the time it has, which is also what makes the variety and pacing incentives meaningful (an agent that leaves after step 2 never gets the chance to explore drink types).

**Net effect:** there is no longer a single dominant strategy ("always drink the strongest thing"). The optimal policy has to sit near a moderate BAC, keep mood reasonably high, avoid triggering health penalties, and spend the episode trying a spread of actions rather than repeating one — which is a much closer analogue to how a person might actually navigate a real evening of drinking than a pure BAC-maximizer would be.

---

## Environment Dynamics (per `step`)

1. `time` advances by 1.
2. Natural BAC decay: every 2 steps, `BAC -= 0.01` (metabolism, independent of action).
3. The chosen drink's effect is applied to `BAC` and `mood`.
4. `BAC` is floored at 0; `mood` is clipped to `[0, 10]`.
5. Health penalties from intoxication (cumulative, thresholds stack):
   - `BAC > 0.15` → `hp -= 1`
   - `BAC > 0.3` → `hp -= 1` (additional)
   - `BAC > 0.5` → `hp -= 3` (additional)
6. `BAC_mind` is resampled from the new true `BAC` (see above).
7. Variety/streak/comfort/mood/hp/alive terms are combined into the step reward.

## Episode Termination

| Condition | Result | Reward used |
|---|---|---|
| `hp <= 0` | `terminated = True` | `-100.0` (fixed penalty) |
| `time > 100` | `truncated = True` | `term_rew` (includes completion bonus) |
| `action == 5` (leave bar) | `terminated = True` | `term_rew` (includes completion bonus) |
| otherwise | episode continues | standard `reward` formula |

## Constructor Parameters

```python
BarEnv_v1(
    m_sigma_BAC=1e-5,
    bias_k=0.6,
    noise_k=1.2,
    target_bac=0.06,
    comfort_sigma=0.05,
    comfort_amp=40.0,
    mood_weight=0.3,
    variety_bonus=3.0,
    streak_penalty_k=0.75,
    alive_bonus=0.05,
    completion_bonus=6.0,
)
```

| Parameter | Default | Description |
|---|---|---|
| `m_sigma_BAC` | `1e-5` | Noise floor for `BAC_mind` even at `BAC = 0` |
| `bias_k` | `0.6` | Strength of the underestimation bias (scales with `BAC²`) |
| `noise_k` | `1.2` | Strength of perception noise growth (scales with `BAC²`) |
| `target_bac` | `0.06` | BAC level that maximizes `comfort_reward` |
| `comfort_sigma` | `0.05` | Width of the comfort-zone Gaussian |
| `comfort_amp` | `40.0` | Peak value of `comfort_reward` |
| `mood_weight` | `0.3` | Weight on the mood term |
| `variety_bonus` | `3.0` | Reward for trying a new drink type for the first time in an episode |
| `streak_penalty_k` | `0.75` | Penalty scale for repeating the same drink 3+ times in a row |
| `alive_bonus` | `0.05` | Flat per-step reward for staying engaged |
| `completion_bonus` | `6.0` | Terminal bonus for trying ≥4 of the 5 drink types |

## Usage Example

```python
import numpy as np
from bar_env_v1 import BarEnv_v1

env = BarEnv_v1()
obs, info = env.reset()

done = False
total_reward = 0.0

while not done:
    action = env.action_space.sample()  # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Episode finished. Total reward: {total_reward:.2f}, drinks tried: {info['tried_drinks']}")
```

## Limitations & Caveats

- This is a **toy simulation for RL research/education**, not a physiological or medical model. The BAC thresholds, decay rate, and reward weights are illustrative approximations, not clinically validated values, and nothing here should be read as guidance about real drinking.
- `bias_k`, `noise_k`, `target_bac`, and the various weights are all tunable — the specific numbers encode a particular design intent (moderate drinking > extreme drinking), but the relative balance between terms is worth sweeping/tuning for your own training setup.
