







# DQN implementation on Cartpole-v1

## Overview
This codebase implements a Deep Q-Network on the cartpole. - a classic control gymnasium environment.

### CartPole-v1 Environment Details

| Property           | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| **Action Space**    | `Discrete(2)` – 2 possible actions (e.g., move left or right)         |
| **Observation Space** | `Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32)` |
| **Import**          | `gymnasium.make("CartPole-v1")`    
| Max_reward          | `500`    



## Optimal hyperparameters

| Parameter         | Value     |
|------------------|----------|
| **epsilon_start**  | 0.9      |
| **epsilon_end**    | 0.5      |
| **decay_steps**    | 80000    |
| **replay_buffer**  | 100000   |
| **warmup**         | 5000     |
| **gamma**          | 0.99     |
| **device**         | "mps"    |
| **target_update**  | 1000     |
| **learning rate (lr)** | 0.00007  |
| **T (Total Timesteps)** | 200000   |