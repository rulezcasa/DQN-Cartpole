







# DQN implementation on Cartpole-v1

## Overview
This codebase implements a Deep Q-Network on the cartpole. - a classic control gymnasium environment.

<img width="204" alt="Screenshot 2025-04-16 at 9 25 49 PM" src="https://github.com/user-attachments/assets/4b89eed2-b5be-429a-ad37-fcc2d6b9cae2" />

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


## Training
<img width="1333" alt="Screenshot 2025-04-16 at 9 26 52 PM" src="https://github.com/user-attachments/assets/a6fe04c1-5e32-43d3-a184-16cd8e502c08" />
