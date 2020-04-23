# DQN_RL_maze

This repository contains the 2nd Reinforcement Learning coursework from the Department of Computing, Imperial College London, Academic Year 2019-2020, delivered by Dr A. Aldo Faisal and Dr Edward Johns. The coursework was developed with their PhD students. <br>

This project consisted of building an DQN implementation to solve a maze problem. <br>
The random_environment.py and train_and_test.py files were given. <br>
The random_environment.py file creates random maze environments. <br>
The train_and_test.py creates an environment, trains the agent 10 minutes, and then tests it against a new maze.

Important features of the DQN implementation in the agent.py file comprise of: <br>
- Epsilon-greedy policy (lines 95-99)
- Epsilon decay with steps before being clipped at 0.005 (lines 111-117)
- Experience replay buffer (lines 279-334)
- Prioritised experience replay (lines 313-318)
- Decaying episode length with number of steps (lines 69-78)
- Reward function which penalises hitting the wall proportionally to the distance from the goal; gives 0 reward for moving vertically; rewards positively when entering a circle close to the reward state (lines 142-156)


An example of a **simple** maze, where the red dot is the agent and the green dot the goal:

<img src="images/maze_simple_example.png" width=500>
