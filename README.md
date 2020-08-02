# Collaboration-Competition-Udacity-Deep-RL-project
MADDPG for Unity ML-Agents Tennis environment

Install
--------------------------------------------------------------------------------
We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup the environment,
- and python 3.7

Setup our environment:
```bash
conda --version

# Clone the repo
git clone https://github.com/KhalilGorsan/Collaboration-Competition-Udacity-Deep-RL-project.git
cd Collaboration-Competition-Udacity-Deep-RL-project

# Create a conda env
conda env create -f environment.yml

source activate deeprl_udacity

# Install pre-commit hooks
pre-commit install
```
Don't forget to add The Tennis.app unity environment in the root of the project.

To install an already built environment for you, you can download it from one
of the links below. You need only to select the environment that matches your operating
system and unzip it:

Version 1: One (1) Agent
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Environment
--------------------------------------------------------------------------------
For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![](https://video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of **8** variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of **+0.5** (over **100** consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least **+0.5**.

Training
--------------------------------------------------------------------------------
To train you _`maddpg`_ agent on the tennis environment, you can run this code
```bash
python train.py
```

After the train is completed, it will provide you the checkpoints of the **actor** and **critic** when the environment is solved as well as a plot of the rewards.