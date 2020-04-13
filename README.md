# DeepRL-Tennis
Multi-agent Tennis game play. Udacity Deep Reinforcement Learning Nanodegree 3rd Project.

## Project overview
In this project we implement a reinforcement learning two agents using [Multi-agent Deep Deterministic Policy Gradients (MADDPG)](https://arxiv.org/abs/1706.02275) algorithm. The agents are required to learn how to play a tennis game. The world environment, where the agents play, is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents) and the environment looks as in the figure below.

![img_1](Figures/player.gif)

## Environment details 
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Each agent recieves three stacked observations at every step, and hence the total state space dimensions for each
agent are 24. Two continuous actions are available for each agent, corresponding to movement toward (or away from) the net, and jumping. Every entry in the action vector should be a number between -1 and 1.

To learn the optimal policy, if an agent hits the ball over the net, it receives a reward of +0.1. On the other hand, if an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

## Installation and Running 

1. Configure a Python 3.6 / PyTorch 0.4.0 environment with the needed requirements as described in the [Udacity repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
      
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)
  
3. Download the `Tennis.ipynb`, `model.py` and `ddpg_agents.py` files in the DeepRL-Tennis GitHub repository and make them accessible to your python environment.   

## Training
Execute the provided notebook within Deep Reinforcement Learning Udacity online workspace for collap-compet project after modifying `Tennis.ipynb` and related files.
