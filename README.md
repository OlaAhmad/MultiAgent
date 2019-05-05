# MultiAgent
[//]: # (Image References)

[image1]: https://github.com/OlaAhmad/MultiAgent/blob/master/image.png "image"

# Project 3: Collaboration and Competition 

### Introduction

![image][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. The goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Getting started

Download the Continous_Control repository from the top-right button. You can also clone the repository and downloaded from a terminal in your workspace directory using the following command line:
    
    git clone https://github.com/OlaAhmad/MultiAgent.git
        
### Usage

Go to the Continuous_Control folder and open the notebook to train the agent as follows:

    cd MultiAgent
    Jupyter notebook Tennis.ipynb

When runing the notebook, the actor-critic agent will start training over a number of episodes; Two neural networks will start training and simultaneously update their parameters every number of iterations. The updated parameters of the trained acrchitectures are saved in checkpoint files

### Codes

I added two files to train the agent for the notebook: 
1. model.py: builds actor and critic neural network architectures. 
2. dqn_agent.py: interacts with Banana enviornement and learns the agent from it.
The code was adapted from the lesson (ddpg-pendulum).

### Resources

* udacity/deep-reinforcement-learning
