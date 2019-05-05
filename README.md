# MultiAgent
[//]: # (Image References)

[image1]: https://github.com/OlaAhmad/MultiAgent/blob/master/image.png "Tennis env"
[image2]: https://github.com/OlaAhmad/MultiAgent/blob/master/scores.png "scores"

# Project 3: Collaboration and Competition 

### Introduction

![Tennis env][image1]

In this environment, two agents play tennis table. The agents control their rackets to bounce a ball over the net. Every agent hits the ball over the net receives a reward of +0.1, and a reward -0.01 if the agent lets the ball hits the ground or hits the ball out of the bounds. The goal of every agent is to maintain the ball in play.

The observation space consists of 8 variables corresponding to position and velocity of the ball and the racket. Each agent receives its own local observation of the environment, and has two actions available corresponding to movement toward (or away from) the net and jumping. 

The task is episodic. To solve the enviornment, the agents must get an average score of at least +0.5 over 100 consecutive episodes after taking the maximum score over both agents. 

### Getting started

Download the MultiAgent repository from the top-right button. You can also clone the repository and downloaded from a terminal in your workspace directory using the following command line:
    
    git clone https://github.com/OlaAhmad/MultiAgent.git
        
### Usage

To train and test the agents, go to the MultiAgent folder and open the Tennis.ipynb notebook:

    cd MultiAgent
    Jupyter notebook Tennis.ipynb

When runing the notebook, the actor-critic agent will start training over a number of episodes; Two neural networks will start training and simultaneously update their parameters every number of iterations. The updated parameters of the trained acrchitectures are saved in checkpoint files

![scores][image2]

### Codes

The repository contains the following codes: 
1. model.py: builds actor and critic neural network architectures. 
2. MADDPG.py: builds the multi agents class based on the DDPG algorithm.
3. Tennis.ipynb: notebook to train DDPG multi agents over 3000 episodes.
4. Trained actor and critic networks saved in .pth formate. 

### Resources

* udacity/deep-reinforcement-learning
