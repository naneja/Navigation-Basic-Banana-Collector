# Project 1: Navigation

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

## Introduction

This project trains an agent to navigate (and collect bananas!) in a large, square world.  


## Project Details

* State Space consist of 37 dimension values as below. The 37 dimensions include agent's velocity, along with ray-based perception of objects around agent's forward direction

[1.         0.         0.         0.         0.84408134 0.
0.         1.         0.         0.0748472  0.         1.
0.         0.         0.25755    1.         0.         0.
0.         0.74177343 0.         1.         0.         0.
0.25854847 0.         0.         1.         0.         0.09355672
0.         1.         0.         0.         0.31969345 0.
0.        ] 

* Action Space four actions as below:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

* Rewards
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  


* Requirement of Solution
The task is episodic, and in order to solve the environment, agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

* Dependencies: Download the following dependencies and keep the uncompress files in the config folder:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## Instructions

Run all cells of `Navigation.ipynb` to get started with training your own agent!  Once the agent has been trained, you may set train to False in the Train Agent cell to re-run all cells without training. 

