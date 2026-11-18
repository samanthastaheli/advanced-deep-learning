# CS 674 Final Project Proposal

*Samantha Staheli*

## What is the problem you're trying to solve? Why is it interesting? What has been done / what is the state-of-the-art?

The problem I am trying to solve is creating and improving a Reinforcement Learning (RL) agents in video games. The video game I will be using is a farming simulator. Meaning a player will grow and harvest crops with time, energy level, and money restraints. Similar to the popular Stardew Valley game. Claude made a simple starting point of a simple farming simulator written in Python. I will still need to workout the details, but Claude created a good starting point.


## How will deep learning be used to solve the problem?

A neural network will be used to create the RL agent. I will make a few different types, such as Q-learning, Minimax, and Multi-agent Arm Bandits, and see which one does the best. 

## What data is required? Is it already available, or do you need to collect it?

There is no predefined dataset that is required for RL agents. RL agents create their own dataset in real time using their functions. 

The simulator is a Python Gymnasium environment. Claude built a starting point environment that requires some adjustments to make it usable for the agent I will create. 

## How will you train your DNN?

Training happens in real time during the farm simulation. A RL agent will gather information and utilize it through reasoning. This reasoning will be probability functions like Q-learning, Minimax, and Multi-agent Arm Bandits.  