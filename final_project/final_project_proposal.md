# CS 674 Final Project Proposal

*Samantha Staheli*

## Project Title

Learning to Farm: Comparing Neural Network Architectures and Reinforcement Learning Algorithms

## What is the problem you're trying to solve? Why is it interesting? What has been done / what is the state-of-the-art?

The problem I am trying to solve is creating and improving a Reinforcement Learning (RL) agent in video games. The video game I will be using is a farming simulator. Meaning a player will grow and harvest crops with time, energy level, and money restraints. Similar to the popular Stardew Valley game. Claude made a simple starting point of a simple farming simulator written in Python. I will still need to workout the details, but Claude created a good starting point.

Currently there are state of the art agents such as 

## How will deep learning be used to solve the problem?

Convolutional neural network (CNN) and multilayer perceptron (MLP) are architecture types that I will be implementing. Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) are reinforcement learning approaches that I will use to solve the problem. The question I am asking is, which policy, CNN-DQN, MLP-DQN, CNN-PPO, or MPL-PPO performs the best given a farm grid?

My hypothesis is that the CNN policy will make better use of the grid’s spatial structure and will retain more of its performance when obstacle layouts differ from those seen during training.

## What data is required? Is it already available, or do you need to collect it?

There is no predefined dataset that is required for RL agents. RL agents create their own dataset in real time using their functions. 

The simulator is a Python Gymnasium environment. Claude built a starting point environment that requires some adjustments to make it usable for the agents I will create. 

## How will you train your DNN?

Training happens in real time during the farm simulation. A RL agent will gather information and utilize it through reasoning. This reasoning will be policy functions like Q-learning, CNN, and MLP.  