# Deep Q Reinforcement Learning

## Description
This repo contains my implementation of a Deep Q Network (DQN), which can play simple games using reinforcement learning. 

### Incredibly brief Background on Q learning:
Traditional Q Learning is the approach of building a Q-table to estimate the "value" of each possible action for any given state in the game. The table includes game states as rows, and possible actions as columns. As the game is played, the famous bellman equation is used to update the value of each action in the Q table. As a state is encountered, an action is selected (either at random or using the table itself, depending on the current rate of exploration) and is executed. The reward as well as the resultant game state are observed. The Q value for each action, i.e. the estimate of maximum discounted future reward, is estimated using the bellamn equation and the value in the table is updated. As such, as the game is played, the values of each action for each state are converged upon, allowing the algorithm to select the best action for a given stiuation in the game.

### Deep Q Learning:
The problem with traditional Q learning is that it is only really viable for trivially very small observation spaces and action spaces. That is to say, the number of actions and metrics that get observed (as well as the possible descretized values for each), must be small in order for it to be feasible to build a Q table. For a game like Snake, the observation space is the number of pixels on the game screen, and the Q table would need ot include a row for every possible state that might occur in the game. Every time the snake moves, a row in the able for that exact combination of pixel values needs to exist. Clearly, this is not feasible for traditional Q learning, but fortunately for us there is a solution. 

The Q table is essentially a function approximator. It takes states as inputs, and returns the approximate values of each available action as the output. As the game plays, the approximations of the action values will (hopefully) become more accurate. Instead of using the Q-table, we could instead use a neural network - after all, it is another kind of function approximator with inputs and outputs. But neural networks are mush more meory efficient, and they're very good at picking up on patterns. The exact combination of pixels observed doesn't need too have been encountered before, just something similar-ish. 

To train the network, we will again use the bellman equation. As we play the game and we encounter states, we choose one action, observe the result (reward and resultant state) and add it to the memory so that we can train from this combination (called a transition) multiple times. When training, we select a mini-batch of transitions from memory. For each transition in this mini-batch, we feed both the starting and resulting states into our neural network. We use the results - what we predicted and what we actually observed - to update the values of our the original predictions, and thenuse these to build the target for training the model.

Repeat this a million times, updating the model with each state to fit data from memory as we collect it, and you (hopefully) have a model that can play the video game!

## Dependencies
- Gym
- Gym_snake
- Matplotlib
- Tensorflow V2.0
- Numpy

## Usage:
Clone this repository. Open DQN_snake.py, scroll to the bottom and uncomment the various options (i.e. to train the model or just play it), and execute the file. To edit hyperparameters, change the class variables in thier definition near the top of the file

## TO DO:
- The current model works, but it quite inefficient with memory. Each state (bitmap) is stored twice - once as the resulting state of one transition, and then again as the starting state of the next. I would like to implement a circular buffer system with indexing to allow storing of each frame only once.

- Include argparse to build a CLI, and remove the need for editing the file to change a parameter

- Move hyperparameters to a separate file, save in JSON format

- Improve the model to use Double-DQN, Dueling DQN, etc.
 
