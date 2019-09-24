import numpy as np
import gym
import gym_snake
import tensorflow as tf # v2.0
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv2D
from tensorflow.python.client import device_lib
import os
import time
from datetime import datetime

from collections import deque
import matplotlib.pyplot as plt

import cProfile
import pstats

class DQN():
    def __init__(self, n_frames):

        # ===============================
        # Hyperparameters:
        # ===============================

        self.env_name = 'snake-v0'
        self.epsilon = 1.0          # Starting epsilon value
        self.epsilon_min = 0.05     # Minimum epsilon value
        self.epsilon_decay_frames = 1000000 # Amount to subtract each frame
        self.gamma = 0.8            # Future reward discount factor
        self.n_steps = 200          # Max steps per game
        self.n_frames = n_frames    # Frames before termination
        self.memory_size = 1000000  # Replay memory size (number of transitions to store)
        self.d_min = 10000          # Disable training before collecting minimum number of transitions
        self.plot_update_freq = 100 # IN episodes

        # Network parameters:
        self.learning_rate = 0.0025
        self.conv1filters = 16
        self.conv1kernel = (4, 4)
        self.conv1stride = (2, 2)
        self.conv2filters = 32
        self.conv2kernel = (4, 4)
        self.conv2stride = (2, 2)
        self.n_hidden_nodes = 512
        self.batch_size = 32
        self.model_transfer_freq = 10000 # number of frames between transfer of weights from training network to prediction network

        # environment details:
        self.grid_height = 8
        self.grid_width = 8
        self.unit_size = 5  # Number of pixels of each grid point
        self.unit_gap = 0   # Disable pixels between grid points
        self.n_snakes = 1   # number of snakes (must be one)
        self.n_foods = 3    # number of foods

        self.tf_setup()
        self.env = gym.make(self.env_name)
        self.env.unit_size = self.unit_size
        self.env.unit_gap = self.unit_gap
        self.env.n_snakes = self.n_snakes
        self.env.n_foods = self.n_foods
        self.env.grid_size = (self.grid_height, self.grid_width)

        self.epsilon_decay_value = (self.epsilon - self.epsilon_min)/self.epsilon_decay_frames
        self.observations_shape = self.env.reset().shape
        self.n_actions = self.env.action_space.n
        self.model_target, self.model = self.build_model()
        self.frame_count = 0
        self.memory = buffer(maxlen=self.memory_size)

    def tf_setup(self):
        # Disable verbose ouput from tensorflow:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable Tensorflow log from console
        local_device_protos = device_lib.list_local_devices() # Find GPUs
        GPU_list = [x.physical_device_desc \
                    for x in local_device_protos \
                    if x.device_type == 'GPU']
        if len(GPU_list) > 0:
            print("Available GPUs:")
            for i in GPU_list:
                # parse GPU descriptions into dict:
                dev = {y.split(':')[0].strip():y.split(':')[1].strip() for y in [x.strip() for x in i.split(",")]}
                # print only the parts we want:
                print(dev['device'], '-', dev['name'])
        else:
            print("CAUTION: No available GPUs. Running on CPU.")

    def build_model(self):

        Inputs = tf.keras.Input(shape=(self.observations_shape[0], self.observations_shape[1], 1))
        Normalize = Lambda(lambda x: x/255)(Inputs)
        x = Conv2D(
            filters=self.conv1filters, 
            kernel_size=self.conv1kernel, 
            strides=self.conv1stride,
            padding='valid',
            activation='relu')(Normalize)
        x = Conv2D(
            filters=self.conv2filters, 
            kernel_size=self.conv2kernel, 
            strides=self.conv2stride,
            padding='valid',
            activation='relu')(x)
        x = Flatten()(x)
        x = Dense(self.n_hidden_nodes, activation="tanh")(x)
        Output = Dense(self.n_actions, activation="linear")(x)
        
        optimizer=tf.optimizers.RMSprop(learning_rate=self.learning_rate) # use default valules

        model_train = tf.keras.Model(inputs=Inputs, outputs=Output)
        model_predict = tf.keras.Model(inputs=Inputs, outputs=Output)

        model_train.compile(optimizer, loss=tf.keras.losses.Huber())        
        model_predict.compile(optimizer, loss=tf.keras.losses.Huber())
        
        model_predict.set_weights(model_train.get_weights())
        
        model_predict.summary()
        return model_train, model_predict

    def choose_action(self, state):

        # Update epsilon if necessary
        if (self.frame_count > self.d_min) and (self.epsilon > self.epsilon_min):
            self.epsilon -= max(0, self.epsilon_decay_value)

        # Choose action based on Epsilon
        if np.random.uniform() > self.epsilon:
            input = np.expand_dims(state, axis = 0)
            return int(np.argmax(self.model.predict(input)))
        else:
            return np.random.randint(0, self.n_actions)

    def train_batch(self):

        # Dont train during pre-training phase (pre-fill memory)
        if self.frame_count < self.d_min:
            return 0 

        # Extract minibatch:
        states, actions, rewards, states_new, terminals = self.memory.sample(1, self.batch_size)

        # Calculate discounted future reward
        discounted_max_future_reward = self.gamma * np.max(self.model_target.predict(states_new), axis = 1)
        discounted_max_future_reward[np.where(terminals == True)] = 0 # terminal states have no future reward

        targets = self.model.predict(states)
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + discounted_max_future_reward[i]

        history = self.model.fit(states, targets, verbose = 0)

        if self.frame_count % self.model_transfer_freq == 0:
            self.model_target.set_weights(self.model.get_weights())
            self.model.save(f'./saved_models/snake_{self.frame_count}.h5') # Back up progress thus far

        return history.history['loss'][0]

    def plot(self, scores, average_Qs, losses, epsilons, display=False):
        '''
        Display or save dashboard
        '''
        f, axes = plt.subplots(2, 2, sharex=True, figsize=(12,6))

        scores_array = np.array(scores)
        losses_array = np.array(losses)
        epsilons_array = np.array(epsilons)
        average_Qs_array = np.array(average_Qs)

        axes[0, 0].plot(scores_array[:, 0], scores_array[:, 1])
        axes[1, 0].plot(losses_array[:, 0], losses_array[:, 1])
        axes[0, 1].plot(epsilons_array[:, 0], epsilons_array[:, 1])
        axes[1, 1].plot(average_Qs_array[:, 0], average_Qs_array[:, 1])
        
        # axes[0, 0].set_ylim([-5, 510])
        axes[0, 1].set_ylim([0, 1])        
        axes[0, 0].set_ylabel('Average Evaluation Score')
        axes[1, 0].set_ylabel('Loss')
        axes[0, 1].set_ylabel('Epsilon')
        axes[1, 1].set_ylabel('Average Action Value')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 0].set_xlabel('Episode')
        
        plt.savefig('progress.jpg', format='jpg')
        
        if display:
            plt.show()
        else:
            plt.close() 

    def preprocess(self, observation):
        '''
        Preprocessing involves isolating the snake head, it's body, and the food in the image, 
        then to combine them each with different values to make them distinct.
        The head needs to be distinct so that the system has a sense of which direction it is moving
        (no need to stack frames).
        '''
        food = observation[:, :, 2]         # Use last layer only for food
        snake = 255 - observation[:, :, 1]  # Select middle layer (inverted) for food
        snake[snake == 245] = 200           # Change head value to make it distinct
        snake[np.where(food > 0)] = 0       # Remove food in snake layer
        output = snake*1 + food*0.5         # Compile layers into one, but make snake and food distinct values
        output = np.expand_dims(output, 2)  # Expand dims to match what the network expects
        output = output.astype(np.uint8)    # Save as byte dtype to save memory

        return output
    
    def evaluate(self, n_games=10, saved_model=None, render = False, epsilon_eval=0.0, max_steps=100):
        '''
        inputs:
        n_games: number of episodes on which to evaluate.
        saved_model: Use a specific saved model file. If not supplied, use the current class model (self.model).
        render: boolean value to display the evaluation episodes or not
        epsilon_eval: probability of selecting a random action (set as 0.05 to avoid moving in 
            loops in partially-trained models)
        max_steps: End game after this many steps

        returns: score, frames, average_Q
        score: average game score (sum(score)/n_games)
        frames: Average number of frames for each episode
        average_Q: Average action value for all steps in the game
        '''

        if saved_model is None:
            play_model = self.model
        else:
            play_model = tf.keras.models.load_model(saved_model)

        average_Qs = []
        score = 0
        game_frames = []

        for game in range(n_games):

            state = self.preprocess(self.env.reset())

            step = 0
            while True:

                if np.random.uniform() > epsilon_eval:
                    input = np.expand_dims(state, axis = 0)
                    Qs = play_model.predict(input)
                    average_Qs.append(np.mean(Qs))
                    action = int(np.argmax(Qs))
                else:
                    action = self.env.action_space.sample()
                observation, reward, done, _ = self.env.step(action)

                score += reward
                state = self.preprocess(observation)

                if render:
                    self.env.render()

                step += 1

                if done or (step > max_steps):
                    game_frames.append(step)
                    step = 0
                    break
            
        return score/n_games, np.mean(game_frames), np.mean(average_Qs)

    def train(self):

        '''
        Main training function. This is the part that enacts and observes the markov chain:
        State, action, reward, new state, etc
        '''

        scores = []
        average_Qs = []
        steps = []
        losses = []
        epsilons = []
        episode = 0
        
        start_time = time.time()
        print(f"Starting training on {self.n_frames} game frames at {datetime.now()}")

        training_done = False
        while not training_done:

            # Get first state
            state = self.preprocess(self.env.reset())

            # Advance the frame count
            self.frame_count += 1
            episode_loss = []

            for _ in range(self.n_steps): # game steps

                action = self.choose_action(state)
                observation, reward, done, _ = self.env.step(action)
                state_new = self.preprocess(observation)

                self.memory.remember(state, action, reward, state_new, done)

                state = state_new

                # if self.epsilon <= self.epsilon_min:
                #     self.env.render()

                loss = self.train_batch()
                episode_loss.append(loss)

                self.frame_count += 1

                if self.frame_count >= (self.n_frames+self.d_min):
                    training_done = True

                if done:

                    episode += 1

                    if episode % self.plot_update_freq == 0:                        
                        score, game_frames, average_Q = self.evaluate()
                        scores.append((episode, score))
                        steps.append((episode, game_frames))
                        average_Qs.append((episode, average_Q))
                        epsilons.append((episode, self.epsilon))
                        losses.append((episode, np.mean(episode_loss)))

                        self.plot(scores, average_Qs, losses, epsilons)

                    break

        # Save and plot again at the end of training
        self.model.save(f'./saved_models/snake_{self.frame_count}.h5') # Back up progress thus far
        score, game_frames, average_Q = self.evaluate()
        scores.append((self.frame_count, score))
        steps.append((self.frame_count, game_frames))
        average_Qs.append((self.frame_count, average_Q))
        epsilons.append((self.frame_count, self.epsilon))
        self.plot(scores, average_Qs, losses, epsilons, display=True)
        self.evaluate(n_games=10, saved_model=f'./saved_models/snake_{self.frame_count}.h5', render = True)

        print(f"Training complete at {datetime.now()}")
        print(f"Total time: {time.time() - start_time}")

class buffer():
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
        self.state_memory = deque(maxlen=maxlen)

    def remember(self, state, action, reward, state_, done):
        self.memory.append([state, action, reward, state_, done])

    def sample(self, stack_size, batch_size):
        
        # Extract
        indices = np.random.randint(low = stack_size, high = len(self.memory), size = batch_size)
        minibatch = [self.memory[i] for i in indices]
        transitions = list(zip(*minibatch))
        states      = np.stack(transitions[0], axis=0)
        actions     = np.stack(transitions[1], axis=0)
        rewards     = np.stack(transitions[2], axis=0)
        terminals   = np.stack(transitions[4], axis=0)
        states_new  = np.stack(transitions[3], axis=0)

        return states, actions, rewards, states_new, terminals

if __name__ == "__main__":

    agent = DQN(n_frames=1000000)
    
    # Uncomment to train:
    # ===================
    # agent.train()

    # Uncomment to evaluate (select correct model to load):
    # =====================================================
    score, game_frames, average_Q =agent.evaluate(
        n_games=10,
        saved_model='./saved_models/snake_600000.h5',
        epsilon_eval=0.0,
        render = True,
        max_steps=50)