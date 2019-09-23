import numpy as np
import gym
import tensorflow as tf # v2.0
import os
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.client import device_lib

from collections import deque
import matplotlib.pyplot as plt

import cProfile
import pstats

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

class DQN():
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.epsilon = 1.0      # Starting epsilon value
        self.epsilon_min = 0.05 # Final epsilon value
        self.epsilon_decay_frames = 25000 # Amount to subtract each frame
        self.gamma = 0.99       # Future reward discount factor
        self.n_steps = 500      # Max steps per game
        self.n_frames = 60000   # Frames before termination
        self.memory_size = 200000 # Replay memory size (number of transitions to store)
        self.d_min = 10000      # Disable training before collecting minimum number of transitions
        self.plot_update_freq = 20
        self.stack_count = 1

        # Network parameters:
        self.learning_rate = 0.001
        self.n_hidden_nodes = 32
        self.batch_size = 32
        self.model_transfer_freq = 10000 # number of frames between transfer of weights from training network to prediction network

        self.tf_setup()
        self.env = gym.make(self.env_name)
        self.epsilon_decay_value = (self.epsilon - self.epsilon_min)/self.epsilon_decay_frames
        self.n_observations = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n
        self.model_target, self.model = self.build_model()
        self.frame_count = 0
        self.memory = buffer(maxlen=self.memory_size)

    def tf_setup(self):
        #Set up Tensorflow:
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
        Inputs = tf.keras.Input(shape=(self.n_observations,))
        x = Dense(self.n_hidden_nodes, activation="tanh")(Inputs)
        Output = Dense(self.n_actions, activation="linear")(x)
        
        optimizer=tf.optimizers.RMSprop(learning_rate=self.learning_rate) # use default valules

        model_train = tf.keras.Model(inputs=Inputs, outputs=Output)
        model_predict = tf.keras.Model(inputs=Inputs, outputs=Output)

        model_train.compile(optimizer, loss=tf.keras.losses.Huber())        
        model_predict.compile(optimizer, loss=tf.keras.losses.Huber())
        
        model_predict.set_weights(model_train.get_weights())
        
        return model_train, model_predict

    def choose_action(self, state):

        if (self.frame_count > self.d_min) and (self.epsilon > self.epsilon_min):
            self.epsilon -= max(0, self.epsilon_decay_value)

        if np.random.uniform() > self.epsilon:
            input = np.expand_dims(state, axis = 0)
            return np.argmax(self.model.predict(input)).astype(np.int8)
        else:
            return np.random.randint(0, self.n_actions)

    def train_batch(self):

        # Dont train during pre-training phase (pre-fill memory)
        if self.frame_count < self.d_min:
            return 0 

        # Extract minibatch:
        states, actions, rewards, states_new, terminals = self.memory.sample(self.stack_count, self.batch_size)

        # Calculate discounted future reward
        discounted_max_future_reward = self.gamma * np.max(self.model_target.predict(states_new), axis = 1)
        discounted_max_future_reward[np.where(terminals == True)] = 0 # terminal states have no future reward

        targets = self.model.predict(states)
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + discounted_max_future_reward[i]

        history = self.model.fit(states, targets, verbose = 0)

        if self.frame_count % self.model_transfer_freq == 0:
            self.model_target.set_weights(self.model.get_weights())
            self.model.save(f'./cartpole_{self.frame_count}.h5') # Back up progress thus far

        return history.history['loss'][0]

    def plot(self, scores, average_Qs, losses, epsilons, display=False):
    
        f, axes = plt.subplots(2, 2, sharex=True, figsize=(12,6))

        scores_array = np.array(scores)
        losses_array = np.array(losses)
        epsilons_array = np.array(epsilons)
        average_Qs_array = np.array(average_Qs)

        axes[0, 0].plot(scores_array[:, 0], scores_array[:, 1])
        axes[1, 0].plot(losses_array[:, 0], losses_array[:, 1])
        axes[0, 1].plot(epsilons_array[:, 0], epsilons_array[:, 1])
        axes[1, 1].plot(average_Qs_array[:, 0], average_Qs_array[:, 1])
        
        axes[0, 0].set_ylim([0, 510])
        axes[0, 1].set_ylim([0, 1])        
        axes[0, 0].set_ylabel('Average Evaluation Score')
        axes[1, 0].set_ylabel('Loss')
        axes[0, 1].set_ylabel('Epsilon')
        axes[1, 1].set_ylabel('Average Action Value')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 0].set_xlabel('Frame')
        
        plt.savefig('progress.jpg', format='jpg')
        
        if display:
            plt.show()
        else:
            plt.close()

    def evaluate(self, n_games=10, saved_model=None, render = False, epsilon_eval=0.0):

        if saved_model is None:
            play_model = self.model
        else:
            print("Loading model")
            play_model = tf.keras.models.load_model(saved_model)

        scores = []
        average_Qs = []
        for game in range(n_games):

            state = self.env.reset()
            score = 0
            for step in range(self.n_steps):

                if np.random.uniform() > epsilon_eval:
                    input = np.expand_dims(state, axis = 0)
                    Qs = play_model.predict(input)
                    average_Qs.append(np.mean(Qs))
                    action = np.argmax(Qs).astype(np.int8)
                else:
                    action = self.env.action_space.sample()
                observation, reward, done, _ = self.env.step(action)

                score += reward
                state = observation

                if render:
                    self.env.render()

                if done:
                    scores.append(score)
                    break

        return np.mean(scores), np.mean(average_Qs)

    def train(self):

        scores = []
        average_Qs = []
        losses = []
        epsilons = []
        episode = 0
        
        training_done = False
        while not training_done:

            state = self.env.reset()

            self.frame_count += 1

            for _ in range(self.n_steps): # game steps

                action = self.choose_action(state)
                observation, reward, done, _ = self.env.step(action)
                state_new = observation

                self.memory.remember(state, action, reward, state_new, done)
                state = state_new

                # if self.epsilon <= self.epsilon_min:
                #     self.env.render()

                loss = self.train_batch()
                losses.append((self.frame_count, loss))

                self.frame_count += 1

                if self.frame_count == self.n_frames:
                    training_done = True

                if done:
                    episode += 1

                    if episode % self.plot_update_freq == 0:                        
                        score, average_Q = self.evaluate()
                        scores.append((self.frame_count, score))
                        average_Qs.append((self.frame_count, average_Q))
                        epsilons.append((self.frame_count, self.epsilon))
                        self.plot(scores, average_Qs, losses, epsilons)

                    break

        self.model.save(f'./cartpole_{self.frame_count}.h5') # Back up progress thus far
        score, average_Q = self.evaluate()
        scores.append((self.frame_count, score))
        average_Qs.append((self.frame_count, average_Q))
        epsilons.append((self.frame_count, self.epsilon))
        self.plot(scores, average_Qs, losses, epsilons, display=True)
        self.evaluate(n_games=10, saved_model=f'./cartpole_{self.frame_count}.h5', render = True)

if __name__ == "__main__":

    agent = DQN()
    
    # Uncomment to train:
    agent.train()
    
    # Uncomment to evaluate (select correct model to load):
    # agent.evaluate(n_games=10, saved_model='./cartpole.h5', render = True)
