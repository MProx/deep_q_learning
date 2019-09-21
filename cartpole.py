import numpy as np
import gym
import tensorflow as tf # v2.0
import random
import itertools
import os
from tensorflow.keras.layers import Dense
from tensorflow.python.client import device_lib
from collections import deque
import matplotlib.pyplot as plt

class buffer():
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
        self.state_memory = deque(maxlen=maxlen)

    def remember(self, action, reward, state_, done):
        self.memory.append([action, reward, done])
        self.state_memory.append(state_)

    def sample(self, stack_size, batch_size):
        indices = np.random.randint(low = stack_size, high = len(self.memory), size = batch_size)
        minibatch = [self.memory[i] for i in indices]
        states      = np.stack(axis=0, arrays=[self.state_memory[i-1] for i in indices])
        states_new  = np.stack(axis=0, arrays=[self.state_memory[i] for i in indices])
        actions     = np.stack(axis=0, arrays=[a for a, r, t in minibatch])
        rewards     = np.stack(axis=0, arrays=[r for a, r, t in minibatch])
        terminals   = np.stack(axis=0, arrays=[t for a, r, t in minibatch])

        return states, actions, rewards, states_new, terminals

class DQN():
    def __init__(self):
        self.env_name = "CartPole-v1"
        self.epsilon = 1.0 # Starting value
        self.epsilon_min = 0.00 # Final value
        self.epsilon_decay_frames = 40000 # Amount to subtract each frame
        self.n_steps = 500
        self.n_episodes = 20000
        self.memory_size = 100000
        self.obs_max = [1.5, 1, 0.41887903, 1]     # Approximate upper lim for observations (used for normalizing)
        self.obs_min = [-1.5, -1, -0.41887903, -1] # Approximate upper lim for observations (used for normalizing)
        self.batch_size = 126
        self.d_min = 10000
        self.gamma = 0.85
        self.plot_update_freq = 20
        self.model_transfer_freq = 100 # number of frames between transfer of weights from training network to prediction network
        self.img_stack_count = 1

        self.tf_setup()
        self.epsilon_decay_value = (self.epsilon - self.epsilon_min)/self.epsilon_decay_frames
        self.env = gym.make(self.env_name)
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
        Inputs = tf.keras.Input(shape=(self.n_observations,), name='img_input')
        H1 = Dense(256, activation="relu")(Inputs)
        H2 = Dense(256, activation="relu")(H1)
        Output = Dense(self.n_actions, activation="linear")(H2)
        
        # optimizer=tf.optimizers.Adam(lr=self.learning_rate, decay=self.learning_rate_decay)
        optimizer=tf.optimizers.RMSprop(learning_rate=5E-6, rho=0.95) # use default valules

        model_train = tf.keras.Model(inputs=Inputs, outputs=Output)
        model_train.compile(optimizer, loss=tf.keras.losses.Huber())
        
        model_predict = tf.keras.Model(inputs=Inputs, outputs=Output)
        model_predict.compile(optimizer, loss=tf.keras.losses.Huber())
        
        model_predict.set_weights(model_train.get_weights())
        
        return model_train, model_predict

    def preprocess(self, observation):
        return [(obs - self.obs_min[i])/(self.obs_max[i] - self.obs_min[i]) for i, obs in enumerate(observation)]
            
    def episode_setup(self):

        observation = self.env.reset()
        state = self.preprocess(observation)
        return state

    def choose_action(self, state):

        if (self.frame_count > self.d_min) and (self.epsilon > self.epsilon_min):
            self.epsilon -= max(0, self.epsilon_decay_value)

        if np.random.uniform() > self.epsilon:
            input = np.expand_dims(state, axis = 0)
            return np.argmax(self.model.predict(input)).astype(np.int8)
        else:
            # Random policy
            return np.random.randint(0, self.n_actions)

    def train_batch(self):

        if self.frame_count < self.d_min:
            return 0

        # Extract minibatch (never select index 0 - there must always be one previous):
        states, actions, rewards, states_new, terminals = self.memory.sample(self.img_stack_count, self.batch_size)

        # Calculate discounted future reward
        discounted_max_future_reward = self.gamma * np.max(self.model_target.predict(states_new), axis = 1)
        discounted_max_future_reward[np.where(terminals == True)] = 0 # terminal states have no future reward

        targets = self.model.predict(states)

        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + discounted_max_future_reward[i]

        history = self.model.fit(states, targets, verbose = 0)

        if self.frame_count % self.model_transfer_freq == 0:
            self.model_target.set_weights(self.model.get_weights())

        return history.history['loss'][0]

    def plot(self, frames, scores, losses, epsilons, display=False):
        # find average of scores:
        average_scores = [scores[0]]
        for s in scores[1:]:
            average_scores.append(average_scores[-1]*0.9 + s*0.1)

        # plot:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(frames, scores, 'o', markersize=1)
        ax1.plot(frames, average_scores, linewidth=1, color='red')
        ax2.plot(frames, losses)
        ax3.plot(frames, epsilons)
        ax1.set_ylim([0, 510])
        ax3.set_ylim([0, 1])
        ax1.set_ylabel('Score')
        ax2.set_ylabel('Loss')
        ax3.set_ylabel('Epsilon')
        ax3.set_xlabel('Frame')
        plt.savefig('progress.jpg', format='jpg')
        if display:
            plt.show()
        else:
            plt.close()


    def run(self):

        losses = []
        scores = []
        epsilons = []
        frames = []

        for episode in range(self.n_episodes):

            state = self.episode_setup()
            episode_score = 0
            episode_losses = []

            self.memory.remember(0, 0, state, False)
            self.frame_count += 1

            for _ in range(self.n_steps): # game steps

                action = self.choose_action(state)

                observation, reward, done, _ = self.env.step(action)
                state_new = self.preprocess(observation)

                self.memory.remember(action, reward, state_new, done)

                # if self.epsilon <= self.epsilon_min:
                #     self.env.render()

                loss = self.train_batch()

                episode_losses.append(loss)
                self.frame_count += 1
                episode_score += reward

                state = state_new

                if done:
                    
                    frames.append(self.frame_count)
                    scores.append(episode_score)
                    losses.append(np.mean(episode_losses))
                    epsilons.append(self.epsilon)

                    if episode % self.plot_update_freq == 0:
                        print("episode:", episode, "F: ", self.frame_count, "E:", round(self.epsilon, 2), "S:", episode_score, "L:", round(np.mean(episode_losses), 2))
                        # save plot every 10th episode
                        # (overwrite previous plot)
                        self.plot(frames, scores, losses, epsilons)
                    break

        self.model.save(f'./cartpole.h5') # Back up progress thus far
        self.plot(frames, scores, losses, epsilons, display=True)

if __name__ == "__main__":

    agent = DQN()

    agent.run()
