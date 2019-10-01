import numpy as np
import tensorflow as tf # v2.0
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv2D
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import gym
import gym_snake
import os
import time
from datetime import datetime
import argparse

class DQN():
    def __init__(self, mode):

        # ===============================
        # Hyperparameters:
        # ===============================

        env_name = 'snake-v0'
        self.epsilon = 1.0                    # Starting epsilon value
        self.epsilon_min = 0.1                # Minimum epsilon value
        self.epsilon_decay_frames = 1000000   # number of frames to decay linearly from epsilon to epsilon_min
        self.gamma = 0.9                      # Future reward discount factor (Bellman equation)
        self.n_steps = 200                    # Max frames per game before automatic termination (quit endless loops)
        self.memory_size = 1000000            # Replay memory size (number of transitions to store)
        self.d_min = 10000                    # Disable training before collecting minimum number of transitions
        self.eval_freq = 100                  # Frequency, in episodes, that evaluation data is pushed to tensorboard
        self.img_stack_count = 1              # Number of images to stack in each input state

        # Network parameters:
        self.learning_rate = 0.001            # For RMSProp optimizer
        self.conv1filters = 16      
        self.conv1kernel = (4, 4)
        self.conv1stride = (2, 2)
        self.conv2filters = 32
        self.conv2kernel = (4, 4)
        self.conv2stride = (2, 2)
        self.n_hidden_nodes = 512
        self.batch_size = 32                  # Number of samples in training mini-batch
        self.model_transfer_freq = 10000      # number of frames between transfer of weights from training network to prediction network
        self.log_dir = "./logs/"              # Directory for tensorboard logs

        # Environment details:
        self.grid_height = 10
        self.grid_width = 10
        self.unit_size = 5                    # Number of pixels of each grid point
        self.unit_gap = 0                     # Disable pixels between grid points
        self.n_snakes = 1                     # number of snakes (must be one)
        self.n_foods = 1                      # number of foods (possibly greater than 1 during training to make positive rewards more likely)

        self.tf_setup(mode)
        self.env = gym.make(env_name)
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
        self.memory = buffer(
            maxlen=self.memory_size, 
            state_img_width=self.observations_shape[0],
            state_img_height=self.observations_shape[1])

    def tf_setup(self, mode):
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

        # Set up tensorboard for training logging:
        if mode == 'train':
            self.tf_summary_writer = tf.summary.create_file_writer(self.log_dir+f'/{datetime.now()}/')

    def build_model(self):

        Inputs = tf.keras.Input(shape=(self.observations_shape[0], self.observations_shape[1], self.img_stack_count))
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
            input = np.expand_dims(input, axis = 3)
            return int(np.argmax(self.model.predict(input)))
        else:
            return np.random.randint(0, self.n_actions)

    def train_batch(self):

        # Dont train during pre-training phase (pre-fill memory)
        if self.frame_count < self.d_min:
            return 0 

        # Extract minibatch:
        states, actions, rewards, states_new, terminals = self.memory.sample(self.img_stack_count, self.batch_size)

        # Calculate discounted future reward
        discounted_max_future_reward = self.gamma * np.max(self.model_target.predict(states_new), axis = 1)
        discounted_max_future_reward[np.where(terminals == True)] = 0 # terminal states have no future reward

        targets = self.model.predict(states)
        for i in range(self.batch_size):
            targets[i, actions[i]] = rewards[i] + discounted_max_future_reward[i]

        history = self.model.fit(x=states, y=targets, verbose=0)

        # Log data to tensorboard:
        with self.tf_summary_writer.as_default():
            tf.summary.scalar('loss', history.history['loss'][0], step=self.frame_count)
            
        if self.frame_count % self.model_transfer_freq == 0:
            self.model_target.set_weights(self.model.get_weights())
            self.model.save(f'./snake.h5') # Back up progress thus far

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
        output = output.astype(np.uint8)    # Save as byte dtype to save memory

        return output
    
    def evaluate(self, n_episodes=5, saved_model=None, render = False, epsilon_eval=0.0, max_steps=100, save=None):
        '''
        inputs:
        n_episodes: number of episodes on which to evaluate.
        saved_model: Use a specific saved model file. If not supplied, use the current class model (self.model).
        render: boolean value to display the evaluation episodes or not
        epsilon_eval: probability of selecting a random action (set as 0.05 to avoid moving in 
            loops in partially-trained models)
        max_steps: End game after this many steps

        returns: score, frames, average_Q
        score: average game score (sum(score)/n_episodes)
        frames: Average number of frames for each episode
        average_Q: Average action value for all steps in the game
        save: string containing save path or None to disable
        '''

        if saved_model is None:
            play_model = self.model
        else:
            play_model = tf.keras.models.load_model(saved_model)

        average_Qs = []
        score = 0
        game_frames = []

        for _ in range(n_episodes):

            observation = self.env.reset()
            
            if save is not None:
                plt.imshow(observation)
                plt.savefig(save+'/img0.jpg', dpi=200)
                plt.close()

            state = self.preprocess(observation)

            step = 0
            while True:

                if np.random.uniform() > epsilon_eval:
                    input = np.expand_dims(state, axis = 0)
                    input = np.expand_dims(input, axis = 3)
                    Qs = play_model.predict(input)
                    average_Qs.append(np.mean(Qs))
                    action = int(np.argmax(Qs))
                else:
                    action = self.env.action_space.sample()
                observation, reward, done, _ = self.env.step(action)
                if save is not None:
                    plt.imshow(observation)
                    plt.savefig(save + f'/img{step+1}.jpg', dpi=200)
                    plt.close()


                score += reward
                state = self.preprocess(observation)

                if render:
                    self.env.render()

                step += 1

                if done or (step > max_steps):
                    game_frames.append(step)
                    step = 0
                    break
            
        return score/n_episodes, np.mean(game_frames), np.mean(average_Qs)

    def train(self, n_frames=1000000):

        '''
        Main training function. This is the part that enacts and observes the markov chain:
        State, action, reward, new state, etc

        Arguments:
        n_frames: number of frames to train on (default: 1,000,000)
        '''

        start_time = time.time()
        print(f"Starting training on {n_frames} game frames at {datetime.now()}")
        print(f"Use tensorboard to view training stats")

        episode = 0
        training_done = False
        while not training_done:

            # Get first state
            state = self.preprocess(self.env.reset())

            # Add start frames to memory:
            for _ in range(self.img_stack_count):
                # Set valid to false to prevent training on these frames:
                self.memory.remember(0, 0, state, False, train=False)

            # Advance the frame count
            self.frame_count += 1

            for _ in range(self.n_steps): # game steps

                action = self.choose_action(state)
                observation, reward, done, _ = self.env.step(action)
                state_new = self.preprocess(observation)

                self.memory.remember(action, reward, state_new, done)
                state = state_new

                self.train_batch()

                self.frame_count += 1

                if self.frame_count >= (n_frames+self.d_min):
                    training_done = True

                if done:

                    episode += 1
                    if episode % self.eval_freq == 0:

                        score, game_frames, average_Q = self.evaluate()

                        with self.tf_summary_writer.as_default():
                            tf.summary.scalar('Score', score, step=episode)
                            tf.summary.scalar('Game Frames', game_frames, step=episode)
                            tf.summary.scalar('Average Q', average_Q, step=episode)
                            tf.summary.scalar('epsilon', self.epsilon, step=self.frame_count)

                    break

        print(f"Training complete at {datetime.now()}")
        print(f"Total time: {round((time.time() - start_time)/3600, 2)} hours")

class buffer():
    '''
    Ths class implements a circular buffer system, but also allows that the state only be stored once and indexed. I.e. the state 
    doesn't need to be stored separately as the starting frame of one transition and the resultant frame of another.

    The current memory position is stored in the variable memory_counter, and when memory_counter exceeds maxlen, it is reset to zero.
    '''
    def __init__(self, maxlen, state_img_width, state_img_height):
        self.state_memory = np.zeros(shape=(state_img_width, state_img_height, maxlen), dtype=np.uint8)
        self.action_memory = np.zeros(shape=maxlen, dtype = np.uint8)
        self.reward_memory = np.zeros(shape=maxlen, dtype = np.int8) # unsigned - could be negative
        self.done_memory = np.zeros(shape=maxlen, dtype=bool)   
        self.train_memory = np.zeros(shape=maxlen, dtype=bool)
        self.memory_counter = 0
        self.maxlen = maxlen

    def remember(self, action, reward, state, done, train=True):
        '''
        Arguments:
        action, reward, state, done: training inputs for a transition (use only final state)
        train: flag to indicate whether the transition is valid for training. The first transition (or transitions if
            frame stacking is performed) will include transitions from the previous episode, and as such should not be 
            used for training
        '''
        self.action_memory[self.memory_counter] = action
        self.reward_memory[self.memory_counter] = reward
        self.state_memory[:, :, self.memory_counter] = state
        self.done_memory[self.memory_counter] = done
        self.train_memory[self.memory_counter] = train

        # Advance memory counter and return to the beginning if buffer overflows
        self.memory_counter = (self.memory_counter + 1) % self.maxlen

    def sample(self, stack_size, batch_size):
        
        # Extract
        valid_indices = np.where(self.train_memory[stack_size:] == True)[0]
        indices = np.random.choice(valid_indices, size=batch_size, replace=False)

        states = np.stack([self.state_memory[:, :, x - 1] for x in indices], axis = 0) # state_memory stores resultant state, so subtract 1 from index to get starting frame
        states_new = np.stack([self.state_memory[:, :, x] for x in indices], axis = 0)
        actions = self.action_memory[indices]
        rewards = self.reward_memory[indices]
        terminals = self.done_memory[indices]
        
        states = np.expand_dims(states, axis=3)
        states_new = np.expand_dims(states_new, axis=3)

        return states, actions, rewards, states_new, terminals

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the DQN model with the Snake environment')
    parser.add_argument('--n_frames', '-n', type=int, help='Number of frames to train on (default: 1,000,000)', default=1000000)
    parser.add_argument('--n_episodes', '-g', type=int, help='Number of episodes to evaluate on (default: 10)', default=10)
    parser.add_argument('--max_steps', '-s', type=int, help='Maximum number of frames in evaluation episodes (default: 200)', default=200)
    parser.add_argument('--epsilon', '-e', type=float, help='Epsilon value for evaluation (default: 0.0)', default=0.0)
    parser.add_argument('--mode', '-m', type=str, help='Mode: "test" or "train" (default: "train")', default='train')
    parser.add_argument('--model_file', '-f', type=str, help='Path to tensorflow model file (default: use untrained model)', default=None)
    parser.add_argument('--render', '-r', type=bool, help='Boolean flag to disable rendering during evaluation (default: True)', default=True)
    args = parser.parse_args()

    # Instantiate agent:
    agent = DQN(mode=args.mode)

    if args.mode == 'train':
        agent.train(n_frames = args.n_frames)
    else:
        score, game_frames, average_Q = agent.evaluate(
            saved_model = args.model_file, # comment to play with an untrained model
            n_episodes = args.n_episodes,  # Play 10 episodes
            epsilon_eval = args.epsilon,   # No random actions
            max_steps = args.max_steps,    # Terminate episode after N steps (avoid snake getting stuck in loops)
            render  =  args.render)        # Render display to monitor