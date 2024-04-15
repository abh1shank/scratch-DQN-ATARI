import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from collections import deque
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error

class DeepQLearning:

    def __init__(self, env, gamma, epsilon, num_episodes):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.state_dim = 128
        self.action_dim = 9
        self.buffer_size = 300
        self.batch_size = 100
        self.update_period = 100
        self.update_counter = 0
        self.sum_rewards = []
        self.buffer = deque(maxlen=self.buffer_size)
        self.main_net = self.create_network()
        self.target_net = self.create_network()
        self.target_net.set_weights(self.main_net.get_weights())
        self.actions = []

    def create_network(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dim, activation='selu'))
        model.add(Dense(56, activation='selu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(optimizer=RMSprop(), loss=mean_squared_error, metrics=['accuracy'])
        return model

    def train_network(self):
        if len(self.buffer) > self.batch_size:
            random_sample = random.sample(self.buffer, self.batch_size)
            current_state_batch = np.zeros(shape=(self.batch_size, 128))
            next_state_batch = np.zeros(shape=(self.batch_size, 128))
            for index, (current_state, action, _, next_state, _) in enumerate(random_sample):
                current_state_batch[index, :] = current_state
                next_state_batch[index, :] = next_state
            q_next_state_target = self.target_net.predict(next_state_batch)
            q_current_state_main = self.main_net.predict(current_state_batch)
            input_network = current_state_batch
            output_network = np.zeros(shape=(self.batch_size, 9))
            self.actions = []
            for index, (current_state, action, reward, _, terminated) in enumerate(random_sample):
                if terminated:
                    y = reward
                else:
                    y = reward + self.gamma * np.max(q_next_state_target[index])
                self.actions.append(action)
                output_network[index] = q_current_state_main[index]
                output_network[index, action] = y
            self.main_net.fit(input_network, output_network, batch_size=self.batch_size, verbose=0, epochs=40)
            self.update_counter += 1
            if self.update_counter > (self.update_period - 1):
                self.target_net.set_weights(self.main_net.get_weights())
                print("Target network updated!")
                print("Counter value {}".format(self.update_counter))
                self.update_counter = 0

    def training_episodes(self):
        for index_episode in range(self.num_episodes):
            rewards_episode = []
            print("Simulating episode {}".format(index_episode))
            current_state, _ = self.env.reset()
            terminal_state = False
            while not terminal_state:
                action = self.select_action(current_state, index_episode)
                next_state, reward, terminal_state, _, _ = self.env.step(action)
                rewards_episode.append(reward)
                self.buffer.append((current_state, action, reward, next_state, terminal_state))
                self.train_network()
                current_state = next_state
            print("Sum of rewards {}".format(np.sum(rewards_episode)))
            self.sum_rewards.append(np.sum(rewards_episode))

    def select_action(self, state, index):
        if index < 1:
            return np.random.choice(self.action_dim)
        random_number = np.random.random()
        if index > 200:
            self.epsilon = 0.999 * self.epsilon
        if random_number < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            q_values = self.main_net.predict(state.reshape(1, 128))
            return np.random.choice(np.where(q_values[0, :] == np.max(q_values[0, :]))[0])
