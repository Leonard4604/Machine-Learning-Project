# Import the packages
import gymnasium as gym
import numpy as np
import random
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

# Define the environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Define the Hyperparameters
train_episodes = 500
test_episodes = 100
max_steps = 600
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
action_space = env.action_space
batch_size = 20


# Define our class agent
class Agent:
    def __init__(self, state_size, action_size, action_space):
        self.experience = deque(maxlen=100_000)
        self.learning_rate = 0.001
        self.epsilon = 1
        self.max_eps = 1
        self.min_eps = 0.01
        self.eps_decay = 0.01/3
        self.gamma = 0.9
        self.state_size = state_size
        self.action_size = action_size
        self.action_space = action_space
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def add_experience(self, new_state, reward, terminated, state, action):
        self.experience.append((new_state, reward, terminated, state, action))

    def action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def predict(self, state):
        return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.experience, batch_size)
        for new_state, reward, terminated, state, action in minibatch:
            target = reward
            if not terminated:
                target += self.gamma * np.max(self.model.predict(new_state)[0])
            target_function = self.model.predict(state)
            target_function[0][action] = target
            self.model.fit(state, target_function, epochs=1, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon = (self.max_eps - self.min_eps) * np.exp(-self.eps_decay * episode) + self.min_eps

    def load_model(self, name):
        self.model = load_model(f'model/{name}')

    def save_model(self, name):
        self.model.save(f'model/{name}')

    def load_weights(self, name):
        self.model.load_weights(f'weights/{name}')

    def save_weights(self, name):
        self.model.save_weights(f'weights/{name}')


# Create an instance of our agent's class
agent = Agent(state_size, action_size, action_space)

# Train our model
for episode in range(train_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    for step in range(max_steps):
        action = agent.action(state)
        new_state, reward, terminated, _, _ = env.step(action)
        new_state = np.reshape(new_state, [1, state_size])
        agent.add_experience(new_state, reward, terminated, state, action)
        state = new_state

        if terminated:
            break

    if len(agent.experience) > batch_size:
        agent.replay(batch_size)

# Render the environment in human mode to see the effective path
env = gym.make("CartPole-v1", render_mode="human")

# Evaluate the model
for episode in range(test_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])

    for step in range(max_steps):
        action = agent.predict(state)
        new_state, reward, terminated, _, _ = env.step(action)
        new_state = np.reshape(new_state, [1, state_size])
        state = new_state

        if terminated:
            break

env.close()