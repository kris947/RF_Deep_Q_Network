import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = 1.0
        # Action space
        self.action_space = action_space
        # Memory
        self.memory = deque(maxlen=1000000)
        # Create a model
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001))
        self.state=0

    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience(self):
        if len(self.memory) < 20:
            return
        history = random.sample(self.memory, 20)
        for state, action, reward, state_next, terminal in history:
            new_q = reward
            if not terminal:
                new_q = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = new_q
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= 0.995
        self.exploration_rate = max(0.01, self.exploration_rate)


def main():
    # Kornyezet import
    env = gym.make(ENV_NAME)
    #  Pont szamolas
    score_logger = ScoreLogger(ENV_NAME)
    #  Observation space + action space definialas
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    #  RL modell
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0

    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            env.render()
            
            action = dqn_solver.act(state)

            state_next, reward, terminal, info = env.step(action)

            reward = reward if not terminal else -reward

            state_next = np.reshape(state_next, [1, observation_space])


            dqn_solver.add_to_memory(state, action, reward, state_next, terminal)


            state = state_next
            if terminal:
                result="Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step)
                print (result)
                score_logger.add_score(step, run)
                break
            dqn_solver.experience()


   



if __name__ == "__main__":
    main()
