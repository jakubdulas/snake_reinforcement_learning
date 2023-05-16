import numpy as np
import json
from Direction import Direction
import random 
import os


class SnakeBot:
    ACTIONS = [
        Direction.UP,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT
    ]

    def __init__(self, file, training=False) -> None:
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.99
        self.min_exploration_rate = 0.1
        self.max_exploration_rate = 1
        self.exploration_decay_rate = 0.001
        self.epsilon = self.min_exploration_rate
        self.training = training
        self.file = file

        if os.path.exists(self.file):
            with open(self.file, 'r') as f:
                self.q_table = json.load(f)

    def make_move(self, state):
        state = str(state)
        if state not in self.q_table:
            self.q_table[state] = [0]*len(self.ACTIONS)

        threshold = random.random()

        # if threshold > self.epsilon:
        #     max_ = max(self.q_table[state])
        #     action = random.choice([self.ACTIONS[i] for i, x in enumerate(self.q_table[state]) if x == max_])
        # else:
        #     action = random.choice(self.ACTIONS)

        if threshold > self.epsilon or not self.training:
            q_values = self.q_table[state]

            max_ = max(q_values)
            action = random.choice([self.ACTIONS[i] for i, x in enumerate(q_values) if x == max_])
        else:
            action = random.choice(self.ACTIONS)

        return action
    
    def update_table(self, state, action, reward, next_state, done):
        state = str(state)
        next_state = str(next_state)
        if done:
            self.q_table[state][action.value] = (1 - self.alpha)*self.q_table[state][action.value] \
            + self.alpha*(reward)
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = [0]*len(self.ACTIONS)

            self.q_table[state][action.value] = (1 - self.alpha)*self.q_table[state][action.value] \
                + self.alpha*(reward+self.gamma*np.max(self.q_table[next_state]))
        

    def update_epsilon(self, episode):
        if episode % 100 == 0:
            json_q_table = json.dumps(self.q_table)
            with open(self.file, 'w') as f:
                f.write(json_q_table)

            with open(self.file, 'r') as f:
                self.q_table = json.load(f)

        self.epsilon = self.min_exploration_rate + \
              (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-episode*self.exploration_decay_rate)

