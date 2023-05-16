import numpy as np
from Queue import Queue
from Direction import Direction
import random 
import os
import tensorflow as tf
import json


class SnakeBot:
    ACTIONS = [
        Direction.UP,
        Direction.RIGHT,
        Direction.DOWN,
        Direction.LEFT
    ]

    def __init__(self, file, training=False) -> None:
        self.alpha = 0.1
        self.gamma = 0.99
        self.min_exploration_rate = 0.1
        self.max_exploration_rate = 1
        self.exploration_decay_rate = 0.001
        self.epsilon = self.min_exploration_rate
        self.training = training
        self.file = file
        self.q_table = {}
        self.buffer = Queue(1000)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input((12)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])

        self.model.compile('adam', 'mse')
        self.model.summary()

        if os.path.exists(self.file):
            self.model.load_weights(self.file)

    def make_move(self, frame):
        threshold = random.random()

        if threshold > self.epsilon or not self.training:
            q_values = self.model.predict(np.expand_dims(frame, axis=0), verbose=0)[0]
            max_ = max(q_values)
            action = random.choice([self.ACTIONS[i] for i, x in enumerate(q_values) if x == max_])
        else:
            action = random.choice(self.ACTIONS)

        return action
    
    def update_table(self, state, action, reward, next_state, done):
        pred_state = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        pred_next_state = self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]

        if done:
            true_val = (1 - self.alpha)*pred_state[action.value] + self.alpha*(reward)
        else:
            true_val = (1 - self.alpha)*pred_state[action.value] \
                + self.alpha*(reward+self.gamma*np.max(pred_next_state))
    
        true_y = np.copy(pred_state)
        true_y[action.value] = true_val
        
        self.q_table[str(state)] = true_y.tolist()
        
        self.model.fit(np.expand_dims(state, axis=0), np.expand_dims(true_y, axis=0), verbose=0)

        self.buffer.add((state, action, reward, next_state, done))

    def train_on_batch(self, batch_size):
        batch = random.choices(self.buffer.array, k=batch_size)
        X = []
        y = []
        for state, action, reward, next_state, done in batch:
            pred_state = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            pred_next_state = self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0]

            if done:
                true_val = (1 - self.alpha)*pred_state[action.value] + self.alpha*(reward)
            else:
                true_val = (1 - self.alpha)*pred_state[action.value] \
                    + self.alpha*(reward+self.gamma*np.max(pred_next_state))
        
            true_y = np.copy(pred_state)
            true_y[action.value] = true_val
            X.append(state)
            y.append(true_y)
        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y)

    def update_epsilon(self, episode):
        if episode % 100 == 0:
            self.model.save_weights(self.file)

            with open("q_table_DNN_without_segments.json", 'w') as f:
                json.dump(self.q_table, f)

            self.train_on_batch(64)

        self.epsilon = self.min_exploration_rate + \
              (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-episode*self.exploration_decay_rate)

