import numpy as np
from Queue import Queue
from Direction import Direction
import random 
import os
import tensorflow as tf


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
        self.buffer = Queue(1000)

        self.build_model()

        if os.path.exists(self.file):
            self.model.load_weights(self.file)
        

    def build_model(self):
        input1 = tf.keras.layers.Input((64, 64, 1))

        x = tf.keras.layers.Conv2D(5, (3, 3), activation='relu')(input1)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(5, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x1 = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(4, activation='linear')(x)

        self.model = tf.keras.Model(inputs=[input1], outputs=[x])

        self.model.compile('adam', 'categorical_crossentropy')
        self.model.summary()


    def make_move(self, state):
        threshold = random.random()

        if threshold > self.epsilon or not self.training:
            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
            print(q_values)

            max_ = max(q_values)
            action = random.choice([self.ACTIONS[i] for i, x in enumerate(q_values) if x == max_])
        else:
            action = random.choice(self.ACTIONS)

        return action
    
    def update_table(self, state, action, reward, next_state, done):
        pred_state = self.model.predict(state, verbose=0)[0]
        pred_next_state = self.model.predict(next_state, verbose=0)[0]

        if done:
            true_val = (1 - self.alpha)*pred_state[action.value] + self.alpha*(reward)
        else:
            true_val = (1 - self.alpha)*pred_state[action.value] \
                + self.alpha*(reward+self.gamma*np.max(pred_next_state))
    
        true_y = np.copy(pred_state)
        true_y[action.value] = true_val
        self.model.fit(state, np.expand_dims(true_y, axis=0), verbose=0)

        self.buffer.add((state, action, reward, next_state, done))

    # def train_on_batch(self, batch_size):
    #     batch = random.choices(self.buffer.array, k=batch_size)
    #     X = []
    #     y = []
    #     for state, action, reward, next_state, done in batch:
    #         pred_state = self.model.predict(state, verbose=0)[0]
    #         pred_next_state = self.model.predict(next_state, verbose=0)[0]

    #         if done:
    #             true_val = (1 - self.alpha)*pred_state[action.value] + self.alpha*(reward)
    #         else:
    #             true_val = (1 - self.alpha)*pred_state[action.value] \
    #                 + self.alpha*(reward+self.gamma*np.max(pred_next_state))
        
    #         true_y = np.copy(pred_state)
    #         true_y[action.value] = true_val
    #         X.append(state)
    #         y.append(true_y)
    #     X = np.array(X)
    #     y = np.array(y)
    #     self.model.fit(X, y)

    def train_on_batch(self, batch_size):
        batch = random.choices(self.buffer.array, k=batch_size)

        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []

        for state, action, reward, next_state, done in batch:
            states.append(np.expand_dims(state, axis=0))
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        pred_states = self.model.predict(np.array(states))
        pred_next_states = self.model.predict(np.array(next_states))

        X = []
        y = []

        for pred_state, action, reward, pred_next_state, done in zip(pred_states, actions, rewards, pred_next_states, dones):
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
        if episode % 4 == 0:
            self.model.save_weights(self.file)
            self.train_on_batch(2)

        self.epsilon = self.min_exploration_rate + \
              (self.max_exploration_rate - self.min_exploration_rate)*np.exp(-episode*self.exploration_decay_rate)

