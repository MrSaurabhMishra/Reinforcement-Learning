import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import random

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        
        self.fc1 = layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(action_size, activation=None) 

    def call(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

class NexusAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optimizers.Adam(learning_rate=0.003)
        self.criterion = losses.MeanSquaredError()

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = np.array(state).reshape(1, -1)
        q_values = self.model(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)
        
        target = float(reward)
        if not done:
            
            next_q_values = self.model(next_state)
            target = float(reward + 0.99 * np.max(next_q_values))

        with tf.GradientTape() as tape:
            all_q_values = self.model(state)
            current_q = all_q_values[0][action]
            loss = self.criterion([target], [current_q])
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))