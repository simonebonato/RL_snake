from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from collection import deque
import time

import random

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = '256x2'
MINIBATCH_SIZE = 64
DISCOUNT = 0.99

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):


    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):
        '''
        we create 2 models, the first one will use "fit" for every step while
        the other one always makes "predict", else using just one will go crazy
        '''
        # main model
        self.model = self.create_model()

        # target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # a list of a max size of maxlen
        # useful because when we fit the model, if we use just one value at a
        #  time it gest overfitted at every single fit
        # with this we can make our batch of REPLAY_MEMORY_SIZE
        # RANDOMLY SELECTED ELEMENTS

        self.replay_memory= deque(maxlen=REPLAY_MEMORY_SIZE)
        self.TensorBoard = ModifiedTensorBoard(log_dir=f'logs/{MODEL_NAME}-{int(time.time())}')
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3,3), input_shape= env.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2,2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation = 'linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics =['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model_predict(np.array.reshape(-1, *state.shape)/255)[0]


    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transitions[0] for transitions in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transitions[3] for transitions in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_states, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]

            kk = 'asdasdsdfg dsfg'
