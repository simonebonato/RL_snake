import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import pygame

import random

class cube(object):
    rows = 20
    w = 500

    def __init__(self, start, dirnx=1, dirny=0, color =(255,0,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny

        self.pos = (self.pos[0]+ self.dirnx,self.pos[1]+ self.dirny )
    def draw(self, surface, eyes = False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2,dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)

class snake(object):
    body = []
    turns = {}

    def __init__(self, color, pos):
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.SIZE = 20
        self.useless_steps = 0

    def death(self):

        self.reset()
        dead_counts += 1
        just_a_step = False

        if yes_decay:
            if epsilon > 0:
                epsilon -= epsilon_decay_value
            else:
                epsilon = 0
                yes_decay = False
        reward = DEATH_REWARD
        return reward


    # FOR CNN #
    def get_image(self, snack):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)

        for cube in self.body:  # starts an rbg of our size
            env[cube.pos[0]][cube.pos[1]] = (255, 175, 0)  # sets the food location tile to green color

        env[snack[0]][snack[1]] = (0, 255, 0)
        # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

    def move(self, action):
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()

        #move left
        if action == 0:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        #move right
        elif action == 1:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        #move up
        elif action == 2:
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        #move down
        elif action ==3:
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i,c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)

            else:
                if (c.dirnx == -1 or c.dirny == 1 or c.dirny == -1) and c.pos[0] < 0: c.pos = (c.rows-1, c.pos[1])
                elif (c.dirnx == 1 or c.dirny == 1 or c.dirny == -1) and c.pos[0] > c.rows-1: c.pos = (0,c.pos[1])
                elif (c.dirny == 1 or c.dirnx == 1 or c.dirnx ==-1) and c.pos[1] > c.rows-1: c.pos = (c.pos[0], 0)
                elif (c.dirny == -1 or c.dirnx == 1 or c.dirnx == -1) and c.pos[1] < 0: c.pos = (c.pos[0],c.rows-1)
                else: c.move(c.dirnx,c.dirny)


            # else:
            #     if c.dirnx == -1 and c.pos[0] <= 0: c.pos = (c.rows-1, c.pos[1])
            #     elif c.dirnx == 1 and c.pos[0] >= c.rows-1: c.pos = (0,c.pos[1])
            #     elif c.dirny == 1 and c.pos[1] >= c.rows-1: c.pos = (c.pos[0], 0)
            #     elif c.dirny == -1 and c.pos[1] <= 0: c.pos = (c.pos[0],c.rows-1)
            #     else: c.move(c.dirnx,c.dirny)

    def new_state(self):

        just_a_step = True

        if self.head.pos == snack.pos:
            reward = FOOD_REWARD
            rewards_sum += reward
            self.addCube()
            snack_coordinates = randomSnack(rows,self)
            snack = cube(snack_coordinates, color =(0,255,0))
            just_a_step = False
            self.useless_steps = 0
            # print('Took the candy!')


        for x in range(len(self.body)):
            if self.body[x].pos in list(map(lambda z:z.pos,self.body[x+1:])):
                self.death()
                break

        if (self.head.pos[0] < 0 or self.head.pos[1] < 0 or self.head.pos[0] > rows-1 or self.head.pos[1] > rows-1):
            self.death()


        # previous_relative = relative_position
        # relative_position = (self.head.pos[0] - snack_coordinates[0] , self.head.pos[1] - snack_coordinates[1])

        if just_a_step:
            if (np.abs(relative_position[0]) < np.abs(previous_relative[0])) or (np.abs(relative_position[1]) < np.abs(previous_relative[1])):
                reward = STEP_CLOSER_REWARD

            else:
                reward = STEP_REWARD
                self.useless_steps += 1

        if self.useless_steps == 400:
            self.death()

        new_observation = np.array(self.get_image())

        return new_observation, reward



    def reset(self):
        self.body = []
        self.head = cube((np.random.randint(0,20),np.random.randint(0,20)))
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1
        self.useless_steps = 0

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(cube((tail.pos[0]-1,tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.pos[0]+1,tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.pos[0],tail.pos[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.pos[0],tail.pos[1]+1)))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy




    def draw(self,surface):
        for i,c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)
def drawGrid(w, rows, surface):
    sizeBtwn = w // rows

    x = 0
    y = 0

    for l in range(rows):
         x = x + sizeBtwn
         y = y + sizeBtwn

         pygame.draw.line(surface, (255,255,255), (x,0), (x,w))
         pygame.draw.line(surface, (255,255,255), (0,y), (w,y))
def redrawWindow(surface):
    global rows, width,self, snack

    surface.fill((0,0,0))
    self.draw(surface)
    snack.draw(surface)
    drawGrid(width, rows,surface)
    pygame.display.update()
def randomSnack(rows, item):

    positions = item.body

    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z: z.pos == (x,y), positions))) > 0:
            continue
        else:
            break

    return (x,y)
def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass
def obstacles(snake):
    head_coordinates = snake.head.pos
    obs_up = 0
    obs_down = 0
    obs_left = 0
    obs_right = 0

    if snake.head.pos[0] == 0: obs_left = 1
    if snake.head.pos[0] == 19: obs_right = 1
    if snake.head.pos[1] == 0: obs_up = 1
    if snake.head.pos[1] == 19: obs_down = 1


    for cube in snake.body[1:]:
        if (head_coordinates[0] == cube.pos[0] and cube.pos[1] == head_coordinates[1] + 1) or (head_coordinates[0] == cube.pos[0] and cube.pos[1] == head_coordinates[1] - 19):
            obs_down = 1
        if (head_coordinates[0] == cube.pos[0] and cube.pos[1] == head_coordinates[1] - 1) or (head_coordinates[0] == cube.pos[0] and cube.pos[1] == head_coordinates[1] + 19):
            obs_up = 1
        if (head_coordinates[0] == cube.pos[0] + 1 and cube.pos[1] == head_coordinates[1]) or (head_coordinates[0] == cube.pos[0] - 19 and cube.pos[1] == head_coordinates[1]):
            obs_left = 1
        if (head_coordinates[0] == cube.pos[0] - 1 and cube.pos[1] == head_coordinates[1]) or (head_coordinates[0] == cube.pos[0] + 19 and cube.pos[1] == head_coordinates[1]):
            obs_right = 1


    return (obs_up,obs_left,obs_down,obs_right)




SIZE = 20
OBSERVATION_SPACE_VALUES =(SIZE,SIZE,3) # RGB images of the size of the grid
ACTION_SPACE_SIZE = 4 # possible actions



DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20


# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

#MOST IMPORTANT PARAMETERS
EPISODES = 300000
watch = True
epsilon = 0.9
yes_decay = True

#EPSILON
#the higher the more randomly we choose the action to perform, and not he one with higher Q-value
EPSILON_ZERO_AT = 0.95 #percent of the episodes
epsilon_decay_value = epsilon / (EPISODES * EPSILON_ZERO_AT)



#REWARDS
FOOD_REWARD = +25
STEP_REWARD = -5
STEP_CLOSER_REWARD = 10
DEATH_REWARD = -30

class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # Used to count when to update target network with main network'self weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step'self data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]




global width, rows, snack, self
width = 500
rows = 20
if watch:
    win = pygame.display.set_mode((width, width))
    clock = pygame.time.Clock()
    epsilon = 0

starting_position = (np.random.randint(0,20),np.random.randint(0,20))
self = snake((255,0,0), starting_position)
snack_coordinates = randomSnack(rows,self)
snack = cube(snack_coordinates, color =(0,255,0))


rewards_list = []
scores_list = []
rewards_sum = 0
dead_counts = 0
previous_relative = (99,99)
useless_steps = 0
max_len = 1
while dead_counts <= EPISODES:

    if watch:
        pygame.time.delay(50) #the lower the faster, original = 50
        clock.tick(10) #the lower the slower, original = 10

    relative_position = (self.head.pos[0] - snack_coordinates[0] , self.head.pos[1] - snack_coordinates[1])
    current_state = (relative_position, obstacles(self))

    if np.random.random() > epsilon:
        action = np.argmax(q_table[current_state])
    else:
        action = np.random.randint(0, 4)



    self.move(action)
    self.new_state()




    if watch:
        redrawWindow(win)
