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




    def reset(self):
        self.body = []
        self.head = cube((np.random.randint(0,20),np.random.randint(0,20)))
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

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
    global rows, width,s, snack

    surface.fill((0,0,0))
    s.draw(surface)
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
def death():
    global dead_counts, just_a_step, previous_relative, useless_steps, epsilon, max_len, rewards_list, scores_list,rewards_sum, yes_decay
    print('The score is:', len(s.body), f'Match number: {dead_counts + 1}\n')
    if len(s.body) > max_len: max_len = len(s.body)
    scores_list.append(len(s.body))
    s.reset()
    useless_steps = 0
    previous_relative = (99,99)
    dead_counts += 1
    just_a_step = False
    rewards_list.append(rewards_sum)
    rewards_sum = 0

    if yes_decay:
        if epsilon > 0:
            epsilon -= epsilon_decay_value
        else:
            epsilon = 0
            yes_decay = False





SIZE = 20
start_q_table = 'best_so_far.pickle' #'prima_prova.pickle' # 'None' if start from the beginning
q_table_name = '300000episodes'#'epsilon_05_learning_05_discount_097'

#MOST IMPORTANT PARAMETERS
EPISODES = 300000
watch = True
epsilon = 0.9
yes_decay = True

#LEARNING PARAMETERS
LEARNING_RATE = 0.5 #between 0 and 1, standard is 0.1
DISCOUNT = 0.97 #original: 0.97, the highter the more the agent will strive for a long-term reeward

#EPSILON
#the higher the more randomly we choose the action to perform, and not he one with higher Q-value
EPSILON_ZERO_AT = 0.95 #percent of the episodes
epsilon_decay_value = epsilon / (EPISODES * EPSILON_ZERO_AT)



#REWARDS
FOOD_REWARD = +25
STEP_REWARD = -5
STEP_CLOSER_REWARD = 10
DEATH_REWARD = -30


print(f'The food reward is: {FOOD_REWARD}\n The step reward is: {STEP_REWARD}\n The step closer reward is: {STEP_CLOSER_REWARD}\n The death reward is: {DEATH_REWARD}\n')

if start_q_table is None:
    q_table = {}
    for x_food_rel in range(-SIZE+1,SIZE):
        for y_food_rel in range(-SIZE+1,SIZE):
            for obs_up in range(2):
                for obs_left in range(2):
                    for obs_down in range(2):
                        for obs_right in range(2):
                            q_table[(x_food_rel,y_food_rel),(obs_up,obs_left,obs_down,obs_right)] = [np.random.uniform(-3,0) for i in range (4)]

else:
    with open(start_q_table, 'rb') as f:
        q_table = pickle.load(f)



global width, rows, snack, s
width = 500
rows = 20
if watch:
    win = pygame.display.set_mode((width, width))
    clock = pygame.time.Clock()
    epsilon = 0

starting_position = (np.random.randint(0,20),np.random.randint(0,20))
s = snake((255,0,0), starting_position)
snack_coordinates = randomSnack(rows,s)
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

    relative_position = (s.head.pos[0] - snack_coordinates[0] , s.head.pos[1] - snack_coordinates[1])
    current_state = (relative_position, obstacles(s))

    if np.random.random() > epsilon:
        action = np.argmax(q_table[current_state])
    else:
        action = np.random.randint(0, 4)



    s.move(action)
    just_a_step = True

    if s.head.pos == snack.pos:
        reward = FOOD_REWARD
        rewards_sum += reward
        s.addCube()
        snack_coordinates = randomSnack(rows,s)
        snack = cube(snack_coordinates, color =(0,255,0))
        just_a_step = False
        useless_steps = 0
        # print('Took the candy!')


    for x in range(len(s.body)):
        if s.body[x].pos in list(map(lambda z:z.pos,s.body[x+1:])):
            reward = DEATH_REWARD
            rewards_sum += reward
            death()
            break

    if (s.head.pos[0] < 0 or s.head.pos[1] < 0 or s.head.pos[0] > rows-1 or s.head.pos[1] > rows-1):
        reward = DEATH_REWARD
        rewards_sum += reward
        death()


    previous_relative = relative_position
    relative_position = (s.head.pos[0] - snack_coordinates[0] , s.head.pos[1] - snack_coordinates[1])

    if just_a_step:
        if (np.abs(relative_position[0]) < np.abs(previous_relative[0])) or (np.abs(relative_position[1]) < np.abs(previous_relative[1])):
            reward = STEP_CLOSER_REWARD
            rewards_sum += reward
        else:
            reward = STEP_REWARD
            rewards_sum += reward
            useless_steps += 1


    new_state = (relative_position, obstacles(s))
    max_future_q = np.max(q_table[new_state])
    current_q = q_table[current_state][action]
    new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
    q_table[current_state][action] = new_q

    if useless_steps == 400:
        death()

    if watch:
        redrawWindow(win)

with open(f'{q_table_name}.pickle', 'wb') as f:
    pickle.dump(q_table,f)

print(f'The model {q_table_name} has been created!\nThe max score reached is: {max_len}')


fig, axs = plt.subplots(2)
fig.suptitle(f'Rewards and Scores: {q_table_name}')
axs[0].plot(range(EPISODES+1), rewards_list)
axs[1].plot(range(EPISODES+1), scores_list)
plt.show()
