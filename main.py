# important necessary packages
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
import math
import matplotlib.pyplot as plt


class SpaceshipMove(Env):
    # initialize the values
    def __init__(self):
        # either turn left, keep going straight, or turn right
        self.action_space = Discrete(3)
        # Box of x coordinate and y coordinate with the heading (angle of travel)
        self.observation_space = Box(low=np.array([0,0,0]), high=np.array([100,100,90]))
        # starts at origin and is pointed 40-50 degrees above the x-axis
        self.state = np.array([0,0,random.randint(40, 50)])
        # coordinate values
        self.x = 0
        self.y = 0
        # for plotting purposes (see the trajectory of spacecraft
        self.plot = [tuple([self.x, self.y])]
        # experimental value that keeps track of heading changes
        self.energy_expended = 0

        # possibly add a length? or an energy expended limit?

    def step(self, action):
        # get the new direction based on the action
        newHeading = self.state[2] + action - 1

        # every time direction changes, expel one unit of energy
        if not action == 1:
            self.energy_expended += 1

        # change the coordinate values
        self.x += math.cos(math.radians(newHeading))
        self.y += math.sin(math.radians(newHeading))

        #
        self.state = np.array([self.x, self.y, newHeading])
        self.plot.append(tuple([self.x, self.y]))

        reward = -1
        if self.get_distance(100, 100) <= 3:
            reward = 0

        done = False
        # if spaceship goes out of bounds or reaches the target within acceptable margin of error
        # episode is finished
        if self.x > 100 or self.y > 100 or self.state[2] < 0 or self.state[2] > 90 or reward == 0:
            done = True

        info = {}
        return self.state, reward, done, info

    def render(self):
        # render: show the graph of the trajectory of the spaceship
        circle1 = plt.Circle((100, 100), 3, color='r')
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.add_patch(circle1)
        ax.set_xlim(left=0, right=100)
        ax.set_ylim(bottom=0, top=100)
        ax.plot([a[0] for a in self.plot], [a[1] for a in self.plot], 'b')
        plt.show()

    # set back to original parameters
    def reset(self):
        self.state = np.array([0,0,random.randint(40, 50)])
        self.x = 0
        self.y = 0
        self.plot = [tuple([self.x, self.y])]
        self.energy_expended = 0
        return self.state

    # helper functions
    def get_distance(self, target_x, target_y):
        return math.sqrt((self.x - target_x) ** 2 + (self.y - target_y) ** 2)

    def get_energy(self):
        return self.energy_expended


# make environment
env = SpaceshipMove()
# run 100 episodes to see how it runs
'''
episodes = 100
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        print(n_state, reward)
    if reward == 0:
        print("SUCCESS")
        break
env.close()
'''

# parameters for q-learning
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 1000000
SHOW_EVERY = 2000

# make the discrete values table: 20 values for x and y coordinates and 9 values for the heading
DISCRETE_OS_SIZE = [20, 20, 9]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# epsilon decay function values
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# form the q-table using numpy
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# dictionary to see values more effectively
# keeps track of episode #, average reward, minimum reward, and maximum reward
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# function to assign discrete values for the states
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    returnTuple = list(discrete_state.astype(np.int))
    if returnTuple[0] == 20:
        returnTuple[0] = 19
    if returnTuple[1] == 20:
        returnTuple[1] = 19
    if returnTuple[2] == 9:
        returnTuple[2] = 8
    return tuple(returnTuple)

# keep track of the number of successful attempts
total_right = 0
total_count_last = 0
total_count_last_last = 0

# plotting the average rate of the spaceship making it to the target over the episodes
plot_correct = []

for episode in range(EPISODES):
    render = False
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, info = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state][action]

            # q-value function
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q

        elif math.sqrt((new_state[0] - 100) ** 2 + (new_state[1] - 100) ** 2) <= 3:
            q_table[discrete_state][action] = 0
            '''
            print("-------------------------DONE-------------------------")
            print("Episode:  ", episode)
            print("State:    ", env.state)
            print("Distance: ", math.sqrt((env.x - 100) ** 2 + (env.y - 100) ** 2))
            print("Energy:   ", env.get_energy())
            '''
            total_right += 1
            if episode >= 900000:
                total_count_last += 1
                if episode >= 999000:
                    total_count_last_last += 1

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    plot_correct.append(tuple([episode, total_right / (episode + 1)]))
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY: ]) / len(ep_rewards[-SHOW_EVERY: ])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY: ]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY: ]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY: ])} max: {max(ep_rewards[-SHOW_EVERY: ])}")

    '''
    render = bool(episode % 50000 == 0)

    if render and reward == 0:
        print(env.get_energy())
        env.render()
    '''

env.close()

print(total_right / 1000000)
print(total_count_last / 100000)
print(total_count_last_last / 1000)

fig, ax = plt.subplots()
ax.set_xlim(left=0, right=EPISODES)
ax.set_ylim(bottom=0, top=1)
ax.plot([a[0] for a in plot_correct], [a[1] for a in plot_correct], 'b')
plt.show()


plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
