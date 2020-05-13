import numpy as np
import random
import gym
import sys
from collections import defaultdict

def Greedy(env, q_table, epsilon, state):
    # epsilon-greedy 這裡是policy
    if random.random() < epsilon:
        action = env.action_space.sample() # Explore
    else:
        action = np.argmax(q_table[state]) # Exploit
    return action


env = gym.make("FrozenLake8x8-v0") 
env.reset()
# env.render()

actions = env.action_space.n
states = env.observation_space.n

eposides = 200000
epsilon = 0.8
gamma = 0.9
alpha = 0.01#lr

# Create Q table with all rewards = 0
q_table = np.zeros((states, actions))

# Training
total_reward = 0
count = 1
for i in range(eposides):
    env.reset()
    done = False
    state = 0
    steps = 0
    while not done:
        # epsilon-greedy 這裡是policy
        action = Greedy(env, q_table, epsilon, state)
        
        # Move one step
        next_state, reward, done, _ = env.step(action)
            
        # Update Q table
        q_table[state, action] = q_table[state, action] + alpha*(reward + gamma*max(q_table[next_state]) - q_table[state, action])
        state = next_state
        
        # Update statistics
        steps = steps + 1
    total_reward = total_reward + reward
    print("\r{}/{}\t{:.1%}, epsilon={:.3f}   ".format(i+1, eposides, total_reward/(i+1), epsilon),end="")
    
    if(reward > 0):
        count+=1
    if(count%10 == 0 and epsilon > 0.005):
        epsilon *= 0.99
        count=1
    if(not epsilon > 0.005):
        epsilon = 0.8

with open('Q_table.csv', 'w') as f:
    f.write('state,action0,action1,action2,action3\n')
    for state in range(len(q_table)):
        f.write('{},{},{},{},{}\n'.format(state, q_table[state, 0], q_table[state, 1], q_table[state, 2], q_table[state, 3]))

from IPython.display import FileLink, display
local_file = FileLink('Q_table.csv')

'''
======================================================================
test
======================================================================
'''
# Testing: Calculating the average reward of 1000 eposides
test_episodes = 1000 # DON'T CHANGE THIS VALUE
steps = 0
total_reward = 0
for i in range(test_episodes):
    env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        steps = steps + 1
        total_reward = total_reward + reward
    
print("The average results of {} episodes are steps {}, reward {}".format(test_episodes, steps/test_episodes, total_reward/test_episodes))

total_avg_reward = total_reward/test_episodes
# Print results in CSV format and upload to Kaggle
with open('rewards.csv', 'w') as f:
    f.write('Id,Predicted\n')
    f.write('FrozenLake8x8_public,{}\n'.format(total_avg_reward))
    f.write('FrozenLake8x8_private,{}\n'.format(total_avg_reward))

# Download your results!
FileLink('rewards.csv')