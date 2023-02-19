import numpy as np
import random
import sys

global qtable
global N

N = 20

def roll():
    return random.choice([1,2,3,4,5,6])

# scenario 1
def player2play():
    score = 0
    while score<N:
        n = roll()
        if n!=0:
            score+=n
        else:
            return 0
    return 1


def step(state, a):
    done=False
    reward = 0
    if a==0:
        # roll
        n = roll()
        if n==1 and state!=0:
            reward = -1
            new_state = 0
        else:
            reward = 1
            new_state = state + n

    else:
        # hold
        new_state = state
        # player 2 can play
        outcome = player2play()
        if outcome == 1:
            reward = -1
            done = True
        else:
            reward = 1

    if new_state>=N:
        done=True
        reward=10
        new_state = N

    return new_state, done, reward




qtable = np.zeros((N+1, 2))

learning_rate = 0.8
discount_rate = 0.6
# explore a lot at the start
epsilon = 0.9
decay_rate= 0.005

num_episodes = 100
max_steps = 20

print("Training agent...")
for episode in range(num_episodes):
    done = False
    # reset start state
    state = 0
    for s in range(max_steps):
        if random.uniform(0,1) < epsilon:
            # explore
            action = random.choice([0,1])
        else:
            # exploit
            action = np.argmax(qtable[state,:])

        # perfom action
        new_state, done, reward = step(state, action)
       
        # update q table
        qtable[state, action] += learning_rate * (reward + discount_rate *
                                            np.max(qtable[new_state,:])-qtable[state,action]).round(2)


        state = new_state
        if done == True:
            break

        
    epsilon = np.exp(-decay_rate*episode)


print("Agent playing naive player...")
matches = 1000
pl1, pl2 = 0, 0
for i in range(matches):
    done = False
    state = 0
    while not done:
        action = np.argmax(qtable[[state],:])
        new_state, done, reward = step(state, action)
        if done:
            if new_state<20:
                # new state will be 0 if we lost
                pl2 += 1
            else:
                pl1 += 1


        state = new_state

print("Results:")
print("Player 1 won: " + str(pl1*100/matches) + "% of the time")
print("Player 2 won: " + str(pl2*100/matches) + "% of the time")
