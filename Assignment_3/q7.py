# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import random
from copy import deepcopy
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import seaborn as sns

EPOCHS = 1000
ALPHA = 0.01
EPS = 0.01
FILTER = 5

class MDP:

	def __init__(self):

		self.action_space = np.asarray([[-1, 0], [1, 0], [0, -1], [0, 1]]).astype(int)
		self.reset()

	def reset(self):

		self.world = np.zeros([7, 6])
		for i in range(1, 6, 2):
			for j in range(6):
				self.world[i][j] = 1
		self.world[1][2] = self.world[1][4] = 0
		self.world[3][2] = self.world[3][5] = 0
		self.world[5][3] = self.world[5][0] = 0

		self.now = np.zeros(2).astype(int)
		
	def cur_state(self):

		return self.now

	def play(self, action):

		next_state = self.now + self.action_space[action]
		if(next_state[0] < 0 or next_state[0] >= 7 or next_state[1] < 0 or next_state[1] >= 6):
			next_state = self.now
		
		self.now = next_state
		if(self.now[0] == 6 and self.now[1] == 5):
			return self.now, 11, True
		else:
			return self.now, -1, False


if(__name__ == "__main__"):

	model = MDP()
	Q = np.zeros([7, 6, 4])
	
	cum_reward = []

	for i in range(EPOCHS):

		model.reset()
		done = False
		tot_reward = 0

		while(not done):

			cur_state = model.cur_state()
			action = np.argmax(Q[cur_state[0], cur_state[1]])
			rng = random.uniform(0, 1)
			if(rng < EPS):
				action = random.randint(0, 3)

			next_state, reward, done = model.play(action)
			Q[cur_state[0], cur_state[1], action] = (1 - ALPHA) * Q[cur_state[0], cur_state[1], action] + ALPHA * (reward + np.max(Q[next_state[0], next_state[1]]))

			tot_reward += reward

		cum_reward.append(tot_reward)

	X_axis = [i for i in range(len(cum_reward))]
	for i in range(len(cum_reward)):
		cum_reward[i] = np.mean(cum_reward[i : i + FILTER])

	plt.plot(X_axis, cum_reward, label = "Q learning")
	plt.xlabel("Epochs")
	plt.ylabel("Total reward - (-1 for a step and 11 for reaching goal state 6,5)")
	
	model = MDP()
	Q = np.zeros([7, 6, 4])
	
	cum_reward = []

	for i in range(EPOCHS):

		model.reset()
		done = False
		tot_reward = 0

		backup_Q = Q

		while(not done):

			cur_state = model.cur_state()
			action = np.argmax(backup_Q[cur_state[0], cur_state[1]])
			rng = random.uniform(0, 1)
			if(rng < EPS):
				action = random.randint(0, 3)

			next_state, reward, done = model.play(action)
			next_Q_value = 0
			if(not done):
				action_new = np.argmax(backup_Q[next_state[0], next_state[1]])
				rng = random.uniform(0, 1)
				if(rng < EPS):
					action_new = random.randint(0, 3)
				next_Q_value = backup_Q[next_state[0], next_state[1]][action_new]

			Q[cur_state[0], cur_state[1], action] = (1 - ALPHA) * Q[cur_state[0], cur_state[1], action] + ALPHA * (reward + next_Q_value)

			tot_reward += reward

		cum_reward.append(tot_reward)

	X_axis = [i for i in range(len(cum_reward))]
	for i in range(len(cum_reward)):
		cum_reward[i] = np.mean(cum_reward[i : i + FILTER])

	plt.plot(X_axis, cum_reward, label = "SARSA")
	plt.legend()
	plt.savefig("./Results/Q7/reward.jpg")