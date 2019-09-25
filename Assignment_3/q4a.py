# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import random
from copy import deepcopy
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import seaborn as sns

EPOCHS = 100000
ALPHA = 0.01
BLUR = 10
EPS = 0.01

class MDP:

	def __init__(self):

		self.reset()

	def reset(self):

		# index 0 is the face up card. Random numbers are drawn till 12 to simulate face cards
		self.dealer = []
		self.dealer.append(min(10, random.randint(1, 12)))
		self.dealer.append(min(10, random.randint(1, 12)))

		self.player = []
		self.player.append(min(10, random.randint(1, 12)))
		if(self.player[-1] == 1):
			self.player[-1] = 11
		self.player.append(min(10, random.randint(1, 12)))
		if(self.player[-1] == 1):
			self.player[-1] = 11
		while(self.value(self.player, False) < 12):
			self.player.append(min(10, random.randint(1, 12)))
			if(self.player[-1] == 1):
				self.player[-1] = 11
		
	def init_state(self):

		return [self.player_value(), self.dealer[0], self.has_ace()]

	def player_value(self):

		if(self.value(self.player, False) > 21 and self.has_ace()):
			assert(self.player.index(11) != -1)
			self.player[self.player.index(11)] = 1;

		return self.value(self.player, False)


	def has_ace(self):

		for val in self.player:
			if(val == 11):
				return 1
		return 0

	def value(self, player, count_ace):

		cur_sum = 0
		ace = False
		for val in player:
			if(val == 1):
				ace = True
			cur_sum += val
		if(ace and count_ace and cur_sum + 10 <= 21):
			cur_sum += 10
		return cur_sum

	def play(self, action):

		if(self.player_value() == 21):

			# Win on deck
			to_return = deepcopy(([22, self.dealer[0], self.has_ace()], self.value(self.dealer, True) != 21, True))
			self.reset()
			return to_return

		if(action):
			self.player.append(min(10, random.randint(1, 12)))
			if(self.player[-1] == 1):
				self.player[-1] = 11
			if(self.player_value() > 21):
				# player exceeds limit
				to_return = deepcopy(([22, self.dealer[0], self.has_ace()], -1, True))
				self.reset()
				return to_return
			else:
				return [self.player_value(), self.dealer[0], self.has_ace()], 0, False
		else:
			while(self.value(self.dealer, True) < 17):
				self.dealer.append(min(10, random.randint(1, 12)))
			
			# Identifying the player with high value
			if(self.value(self.dealer, True) > 21 or self.value(self.dealer, True) < self.player_value()):
				to_return = deepcopy(([22, self.dealer[0], self.has_ace()], 1, True))
			elif(self.value(self.dealer, True) == self.player_value()):
				to_return = deepcopy(([22, self.dealer[0], self.has_ace()], 0, True))
			else:
				to_return = deepcopy(([22, self.dealer[0], self.has_ace()], -1, True))

			self.reset()
			return to_return



if(__name__ == "__main__"):

	model = MDP()
	Q = np.zeros([31, 11, 2, 2])

	tot_reward = []
	
	for i in range(EPOCHS):

		cur_state = model.init_state()
		action = cur_state[0] < 20
		rng = random.uniform(0, 1)
		if(rng < EPS):
			action = random.randint(0, 1)
		done = False
		sum_reward = 0

		memory = []
		cum_reward = []

		while(not done):
			next_state, reward, done = model.play(action)
			
			memory.append([cur_state, action])
			cum_reward.append(reward)

			sum_reward += reward

			cur_state = next_state
			action = cur_state[0] < 20
			rng = random.uniform(0, 1)
			if(rng < EPS):
				action = random.randint(0, 1)

		cum_reward.append(0)

		for i in range(len(memory) - 1, -1, -1):

			cum_reward[i] += cum_reward[i + 1]
			Q[memory[i][0][0], memory[i][0][1], memory[i][0][2], memory[i][1]] = \
					(1 - ALPHA) * Q[memory[i][0][0], memory[i][0][1], memory[i][0][2], memory[i][1]] + \
					ALPHA * cum_reward[i] 

	for i in range(len(tot_reward)):
		tot_reward[i] = np.mean(tot_reward[i : i + BLUR])	

	plt.clf()

	X_axis = []
	Y_axis = []
	Z_axis = []

	for i in range(1, 11):
		for j in range(12, 22):
			X_axis.append(i)
			Y_axis.append(j)
			Z_axis.append(np.max(Q[j][i][0]))

	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.set_xlabel("Dealer showing")
	ax.set_ylabel("Player sum")
	ax.set_zlabel("Value function")
	ax.plot_trisurf(X_axis, Y_axis, Z_axis)
	fig.savefig("./Results/Q4/No_usable_ace_plot_fixed_policy.jpg")

	X_axis = []
	Y_axis = []
	Z_axis = []

	for i in range(1, 11):
		for j in range(12, 22):
			X_axis.append(i)
			Y_axis.append(j)
			Z_axis.append(np.max(Q[j][i][1]))

	plt.clf()
	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.set_xlabel("Dealer showing")
	ax.set_ylabel("Player sum")
	ax.set_zlabel("Value function")
	ax.plot_trisurf(X_axis, Y_axis, Z_axis)
	fig.savefig("./Results/Q4/Usable_ace_plot_fixed_policy.jpg")
