# Author: Nihesh Anderson
# File	: q6.py

import random
import numpy as np
from matplotlib import pyplot as plt

MC_EPOCHS = 1000
TL_EPOCHS = 1000
ALPHA = 0.01

class MDP:

	def __init__(self):
		self.cur_state = 3

	def reset(self):
		self.cur_state = 3

	def play(self):

		num = random.uniform(0, 1)
		if(num < 0.5):
			self.cur_state -= 1
		else:
			self.cur_state += 1

		if(self.cur_state == 0):
			return self.cur_state, 0, True
		elif(self.cur_state == 6):
			return self.cur_state, 1, True
		else:
			return self.cur_state, 0, False

def MonteCarlo(actual_V):

	V = np.zeros(7)
	for i in range(5):
		V[i + 1] = 0.5

	model = MDP()
	rmse = []

	for i in range(MC_EPOCHS):

		memory = []
		
		model.reset()
		cur_state = 3
		next_state, reward, done = model.play()
		memory.append([cur_state, next_state, reward])

		while(not done):
			cur_state = next_state
			next_state, reward, done = model.play()
			memory.append([cur_state, next_state, reward])

		cur_reward = 0

		for i in range(len(memory) - 1, -1, -1):

			cur_reward += memory[i][2]
			V[memory[i][0]] = (1 - ALPHA) * V[memory[i][0]] + ALPHA * (cur_reward)

		rmse.append(np.linalg.norm(V[:6] - actual_V, 2))

	return V[:6], rmse

def TemporalLearning(actual_V):

	V = np.zeros(7)
	for i in range(5):
		V[i + 1] = 0.5

	model = MDP()
	rmse = []

	for i in range(TL_EPOCHS):
	
		model.reset()
		cur_state = 3
		next_state, reward, done = model.play()
		V[cur_state] = (1 - ALPHA) * V[cur_state] + ALPHA * (reward + V[next_state])

		while(not done):
			cur_state = next_state
			next_state, reward, done = model.play()
			V[cur_state] = (1 - ALPHA) * V[cur_state] + ALPHA * (reward + V[next_state])

		rmse.append(np.linalg.norm(V[:6] - actual_V, 2))

	return V[:6], rmse

if(__name__ == "__main__"):

	V = np.zeros(6)
	for i in range(5):
		V[i + 1] = (i + 1) / 6

	V1, rmse1 = MonteCarlo(V)

	V2, rmse2 = TemporalLearning(V)

	X_axis = np.asarray([i for i in range(6)])

	plt.plot(X_axis, V, label = "true value")
	plt.ylabel("expected reward")
	plt.xlabel("state")
	plt.plot(X_axis, V1, label = "monte carlo estimate")
	plt.plot(X_axis, V2, label = "TD estimate")
	plt.legend()
	plt.savefig("./Results/Q6/state_value_plot.jpg")

	X_axis = np.asarray([i for i in range(TL_EPOCHS)])
	plt.clf()
	plt.ylabel("RMSE")
	plt.xlabel("Episodes")
	plt.plot(X_axis, rmse1, label = "monte carlo")
	plt.plot(X_axis, rmse2, label = "temporal difference")
	plt.legend()
	plt.savefig("./Results/Q6/RMSE.jpg")