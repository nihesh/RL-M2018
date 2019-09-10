# Author: Nihesh Anderson
# Date	: Sept 7, 2019

"""
This module implements policy iteration algorithm.
"""

import numpy as np
import scipy.stats
import pickle
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

MAX_STATE = 20
MAX_CARS = 3
GAMMA = 0.9
MAX_ACTION = 5
MAX_PARKING_SPACE = 10
MEAN_POISSON = [3, 3, 4, 2]

class MDP:

	def __init__(self):
		
		# Possible actions
		self.actions = np.asarray([i for i in range(-MAX_ACTION, MAX_ACTION + 1)]).astype(int)

		# Initial action = no transfer 
		self.action_probability = np.zeros([MAX_STATE, MAX_STATE, 2 * MAX_ACTION + 1])
		for i in range(MAX_STATE):
			for j in range(MAX_STATE):
				self.action_probability[i][j][MAX_ACTION] = 1

	def action_space(self):

		return self.actions

	def improvePolicy(self, value_fn, transitions):

		self.action_probability = np.zeros([MAX_STATE, MAX_STATE, 2 * MAX_ACTION + 1])
		
		for i in range(MAX_STATE):
			for k in range(MAX_STATE):
			
				argmax_action = []
				max_action = -1000000000
				cur_state = np.asarray([i, k]).astype(int)

				# Computing the set of highest value action

				for j in range(self.actions.shape[0]):

					# Next state is a random variable, given the current state. We compute the expected reward, given the current state and action and pick that action that maximises the expectation

					value = 0			
					normalization_const = 0
					for random_behaviour in transitions:
						next_state, reward = self.play(cur_state, self.actions[j], random_behaviour[0][0], random_behaviour[0][1])
						value += random_behaviour[1] * (reward + GAMMA * value_fn[next_state[0]][next_state[1]]) 
						normalization_const += random_behaviour[1]

					# Expected value given the state and action
					value /= normalization_const	

					if(value > max_action):
						max_action = value

				# Compute the argmax action
				for j in range(self.actions.shape[0]):
					
					value = 0
					
					normalization_const = 0
					for random_behaviour in transitions:
						next_state, reward = self.play(cur_state, self.actions[j], random_behaviour[0][0], random_behaviour[0][1])
						value += random_behaviour[1] * (reward + GAMMA * value_fn[next_state[0]][next_state[1]]) 
						normalization_const += random_behaviour[1]

					value /= normalization_const

					if(value == max_action):
						argmax_action.append(j)

				self.action_probability[i][k][argmax_action[0]] = 1 

	def play(self, state, action, first_location, second_location):

		# MDP simulation

		# Invalid action, high penalty
		if(state[0] - action < 0 or state[1] + action < 0):
			return (state, -10000000)

		next_state = np.copy(state) 
		# Check car count overflow
		next_state[0] = min(MAX_STATE - 1, next_state[0] - action)
		next_state[1] = min(MAX_STATE - 1, next_state[1] + action)
		
		# Deduct 2$ for car movement
		cost = -2 * abs(action)

		# Free shuttle case
		if(action >= 1):
			cost += 2

		# Daily business, 10$ per car rented
		cost += 10 * min(next_state[0], first_location[0])
		next_state[0] -= min(next_state[0], first_location[0])
		cost += 10 * min(next_state[1], second_location[0])
		next_state[1] -= min(next_state[1], second_location[0])

		# Returned cars, overflow handled
		next_state[0] = min(MAX_STATE - 1, next_state[0] + first_location[1])
		next_state[1] = min(MAX_STATE - 1, next_state[1] + second_location[1])

		# Parking charges
		cost -= 4 * min(1, max(0, next_state[0] - MAX_PARKING_SPACE))
		cost -= 4 * min(1, max(0, next_state[1] - MAX_PARKING_SPACE))

		return next_state, cost

if(__name__ == "__main__"):

	env = MDP()
	value_function = np.zeros([MAX_STATE, MAX_STATE])
	for i in range(value_function.shape[0]):
		for j in range(value_function.shape[1]):
			value_function[i][j] = 0

	prev_value_function = np.copy(value_function)
	global_prev_value_function = np.copy(value_function)

	# Computing the joint distribution of the 4 tuple independent poisson RVs
	transitions = []
	for i in range(MAX_CARS):
		for j in range(MAX_CARS):
			for k in range(MAX_CARS):
				for l in range(MAX_CARS):
					transitions.append([[[i,j], [k,l]], scipy.stats.poisson.pmf(i, MEAN_POISSON[0]) * scipy.stats.poisson.pmf(i, MEAN_POISSON[1]) * scipy.stats.poisson.pmf(i, MEAN_POISSON[2]) * scipy.stats.poisson.pmf(i, [3])])

	epoch = 0
	while(True):
		
		epoch += 1

		# Policy evaluation
		while(True):

			for i in range(MAX_STATE):
				for j in range(MAX_STATE):

					cur_state = np.asarray([i, j]).astype(int)

					actions = env.action_space()
					
					value = 0
					for action_id in range(len(actions)):

						cur_value = 0
						normalization_const = 0

						# Averaging the next state random variable
						for random_behaviour in transitions:

							next_state, reward = env.play(cur_state, actions[action_id], random_behaviour[0][0], random_behaviour[0][1])
							
							# Value function update
							cur_value += env.action_probability[i][j][action_id] * random_behaviour[1] * (reward + GAMMA * value_function[next_state[0]][next_state[1]])
							normalization_const += random_behaviour[1]

						cur_value /= normalization_const
						value += cur_value

					value_function[i][j] = value

			if(np.linalg.norm(value_function - prev_value_function, 1) <= 0.5):
				break

			prev_value_function = np.copy(value_function)

		print("Epoch",epoch)
		print(np.round(value_function, 1))
		print()

		# Policy improvement
		env.improvePolicy(value_function, transitions)

		# If there is no change in the value function, break. This fixes policy oscillation bug mentioned in Ex 4.4
		if(np.linalg.norm(value_function - global_prev_value_function, 1) <= 0.5):
			break

		global_prev_value_function = np.copy(value_function)

	print("Final value function")
	print(np.round(value_function, 1))
	print()

	print("Action space")
	for i in range(MAX_STATE):
		for j in range(MAX_STATE):
			print(np.argmax(env.action_probability[i][j]) - MAX_ACTION, end = " ")
		print()

	data = []
	for i in range(MAX_STATE):
		for j in range(MAX_STATE):
			data.append([i, j, value_function[i][j]])
	data = np.asarray(data)

	fig = plt.figure()
	ax = plt.axes(projection="3d")
	ax.set_xlabel("cars at first location")
	ax.set_ylabel("cars at second location")
	ax.set_zlabel("value function")
	ax.plot_trisurf(data[:,0], data[:,1], data[:,2])
	fig.savefig("q7_plot.jpg")

	data = np.zeros([MAX_STATE, MAX_STATE]).astype(int)
	for i in range(MAX_STATE):
		for j in range(MAX_STATE):
			data[j][i] = np.argmax(env.action_probability[i][j]) - MAX_ACTION

	plt.clf()
	fig, ax = plt.subplots()
	ax.imshow(data)
	x = [i for i in range(len(data))]
	ax.set_xticks(np.arange(len(x)))
	ax.set_yticks(np.arange(len(x)))
	ax.set_xticklabels(x)
	ax.set_yticklabels(x)
	ax.set_xlabel("second location")
	ax.set_ylabel("first location")
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			ax.text(i, j, str(data[i][j]), ha = "center", va = "center", color = "w")
	fig.savefig("q7_action.jpg")
