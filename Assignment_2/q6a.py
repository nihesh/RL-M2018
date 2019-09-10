# Author: Nihesh Anderson
# Date	: Sept 7, 2019

"""
This module implements policy iteration algorithm.
It has been shown that as iterations increase, the value of every state increases, depicting positive learning
"""

import numpy as np

EPOCHS = 5

class MDP:

	def __init__(self):

		self.grid_size = 4
		
		# Possible actions
		self.actions = np.zeros([4, 2]).astype(int)
		self.actions[0][0] = -1
		self.actions[1][0] = 1
		self.actions[2][1] = -1
		self.actions[3][1] = 1

		# Initializing a uniform random policy
		self.action_probability = np.zeros([16, 4])
		for i in range(self.action_probability.shape[0]):
			for j in range(self.action_probability.shape[1]):
				self.action_probability[i][j] = 0.25

	def action_space(self):

		return self.actions

	def improvePolicy(self, value_fn):

		"""
		Finds the greedy policy given the current value function
		"""

		self.action_probability = np.zeros([16, 4])
		for i in range(16):
			
			argmax_action = []
			max_action = -100000
			cur_state = np.asarray([i / 4, i % 4]).astype(int)

			# Computing the highest value given the actions
			for j in range(self.actions.shape[0]):
				
				next_state, reward = self.play(cur_state, self.actions[j])

				if(value_fn[next_state[0]][next_state[1]] > max_action):
					max_action = value_fn[next_state[0]][next_state[1]]

			# Finding the set of actions that corresponds to the higest value
			for j in range(self.actions.shape[0]):
				next_state, reward = self.play(cur_state, self.actions[j])

				if(value_fn[next_state[0]][next_state[1]] == max_action):
					argmax_action.append(j)

			# Updating the policy corresponding to highest value actions
			for action in argmax_action:
				self.action_probability[i][action] = 1 / len(argmax_action)

	def play(self, state, action):

		# MDP simulation

		next_state = state + action

		if((next_state[0] == 0 and next_state[1] == 0) or (next_state[0] == 3 and next_state[1] == 3)):	
			return(np.asarray([4, 4]).astype(int), -1)

		if(min(next_state) < 0 or max(next_state) >= self.grid_size):
			return (state, -1)
		else:
			return (next_state, -1)

if(__name__ == "__main__"):

	env = MDP()
	value_function = np.zeros([5, 5])
	for i in range(5):
		for j in range(5):
			value_function[i][j] = 0

	# Value function in the previous iteration to compute delta change
	prev_value_function = np.copy(value_function)
	global_prev_value_function = np.copy(value_function)

	epoch = 1

	# Policy evaluation and improvement steps are contained within this loop
	while(True):
		
		# Policy evaluation
		while(True):

			# Update the state value for each state
			for i in range(4):
				for j in range(4):

					# End state conditions
					if((i==0 and j==0) or (i==3 and j==3)):
						value_function[i][j] = 0
						continue

					cur_state = np.asarray([i, j]).astype(int)
					equation_id = 4 * i + j

					actions = env.action_space()
					
					value = 0
					for action_id in range(len(actions)):
						next_state, reward = env.play(cur_state, actions[action_id])
						
						# Value function update
						value += env.action_probability[equation_id][action_id] * (reward + value_function[next_state[0]][next_state[1]])

					value_function[i][j] = value

			if(np.linalg.norm(value_function - prev_value_function, 1) <= 0.5):
				break

			prev_value_function = np.copy(value_function)

		# Policy improvement
		env.improvePolicy(value_function)

		# If there is no change in the value function, break. This fixes policy oscillation bug mentioned in Ex 4.4
		if(np.linalg.norm(value_function - global_prev_value_function) <= 0.5):
			break

		global_prev_value_function = np.copy(value_function)

		print("Epoch", epoch)
		print(np.round(value_function[:4,:4], 1))
		print()
		epoch += 1