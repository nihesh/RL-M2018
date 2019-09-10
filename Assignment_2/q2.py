# Author: Nihesh Anderson
# Date	: Sept 7, 2019

import numpy as np

GAMMA = 0.9

class MDP:

	def __init__(self):

		self.grid_size = 5
		
		# Possible actions
		self.actions = np.zeros([4, 2]).astype(int)
		self.actions[0][0] = -1
		self.actions[1][0] = 1
		self.actions[2][1] = -1
		self.actions[3][1] = 1

		# Special transitions from A to A' and B to B'
		self.special_transition = {}
		self.special_transition[str(np.asarray([0, 1]))] = (np.asarray([4, 1]), 10)
		self.special_transition[str(np.asarray([0, 3]))] = (np.asarray([2, 3]), 5)


	def action_space(self):

		return self.actions

	def play(self, state, action):

		"""
		MDP simulator that returns the next state and reward given the current state and action
		"""

		next_state = state + action
		
		if(str(state) in self.special_transition):
			return self.special_transition[str(state)]

		if(min(next_state) < 0 or max(next_state) >= self.grid_size):
			return (state, -1)
		else:
			return (next_state, 0)

def LinearSystemSolver(linear_system, bias):

	# Solution to Ax = B is x = inv(A) * B
	
	bias = bias.reshape(25, 1)
	return np.matmul(np.linalg.inv(linear_system), bias).reshape(5, 5)

if(__name__ == "__main__"):

	env = MDP()

	linear_system = np.zeros([25, 25])
	bias = np.zeros([25])

	# Forming the linear system considering the items in value function as variables and relating them using bellman equation
	for i in range(5):
		for j in range(5):
			cur_state = np.asarray([i, j])
			equation_id = 5 * i + j

			# -1 because the LHS goes to RHS in bellman equation
			linear_system[equation_id][5 * cur_state[0] + cur_state[1]] += -1
			actions = env.action_space()
			
			for action in actions:
				next_state, reward = env.play(cur_state, action)

				# Future is added because it stays in RHS of bellman equation
				linear_system[equation_id][5 * next_state[0] + next_state[1]] += 0.25 * GAMMA
				
				# bias is subtracted because it goes from RHS to LHS in bellman equation
				bias[equation_id] -= 0.25 * reward

	value_function = LinearSystemSolver(linear_system, bias)
	print(np.round(value_function, 1))