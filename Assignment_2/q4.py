# Author: Nihesh Anderson
# Date	: Sept 7, 2019

import numpy as np
import scipy.optimize

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
		self.special_transition[str(np.asarray([0, 1]).astype(int))] = (np.asarray([4, 1]).astype(int), 10)
		self.special_transition[str(np.asarray([0, 3]).astype(int))] = (np.asarray([2, 3]).astype(int), 5)


	def action_space(self):

		# Returns the action space
		return self.actions

	def improvePolicy(self, value_fn):

		"""
		Given the current value function, this module updates the action probability vector to the greedy policy with respect to the value function 
		"""

		self.action_probability = np.zeros([25, 4])

		# Iterate over all states
		for i in range(25):
			
			argmax_action = []
			max_action = -100000
			cur_state = np.asarray([i / 5, i % 5]).astype(int)

			# Computing the highest value 
			for j in range(self.actions.shape[0]):
				
				next_state, reward = self.play(cur_state, self.actions[j])
				next_state_id = next_state[0] * 5 + next_state[1]

				if(value_fn[next_state_id] > max_action):
					max_action = value_fn[next_state_id]

			# Computing the set of actons corresponding to the highest value
			for j in range(self.actions.shape[0]):
				
				next_state, reward = self.play(cur_state, self.actions[j])
				next_state_id = next_state[0] * 5 + next_state[1]

				if(value_fn[next_state_id] == max_action):
					argmax_action.append(j)

			# Setting non zero probabilites to the action set with the higest value
			for action in argmax_action:
				self.action_probability[i][action] = 1 / len(argmax_action)

	def play(self, state, action):

		"""
		MDP simulation that returs the next state and reward given the current state and action
		"""

		next_state = state + action
		next_state = next_state.astype(int)

		if(str(state) in self.special_transition):
			return self.special_transition[str(state)]

		if(min(next_state) < 0 or max(next_state) >= self.grid_size):
			return (state, -1)
		else:
			return (next_state, 0)

def LinearSystemSolver(C, linear_system, bias):

	"""
	LP Solver 
	"""

	return scipy.optimize.linprog(C, linear_system, bias).x

if(__name__ == "__main__"):

	env = MDP()

	C = np.ones([25])
	linear_system = np.zeros([100, 25])
	bias = np.zeros([100])

	# Building the linear system Ax <= B (here A = linear_system variable and B = bias)
	for i in range(5):
		for j in range(5):
			cur_state = np.asarray([i, j]).astype(int)
			equation_id = 5 * i + j

			actions = env.action_space()
			
			for action_id in range(len(actions)):
				next_state, reward = env.play(cur_state, actions[action_id])

				# LHS variable V(S) is moved to RHS
				linear_system[equation_id * 4 + action_id][5 * cur_state[0] + cur_state[1]] += -1
				
				# The term in RHS that corresponds to GAMMA * V(S'|S,a)
				linear_system[equation_id * 4 + action_id][5 * next_state[0] + next_state[1]] += GAMMA
				
				# The reward in RHS is moved to LHS
				bias[equation_id * 4 + action_id] -= reward
				
	value_function = LinearSystemSolver(C, linear_system, bias)

	# Rounding the value function and updating policy
	env.improvePolicy(np.round(value_function, 1))

	print("Optimal state value function")
	print(np.round(value_function.reshape(5, 5), 1))

	print("Optimal actions: N S W E")
	for i in range(5):
		for j in range(5):
			print("State","[",i,j,"] -",env.action_probability[i * 5 + j])

