import numpy as np
from matplotlib import pyplot as plt
import math

# Args
EPOCHS = 10000
BED_SIZE = 100
MEAN_KERNEL_SIZE = 1
ALPHA = 0.1
DYNAMIC = True
BANDITS = 10



class Bandit:

	k = None					# Number of bandits
	bandit_params = None		# Probabilistic parameters of the bandit model

	def __init__(self, k):

		self.k = k

		# Start with a fixed value of bandit parameters
		self.bandit_params = np.asarray([[np.random.normal(0, 1), 1] for i in range(k)]).astype(float)

	def update_params(self):

		delta = np.asarray([[np.random.normal(0, 0.01), 0] for i in range(self.k)])
		self.bandit_params += delta

	def play(self, bandit):

		return np.random.normal(self.bandit_params[bandit][0], self.bandit_params[bandit][1])

	def No_of_Bandits(self):

		return self.k

	def best_bandit(self):

		return np.argmax(self.bandit_params[:,0])

class Agent:

	Q = None
	env = None
	latest_reward = []
	observations = []
	EPS = None
	alpha = None
	total_actions = 0

	def __init__(self, env, EPS, alpha, init_value):

		self.env = env
		
		self.Q = np.zeros(env.No_of_Bandits())
		for i in range(env.No_of_Bandits()):
			self.Q[i] = init_value

		self.observations = np.zeros(env.No_of_Bandits())
		self.latest_reward = np.zeros(env.No_of_Bandits())
		self.EPS = EPS
		self.alpha = alpha

	def updateQValues(self, bandit):

		if(self.alpha == 0):
			cur_alpha = 1 / self.observations[bandit]
		else:
			cur_alpha = self.alpha

		self.Q[bandit] = self.Q[bandit] * (1 - cur_alpha) + self.latest_reward[bandit] * cur_alpha

	def play(self):

		# Exploration vs Exploitation
		toss = np.random.uniform(0, 1)
		if(UCB):
			bandit = np.argmax(self.Q + SCALER * np.sqrt(math.log(1 + self.total_actions) * (1 / (1 + self.observations))))

		else:
			if(toss < self.EPS):
				bandit = np.random.randint(0, self.env.No_of_Bandits())
			else:
				bandit = np.argmax(self.Q)
		
		reward = self.env.play(bandit)

		self.observations[bandit] += 1
		self.latest_reward[bandit] = reward
		self.total_actions += 1

		self.updateQValues(bandit)

		return (bandit, reward, self.env.best_bandit())

def simulate():

	# 10 armed bandit
	env = Bandit(10)
	agent = [Agent(env, EPSILON, ALPHA, INIT_VALUE) for i in range(BED_SIZE)]

	performance = []
	reward = []

	for i in range(EPOCHS):

		opt_performance = 0
		epoch_reward = []

		for j in range(BED_SIZE):

			result = agent[j].play()
			opt_performance += result[0] == result[2]
			epoch_reward.append(result[1])

		if(DYNAMIC):
			env.update_params()

		performance.append((opt_performance * 100) / BED_SIZE)
		reward.append(np.mean(epoch_reward))

	return performance

if(__name__ == "__main__"):

	INIT_VALUE = 0
	EPSILON = 0.1
	UCB = False
	performance1 = simulate()

	INIT_VALUE = 5
	EPSILON = 0
	UCB = False
	performance2 = simulate()

	INIT_VALUE = 5
	EPSILON = 0
	UCB = True
	SCALER = 2
	performance3 = simulate()

	# Output smoothening
	performance1 = np.asarray(performance1[2:])
	for i in range(len(performance1)):
		performance1[i] = np.mean(performance1[i : i + MEAN_KERNEL_SIZE])

	iteration = [i + 1 for i in range(len(performance1))]

	# Output smoothening
	performance2 = np.asarray(performance2[2:])
	for i in range(len(performance2)):
		performance2[i] = np.mean(performance2[i : i + MEAN_KERNEL_SIZE])

	# Output smoothening
	performance3 = np.asarray(performance3[2:])
	for i in range(len(performance3)):
		performance3[i] = np.mean(performance3[i : i + MEAN_KERNEL_SIZE])

	iteration = [i + 1 for i in range(len(performance1))]

	plt.clf()
	plt.title("% Optimal action vs Epochs")
	plt.xlabel("Epochs")
	plt.ylabel("% Optimal action")
	axes = plt.gca()
	axes.set_ylim([0, 100])
	plt.plot(iteration, performance1, label = "Epsilon greedy")
	plt.plot(iteration, performance2, label = "Optimistic greedy")
	plt.plot(iteration, performance3, label = "UCB")
	plt.legend()
	plt.savefig("./Results/Q4/optaction_vs_epochs_nonstationary.jpg")
