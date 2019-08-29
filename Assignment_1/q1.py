import numpy as np
from matplotlib import pyplot as plt

# Args
EPOCHS = 10000
BED_SIZE = 100
MEAN_KERNEL_SIZE = 1
EPSILON = 0.1
ALPHA = 0			# Change Alpha to 0 for part a and 0.1 for part b
DYNAMIC = True
BANDITS = 10


class Bandit:

	k = None					# Number of bandits
	bandit_params = None		# Probabilistic parameters of the bandit model

	def __init__(self, k):

		self.k = k

		# Start with a fixed value of bandit parameters
		self.bandit_params = np.asarray([[0, 1] for i in range(k)]).astype(float)

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

	def __init__(self, env, EPS, alpha = 0):

		self.env = env
		self.Q = np.zeros(env.No_of_Bandits())
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
		if(toss < self.EPS):
			bandit = np.random.randint(0, env.No_of_Bandits())
		else:
			bandit = np.argmax(self.Q)
		
		reward = env.play(bandit)
		self.observations[bandit] += 1
		self.latest_reward[bandit] = reward

		self.updateQValues(bandit)

		return (bandit, reward, env.best_bandit())


if(__name__ == "__main__"):

	# 10 armed bandit
	env = Bandit(10)
	agent = [Agent(env, EPSILON, ALPHA) for i in range(BED_SIZE)]

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

	# Output smoothening
	performance = np.asarray(performance[2:])
	reward = np.asarray(reward[2:])
	for i in range(len(performance)):
		performance[i] = np.mean(performance[i : i + MEAN_KERNEL_SIZE])
	for i in range(len(reward)):
		reward[i] = np.mean(reward[i : i + MEAN_KERNEL_SIZE])

	iteration = [i + 1 for i in range(len(performance))]

	plt.clf()
	plt.title("% Optimal action vs Epochs")
	plt.xlabel("Epochs")
	plt.ylabel("% Optimal action")
	axes = plt.gca()
	axes.set_ylim([0, 100])
	plt.plot(iteration, performance)
	plt.savefig("./Results/Q1/optaction_vs_epochs_meanbased.jpg")

	plt.clf()
	plt.title("% Reward vs Epochs")
	plt.xlabel("Epochs")
	plt.ylabel("Reward")
	axes = plt.gca()
	axes.set_ylim([0, 2])
	plt.plot(iteration, reward)
	plt.savefig("./Results/Q1/reward_vs_epochs_meanbased.jpg")
