import tensorflow as tf
import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_game_first():
  for episode in range(5):
    env.reset()
    for t in range(goal_steps):
      env.render()
      action = env.action_space.sample() #creating random action in an enviroment
      observation, reward, done, info = env.step(action)
      if done:
        break


def initial_population():
	training_data = []
	scores = []
	accepted_scores = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		prev_observation = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])

			prev_observation =observation
			score += reward
			if done:
				break

		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0,1]
				elif data[1] == 0:
					output = [1,0]

				training_data.append([data[0], output])

		env.reset() #when the game is over we set
		scores.append(score) # Keeping track of all the scores 

	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)

	print('Average accepted score :', mean(accepted_scores))
	print('Median accepted score :',median(accepted_scores))
	print(Counter(accepted_scores))

	return training_data


def neural_network_model(input_size):
	Network = input_data(shape = [None, input_size, 1], name = 'inputs')

	Network = fully_connected(Network, 128, activation = 'relu')
	Network = dropout(Network, 0.8)

	Network = fully_connected(Network, 256, activation = 'relu')
	Network = dropout(Network, 0.8)

	Network = fully_connected(Network, 512, activation = 'relu')
	Network = dropout(Network, 0.8)

	Network = fully_connected(Network, 256, activation = 'relu')
	Network = dropout(Network, 0.8)

	Network = fully_connected(Network, 128, activation = 'relu')
	Network = dropout(Network, 0.8)

	Network = fully_connected(Network, 2, activation = 'softmax')
	Network = regression(Network, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'target')

	model = tflearn.DNN(Network, tensorboard_dir = 'log')

	return model

def train_model(train_model, model = False):
	x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size = len(x[0]))

	model.fit(X_inputs=x, Y_targets=y, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')

	return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(100):
	score = 0
	game_memory = []
	prev_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(prev_obs) == 0:
			action = random.randrange(0,2)
		else:
			action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1)) [0])
		choices.append(action)

		new_observation, reward, done, info = env.step(action)
		prev_obs = new_observation
		game_memory.append([new_observation, action])
		score += reward

		if done:
			break
	scores.append(score)

print('Average score', sum(scores)/len(scores))
print('Choice 1: {} Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))