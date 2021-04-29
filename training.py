import tensorflow as tf
import numpy as np
#import pokemon

class ReplayBuffer():
	def __init__(self, max_size, input_dims):
		self.mem_size = max_size
		self.mem_counter = 0

		self.state_memory = np.zeros((self.mem_size, input_dims),
										dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size, input_dims),
										dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
		self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

	def store_transition(self, state, action, reward, newstate, done):
		index = self.mem_counter % self.mem_size
		self.state_memory[index] = state
		self.new_state_memory[index] = newstate
		self.reward_memory[index] = reward
		self.action_memory[index] = action
		self.terminal_memory[index] = 1 - int(done)
		self.mem_counter += 1

	def sample_buffer(self, batch_size):
		mem_filled = min(self.mem_size, self.mem_counter)
		batch = np.random.choice(mem_filled, batch_size, replace=False)

		states = self.state_memory[batch]
		new_states = self.new_state_memory[batch]
		rewards = self.reward_memory[batch]
		actions = self.action_memory[batch]
		terminal = self.terminal_memory[batch]

		return states, actions, rewards, new_states, terminal

def build_dqn(learn_rate, n_actions, input_dims, fc1_dims, fc2_dims):
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(fc1_dims, activation='relu'),
		tf.keras.layers.Dense(fc2_dims, activation='relu'),
		tf.keras.layers.Dense(n_actions, activation=None)])

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
					loss='mean_squared_error')

	return model

class Agent():
	def __init__(self, learn_rate, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_decrement = 0.001, epsilon_end = 0.01, mem_size = 1000000, fname = 'dqn_model.h5'):
		self.action_space = [i for i in range(n_actions)]
		self.learn_rate = learn_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_min = epsilon_end
		self.epsilon_end = epsilon_end
		self.epsilon_decrement = epsilon_decrement
		self.batch_size = batch_size
		self.model_file = fname
		self.memory = ReplayBuffer(mem_size, input_dims)
		self.q_eval = build_dqn(learn_rate, n_actions, input_dims, 256, 256)

	def store_transition(self, state, action, reward, new_state, done):
		self.memory.store_transition(state, action, reward, new_state, done)

	def choose_action(self, observation):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			state = np.array([observation])
			actions = self.q_eval.predict(state)

			action = np.argmax(actions)

		return action

	def learn(self):
		if self.memory.mem_counter < self.batch_size:
			return

		states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

		q_eval = self.q_eval.predict(states)
		q_next = self.q_eval.predict(new_states)

		q_target = np.copy(q_eval)
		batch_index = np.arange(self.batch_size, dtype=np.int32)

		q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones

		self.q_eval.train_on_batch(states, q_target)

		if self.epsilon > self.epsilon_min:
			self.epsilon = self.epsilon - self.epsilon_decrement
		else:
			self.epsilon = self.epsilon_min

	def save_model(self):
		raise Exception("Not yet implemented")

	def load_model(self):
		raise Exception("Not yet implemented")

class pokemon:

	def __init__(self):
		self.state = np.array([0, 0, 0, 0])
		self.counter = 0

	def fitness(self, side):
		if side == 0:
			return float(self.state[0]) - float(self.state[3] / 2.0)
		else:
			return float(self.state[2]) - float(self.state[1] / 2.0)

	def step(self, action):
		self.counter += 1
		if action == 0:
			self.state[0] += 1
		elif action == 1:
			self.state[0] -= 1
		elif action == 2:
			self.state[1] += 1
		elif action == 3:
			self.state[1] -= 1
		print(self.state)
		return self.state, 1 if self.state[0] == 1 else -1, self.counter == 100

	def getstate(self):
		return self.state

	def reset(self):
		self.state = np.array([0, 0, 0, 0])
		self.counter = 0


# ---------- Constants ----------
learning_rate = 0.001
num_games = 500 # Number of full games to simulate
switch_steps = 10000 # Switch sides after n steps in the training process
max_steps = 10000000 # Maximum number of iterations to perform training

reward_game_win = 1000 # Reward when the AI wins a game
reward_game_loss = -1000 # Reward when the AI loses a game

space_size = 4 # Number of parameters in the observation space
action_size = 4 # Number of actions that can be performed by the player

# ---------- Global variables ----------
train_a = True # Set to True if player A is the one that's currently training

def switch_training():
	train_a = not train_a


def main():
	# Create a model
	model_a = Agent(gamma=0.99, epsilon=0.1, learn_rate=learning_rate,
						input_dims=space_size, n_actions=action_size,
						mem_size=1000000, batch_size=64, epsilon_end=0.01)
	model_b = Agent(gamma=0.99, epsilon=0.1, learn_rate=learning_rate,
						input_dims=space_size, n_actions=action_size,
						mem_size=1000000, batch_size=64, epsilon_end=0.01)

	scores = []
	epsilon_history = []

	game = pokemon()

	# Play n games
	for current_game in range(num_games):
		print('New Game', current_game)
		done = False
		score = 0

		# Create a new instance of the pokemon game simulator
		game.reset()
		observation = game.getstate()

		# Store how many steps player A/B has made since they started being the one training
		current_switch_step = 0

		current_model = model_a if train_a else model_b

		while not done:
			action = current_model.choose_action(observation)
			new_observation, reward, done = game.step(action)
			score += reward
			current_model.store_transition(observation, action, reward, new_observation, done)
			observation = new_observation
			current_model.learn()

		# Increment or reset switch_step
		if current_switch_step == switch_steps:
			current_switch_step = 0
			print("Switching training from player", "A" if train_a else "B", "to player", "B" if train_a else "A")
			# Perform switch so that the opponent begins training
			switch_training()
		else:
			current_switch_step += 1

		epsilon_history.append(current_model.epsilon)
		scores.append(score)

		avg_score = np.mean(scores[-50:])
		print('Episode: ', current_game, 'Score %.2f' % score,
			  'average_score %.2f' % avg_score,
			  'epsilon %.2f' % current_model.epsilon)


if __name__ == '__main__':
	main()
