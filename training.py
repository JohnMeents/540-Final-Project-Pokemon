import tensorflow as tf
import numpy as np
#import pokemon

class pokemon:

	def __init__(self):
		self.state = [0, 0, 0, 0]

	def fitness(side):
		if side == 0:
			return float(self.state[0]) - float(self.state[3] / 2.0)
		else:
			return float(self.state[2]) - float(self.state[1] / 2.0)

	def state():
		return self.state





# ---------- Constants ----------
switch_steps = 10000 # Switch sides after n steps in the training process
max_steps = 10000000 # Maximum number of iterations to perform training

reward_game_win = 1000 # Reward when the AI wins a game
reward_game_loss = -1000 # Reward when the AI loses a game

# ---------- Global variables ----------
train_a = True # Set to True if player A is the one that's currently training

def construct_model():
	# Construct the model
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(4, activation='relu'),
		tf.keras.layers.Dense(2)
	])
	
	# Compile the model
	model.compile(optimizer='adam', loss='meanSquaredError')
	
	# Return the model
	return model

def switch_training():
	train_a = not train_a
	

def main():
	# Create a model
	model_a = construct_model() # The trained model for player A
	model_b = construct_model() # The trained model for player B
	
	# Create a new instance of the pokemon game simulator
	game = new_game()

	# Play n games
	for current_game in range(num_games):
	
		# Store how many steps player A/B has made since they started being the one training
		current_switch_step = 0

		for current_step in range(max_steps):
			# Get the current game's state
			current_state = game.state()
			# Get the fitness of player A
			fitness_a = game.fitness(0)
			
			# Perform training on the actively training player
			result = None
			if(train_a):
				result = model_a.fit(current_state, fitness_a)
			else:
				result = model_a.predict(current_state, fitness_a)
				
			print(result)
			
			# Generate new state and get fitness for player B
			current_state = game.state()
			fitness_b = game.fitness(1)
			
			# Player b makes a move
			result = None
			if(train_a):
				result = model_b.predict(current_state, fitness_b)
			else:
				result = model_b.fit(current_state, fitness_b)
			
			print(result)
			
			# Increment or reset switch_step
			if current_switch_step == switch_steps:
				current_switch_step = 0
				print("Switching training from player", "A" if train_a else "B","to player","B" if train_a else "A")
				# Perform switch so that the opponent begins training
				switch_training()
			else:
				current_switch_step += 1
	
main()