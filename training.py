import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pokemon


# ReplayBuffer class stores information about past events and trajectories useful for future events
class ReplayBuffer:
    def __init__(self, max_size, input_dims):
        # How many past steps of training should be held in memory?
        self.mem_size = max_size
        # Current offset to next memory location to write to.
        # Memory counter loops back to start when the capacity is reached
        self.mem_counter = 0
        # List that stores previous observations
        self.state_memory = np.zeros((self.mem_size, input_dims),
                                     dtype=np.float32)
        # List that stores new observations
        self.new_state_memory = np.zeros((self.mem_size, input_dims),
                                         dtype=np.float32)
        # List that stores actions that were chosen
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        # List that stores rewards that were given
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # List that stores game completion flags that were given in a step
        self.doneflags_memory = np.zeros(self.mem_size, dtype=np.int32)

    # Stores the 'before' and 'after' states, reward, and completion flag after a simulation step is performed
    def store_step_transition(self, old_state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = old_state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.doneflags_memory[index] = 1 - int(done)
        self.mem_counter += 1

    # Returns a collection of lists, each of size batch_size, from memory
    # The lists contain past events that were stored in memory via storing step transitions
    def sample_memory(self, batch_size):
        # Get the size of memory that can be drawn from, either current size or capacity
        # Returns at most, the capacity of the memory
        mem_filled = min(self.mem_size, self.mem_counter)
        # Grab a random sampling of indices of size 'batch_size' with max of mem_filled
        batch = np.random.choice(mem_filled, batch_size, replace=False)

        # batch now contains a random list of indices that are used to pick n samples from memory
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        doneflag = self.doneflags_memory[batch]

        return states, actions, rewards, new_states, doneflag

# Generates a TensorFlow sequential model for training
# Model contains several dense layers with an output shape equal to the number of actions
# n_actions = size of action space, or number of actions agent can make
# input_size = number of parameters in observation/state space
def build_dqn(learn_rate, n_actions, input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_size, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_actions, activation='relu')])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
                  loss='mean_squared_error')

    return model

# A class to represent the DQN agent and its hyperparameters
class Agent:
    def __init__(self, learn_rate, gamma, n_actions, epsilon_start, batch_size, input_size, epsilon_decrement=0.01,
                 epsilon_end=0.01, mem_size=1000000, fname='model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.learn_rate = learn_rate
        self.gamma = gamma # Discount factor for learning
        self.epsilon = epsilon_start # Exploration vs Exploitation
        self.epsilon_start = epsilon_start # Starting chance of exploring
        self.epsilon_min = epsilon_end # Minimum chance of exploring at any point
        self.epsilon_decrement = epsilon_decrement # Linearly decrease of exploration over time
        self.batch_size = batch_size # Number of samples to use for learning
        self.model_file = fname # Filename to save the trained model at end of training
        self.input_size = input_size # Size of the observation space
        self.memory = ReplayBuffer(mem_size, input_size) # Stores information about past events
        self.q_model = build_dqn(learn_rate, n_actions, input_size) # Creates and compiles a TensorFlow sequential model

    # Stores the transition from old state and new state into memory
    # Memory holds what a state looked like before an action, after an action, and
    # the reward and whether the game is over (doneflag)
    def store_step_transition(self, old_state, action, reward, new_state, doneflag):
        self.memory.store_step_transition(old_state, action, reward, new_state, doneflag)

    # Choose an action, provided an observation of the current state space
    def choose_action(self, observation):
        # Is this action utilizing exploration or exploitation?
        if np.random.random() < self.epsilon:
            # Random roll is exploration. Perform a random action
            action = np.random.choice(self.action_space)
        else:
            # Random roll is exploitation. Utilize memory by predicting
            # Prediction accepts a collection of observations; only 1 observation is available
            # Simply put observation into a 1-element nested list
            state = np.array([observation])
            actions = self.q_model.predict(state)

            # Choose the action with the highest prediction
            action = np.argmax(actions)

        return action


    def learn(self):

        # Don't perform learning if our resulting batch size is too small
        if self.memory.mem_counter < self.batch_size:
            return

        # Sample state step transitions from past events (memory)
        states, actions, rewards, new_states, doneflags = self.memory.sample_memory(self.batch_size)

        # Generate a list of Q values for the sampled old states
        q_eval = self.q_model.predict(states)
        # Generate a list of Q values for the sampled resulting new states
        q_next = self.q_model.predict(new_states)
        q_target = np.copy(q_eval) # Python is weird. We want a deep copy
        batch_index = np.arange(self.batch_size, dtype=np.int32) # Generates list of indices from 0 to batch_size

        # Generate target Q values; target represents the "direction" we want prediction estimates to gravitate towards
        # The goal is to have Q targets converge to the best action
        # Each element contains a target based on the action that was taken for that simulation step
        # gamma represents how much of previous memory to consider (discount)
        # q_next term represents the best action (i.e. the action taken using argmax) for that new state
        # doneflag is 0 when at the terminal state, therefore only consider the reward term
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * doneflags

        # Perform training based off of the desired q targets
        # Our goal is to have Q values gravitate towards the ideal actions
        self.q_model.train_on_batch(states, q_target)

        # Update epsilon (exploration vs exploitation)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_decrement
        else:
            self.epsilon = self.epsilon_min

    # Save the TensorFlow model weights to a file using the agent's model_file member
    def save_model(self):
        self.q_model.save(self.model_file)

    # Load a model from the agent's model_file member
    def load_model(self):
        self.q_model = tf.keras.models.load_model(self.model_file)
        #self.q_model.load_weights(self.model_file)


# ---------- Constants ----------
num_games = 5  # Number of games to play
num_games = 100  # Number of full games to simulate
switch_games = 50  # Switch who trains after n games. This doesn't matter if training against an evaluator function

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # Discount for Q targets in learning
epsilon = 1.0  # Percent chance of exploring at start
epsilon_dec = 0.002  # Amount to decrement epsilon per learning call

space_size = 156  # Number of parameters in the observation space
action_size = 6  # Number of actions that can be performed by the player

# Simulation Operation Switches

# The Training simulation switches which opponent is training after a certain
# number of games. If the AI is training against an evaluator function, no
# switching will occur as the evaluator function does not train.

# --- Training ---
#    True  = The simulation performs training on Player A
#    False = The simulation performs training on Player B
# If the Opponent is another Agent, the value that train_a
# is set to is the Agent that will train first.
train_a = True

# Should the AI train against an evaluator function?
#    None     = AI Agent will train against another Agent. The Agent
#               that is actively training will switch defined by switch_games
#    Not None = AI Agent will train against an opponent that picks its move
#               defined by the given function. The AI will not switch
#               training to the opponent as it is an evaluation function
train_against_function = pokemon.evaluator_highest_damage_action

# --- Playing ---
play_ai = False  # Set to True if you should verse the AI instead of training
player_a = True  # Play as Player A (Otherwise, Player B)

save_model = True  # Overwrite saved model file at end of training?
save_plot = True  # Save a plot of the scores?
score_range = 5.0  # Maximum value to show when plotting scores (or outputting to console)

# Switch which Agent is actively training
def switch_training():
    global train_a
    train_a = not train_a

# Generates a DQN Agent from global parameters
def make_agent(fname):
    return Agent(gamma=gamma, epsilon_start=epsilon, learn_rate=learning_rate,
                 input_size=space_size, n_actions=action_size,
                 mem_size=10000, batch_size=64, epsilon_end=0.01, epsilon_decrement=epsilon_dec, fname=fname)

# Generates a matplotlib plot based on Game Scores for both Player A and Player B
# y_data is a nested list, where each element is a pair of scores for both players:
#     [[a_score, b_score], [a_score, b_score], ... <n games played>]
def plot(x_data, y_data, filename, title='', x_label = 'X Axis', y_label='Y Axis', legend=['Player A', 'Player B']):
    # If title is empty, generate one using axis labels
    title = x_label + ' vs ' + y_label if title is '' else title

    plt.figure()
    for col in range(2):
        plt.plot(x_data, [item[col] for item in y_data], label=legend[col])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(filename + '.png')

# Play is called if the user is choosing to play against the AI
def play():
    print('Playing against the AI as player', ('A' if player_a else 'B'))
    # Generate teams
    Team1 = pokemon.generate_team_1()
    Team2 = pokemon.generate_team_2()

    fname = 'model_a.h5' if not player_a else 'model_b.h5'
    agent = make_agent(fname)

    # Load the AI.
    print('Loading saved model \'%s\'...' % fname)
    try:
        agent.load_model()
    except:
        print('Model could not be loaded -- playing against random AI')

    # Play the game, let the simulation know that it's going to get input from the AI
    pokemon.battleSim(Team1, Team2, ai=agent, ai_is_a=not player_a)

# Train is called if the user is training one or more agents
def train():

    if train_against_function is not None:
        print('Training Player %c against evaluator function' % 'A' if train_a else 'B')

    # Create models for player A and B
    if train_against_function is None or train_a:
        # Create model A if not training against an evaluator OR we *are* and training Player A
        model_a = make_agent(fname='model_a.h5')

    if train_against_function is None or not train_a:
        model_b = make_agent(fname='model_b.h5')

    # Keep track of scores for informative output and plotting
    scores = []

    # Generate Pokemone teams for Team A and Team B
    # (Optional) set to the same team to compare equivalent teams!
    team_a = pokemon.generate_team_1()
    team_b = pokemon.generate_team_2()

    # Play n games
    for current_game in range(num_games):
        done = False # Is this game over?
        score_a = 0
        score_b = 0

        # Create a new instance of the pokemon game simulator (RESET)
        # TODO: This may be pretty slow
        team_a = pokemon.generate_team_1()
        team_b = pokemon.generate_team_2()

        # Observation is 'old observation' before actions are performed
        observation = pokemon.getState(team_a, team_b)

        while not done:

            # Choose actions for both teams
            if train_against_function is None or train_a:
                action_a = model_a.choose_action(observation) # Train Agent A
            elif train_against_function is not None and not train_a:
                action_a = train_against_function(team_a) # Use Evaluator Function for Player A

            if train_against_function is None or not train_a:
                action_b = model_b.choose_action(observation) # Train Agent B
            elif train_against_function is not None and train_a:
                action_b = train_against_function(team_b) # Use Evaluator Function for Player B

            # Perform a simulation step using the chosen actions
            new_observation, reward, done = pokemon.step(team_a, team_b, action_a + 1, action_b + 1)
            #print('%.2f %.2f %i %i' % (reward[0], reward[1], action_a, action_b))

            # Accumulate game total scores from reward
            score_a += reward[0]
            score_b += reward[1]

            # Store what we learned into Agents' memories
            if train_against_function is None or train_a:
                model_a.store_step_transition(observation, action_a, reward[0], new_observation, done)

            if train_against_function is None or not train_a:
                model_b.store_step_transition(observation, action_b, reward[1], new_observation, done)

            # Old observation now takes on new value -- used for next loop iteration
            observation = new_observation

            # Perform learning on the AI that's actively training
            # Q values will be generated and used to find Q target
            model_a.learn() if train_a else model_b.learn()

        # Increment or switch who is training
        # Switching should not happen when training against and evaluator function
        if current_game % switch_games == 0 and current_game is not 0 and train_against_function is None:
            print('Switching training from player %c to player %c' %
                  (('A' if train_a else 'B'), ('B' if train_a else 'A')))
            # Perform switch so that the opponent begins training
            switch_training()

        # Cap game score within a range to avoid messy plots
        score_a = max(-score_range, min(score_range, score_a))
        score_b = max(-score_range, min(score_range, score_b))

        scores.append([score_a, score_b])

        avg_score = np.mean(scores[-switch_games*2:], axis=0)
        _epsilon = model_a.epsilon if train_a else model_b.epsilon

        print('[Episode %i/%i]' % (current_game, num_games),
              '[Training %c]' % ('A' if train_a else 'B'),
              '[%c Win over %i turns]' % (('A' if team_a.hasAvailablePokemon else 'B'), team_a.roundNumber),
              '[Scores (%.2f, %.2f)]' % (score_a, score_b),
              '[average_score (%.2f, %.2f)]' % (avg_score[0], avg_score[1]),
              '[epsilon %.2f]' % _epsilon)

    if save_model:
        print('Saving models to file...')
        if train_against_function is None or train_a:
            model_a.save_model()

        if train_against_function is None or not train_a:
            model_b.save_model()

    if save_plot:
        print('Saving plot...')
        plot([x for x in range(num_games)], scores, filename='scores_plot', x_label='Game Number', y_label='Score')

if __name__ == '__main__':
    if not play_ai:
        train()
    else:
        play()
