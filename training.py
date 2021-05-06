import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pokemon
import random


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


def build_dqn(learn_rate, n_actions, input_dims):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(n_actions, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
                  loss='mean_squared_error')

    return model


class Agent():
    def __init__(self, learn_rate, gamma, n_actions, epsilon_start, batch_size, input_dims, epsilon_decrement=0.01,
                 epsilon_end=0.01, mem_size=1000000, fname='model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_end = epsilon_end
        self.epsilon_decrement = epsilon_decrement
        self.batch_size = batch_size
        self.model_file = fname
        self.input_dims = input_dims
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(learn_rate, n_actions, input_dims)

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
        #raise Exception("Not yet implemented")
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = tf.keras.models.load_model(self.model_file)
        #self.q_eval.load_weights(self.model_file)


# ---------- Constants ----------
learning_rate = 0.00005
gamma = 0.99
epsilon = 1.0
epsilon_dec = 0.0001
num_games = 5
num_games=500  # Number of full games to simulate
switch_games = 50  # Switch who trains after n games. This doesn't matter if training against an evaluator function

space_size = 156  # Number of parameters in the observation space
action_size = 6  # Number of actions that can be performed by the player

# ---------- Global variables ----------
train_a = True # Set to True if player A is the one that's currently training
play_ai = False # Set to True if you should verse the AI instead of training
player_a = True # Play as Player A (Otherwise, Player B)
save_model = True # Overwrite saved model at end of training?
save_plot = True # Save a plot of the scores?
# Should the AI train against an evaluator function?
# None = AI will train against A/B opponent
# If evaluator function is not none, training will not switch between opponents
train_against_function = pokemon.evaluator_highest_damage_action

def switch_training():
    global train_a
    train_a = not train_a

# Generates a DQN Agent with global parameters
def make_agent(fname):
    return Agent(gamma=gamma, epsilon_start=epsilon, learn_rate=learning_rate,
                 input_dims=space_size, n_actions=action_size,
                 mem_size=10000, batch_size=64, epsilon_end=0.01, epsilon_decrement=epsilon_dec, fname=fname)

def plot(x_data, y_data, filename, title='', x_label = 'X Axis', y_label='Y Axis', legend=['Player A', 'Player B']):
    # If title is empty, generate one using axis labels
    title = x_label + ' vs ' + y_label if title is '' else title

    plt.figure()
    print(y_data)
    for col in range(2):
        plt.plot(x_data, [item[col] for item in y_data], label=legend[col])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(filename + '.png')

def play():
    print('Playing against the AI as player', ('A' if player_a else 'B'))
    # Play the game, let the simulation know that it's going to get input from the AI
    # Generate teams
    Team1 = pokemon.generate_team_1()
    Team2 = pokemon.generate_team_2()

    fname = 'model_a.h5' if not player_a else 'model_b.h5'
    agent = make_agent(fname)

    # Load the AI
    print('Loading saved model \'%s\'...' % fname)
    agent.load_model()

    # battle the teams
    pokemon.battleSim(Team1, Team2, ai=agent, ai_is_a=not player_a)

def train(plot_name='plot'):

    if train_against_function is not None:
        print('Training Player %c against evaluator function' % 'A' if train_a else 'B')

    # Create models for player A and B
    if train_against_function is None or train_a:
        # Create model A if not training against an evaluator OR we *are* and training Player A
        model_a = make_agent(fname='model_a.h5')

    if train_against_function is None or not train_a:
        model_b = make_agent(fname='model_b.h5')

    scores = []

    team_a = pokemon.generate_team_1()
    team_b = pokemon.generate_team_1()

    # Play n games
    for current_game in range(num_games):
        done = False
        score_a = 0
        score_b = 0

        # Create a new instance of the pokemon game simulator
        # RESET
        # TODO: This may be pretty slow
        team_a = pokemon.generate_team_1()
        team_b = pokemon.generate_team_1()

        observation = pokemon.getState(team_a, team_b)

        while not done:

            # Choose actions for both teams
            if train_against_function is None or train_a:
                action_a = model_a.choose_action(observation)
            elif train_against_function is not None and not train_a:
                action_a = train_against_function(team_a)

            if train_against_function is None or not train_a:
                action_b = model_b.choose_action(observation)
            elif train_against_function is not None and train_a:
                action_b = train_against_function(team_b)

            new_observation, reward, done = pokemon.step(team_a, team_b, action_a + 1, action_b + 1)
            # print(new_observation)
            score_a += reward[0]
            score_b += reward[1]

            if train_against_function is None or train_a:
                model_a.store_transition(observation, action_a, reward[0], new_observation, done)

            if train_against_function is None or not train_a:
                model_b.store_transition(observation, action_b, reward[1], new_observation, done)

            observation = new_observation

            model_a.learn() if train_a else model_b.learn()

        # Increment or switch who is training
        # Switching should not happen when training against and evaluator function
        if current_game % switch_games == 0 and current_game is not 0 and train_against_function is None:
            print('Switching training from player %c to player %c' %
                  (('A' if train_a else 'B'), ('B' if train_a else 'A')))
            # Perform switch so that the opponent begins training
            switch_training()

        score_a = max(-1.0, min(1.0, score_a))
        score_b = max(-1.0, min(1.0, score_b))

        scores.append([score_a, score_b])

        # Negate train_a returns 0 for a, 1 for b (indices)
        avg_score = np.mean(scores[-switch_games*2:], axis=0)[int(not train_a)]
        _score = score_a if train_a else score_b
        _epsilon = model_a.epsilon if train_a else model_b.epsilon

        print('[Episode %i/%i]' % (current_game, num_games),
              '[Training %c]' % ('A' if train_a else 'B'),
              '[%c Win over %i turns]' % (('A' if team_a.hasAvailablePokemon else 'B'), team_a.roundNumber),
              '[Score %.2f]' % _score,
              '[average_score %.2f]' % avg_score,
              '[epsilon %.2f]' % _epsilon)
        # Uncomment to display actions chosen for the game
        #print(model_a.memory.mem_counter, model_a.memory.action_memory[model_a.memory.mem_counter-team_a.roundNumber:model_a.memory.mem_counter - 1])

    if save_model:
        print('Saving models to file...')
        if train_against_function is None or train_a:
            model_a.save_model()

        if train_against_function is None or not train_a:
            model_b.save_model()

    if save_plot:
        print('Saving plot...')
        plot([x for x in range(num_games)], scores, filename=plot_name, x_label='Game Number', y_label='Score')

if __name__ == '__main__':
    if not play_ai:
        for x in range(10):
            train(str(x))
    else:
        play()
