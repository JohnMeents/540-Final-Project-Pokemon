import tensorflow as tf
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
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_actions, activation=None)])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
                  loss='mean_squared_error')

    return model


class Agent():
    def __init__(self, learn_rate, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_decrement=0.001,
                 epsilon_end=0.01, mem_size=1000000, fname='model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.epsilon = epsilon
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
learning_rate = 0.001
num_games = 200  # Number of full games to simulate
switch_games = 25  # Switch who trains after n games

space_size = 154  # Number of parameters in the observation space
action_size = 6  # Number of actions that can be performed by the player

# ---------- Global variables ----------
train_a = False  # Set to True if player A is the one that's currently training
play_ai = True # Set to True if you should verse the AI instead of training
player_a = True # Play as Player A (Otherwise, Player B)

def switch_training():
    global train_a
    train_a = not train_a

def play():
    print('Playing against the AI as player', ('A' if player_a else 'B'))
    # Play the game, let the simulation know that it's going to get input from the AI
    # Generate teams
    Team1 = pokemon.generate_team_1()
    Team2 = pokemon.generate_team_2()

    # Load the AI
    fname = 'model_a.h5' if player_a else 'model_b.h5'
    print('Loading saved model...')
    agent = Agent(gamma=0.99, epsilon=0.1, learn_rate=learning_rate,
                    input_dims=space_size, n_actions=action_size,
                    mem_size=1000000, batch_size=64, epsilon_end=0.01, fname='model_a.h5')
    agent.load_model()

    # battle the teams
    pokemon.battleSim(Team1, Team2, ai=agent, ai_is_a=player_a)

def train():
    # Create a model
    model_a = Agent(gamma=0.99, epsilon=0.1, learn_rate=learning_rate,
                    input_dims=space_size, n_actions=action_size,
                    mem_size=1000000, batch_size=64, epsilon_end=0.01, fname='model_a.h5')
    model_b = Agent(gamma=0.99, epsilon=0.1, learn_rate=learning_rate,
                    input_dims=space_size, n_actions=action_size,
                    mem_size=1000000, batch_size=64, epsilon_end=0.01, fname='model_b.h5')

    scores = []
    epsilon_history = []

    team_a = pokemon.generate_team_1()
    team_b = pokemon.generate_team_2()

    # Play n games
    for current_game in range(num_games):
        done = False
        score_a = 0
        score_b = 0

        # Create a new instance of the pokemon game simulator
        # RESET
        # TODO: This may be pretty slow
        team_a = pokemon.generate_team_1()
        team_b = pokemon.generate_team_2()

        observation = pokemon.getState(team_a, team_b)

        while not done:
            # Choose actions for both teams
            action_a = model_a.choose_action(observation)
            action_b = model_b.choose_action(observation)

            new_observation, reward, done = pokemon.step(team_a, team_b, action_a, action_b)
            # print(new_observation)
            score_a += reward[0]
            score_b += reward[1]

            model_a.store_transition(observation, action_a, reward[0], new_observation, done)
            model_b.store_transition(observation, action_b, reward[1], new_observation, done)

            observation = new_observation
            model_a.learn() if train_a else model_b.learn()

        # Increment or reset switch_step
        if current_game % switch_games == 0 and current_game is not 0:
            print("Switching training from player", "A" if train_a else "B", "to player", "B" if train_a else "A")
            # Perform switch so that the opponent begins training
            switch_training()

        epsilon_history.append(model_a.epsilon if train_a else model_b.epsilon)
        scores.append([score_a, score_b])

        # Negate train_a returns 0 for a, 1 for b (indices!)
        avg_score = np.mean(scores[-20:], axis=0)[int(not train_a)]
        _score = score_a if train_a else score_b
        _epsilon = model_a.epsilon if train_a else model_b.epsilon

        print('[Episode %i/%i]' % (current_game, num_games),
              '[Training %c]' % ('A' if train_a else 'B'),
              '[%c Win]' % ('A' if team_a.hasAvailablePokemon else 'B'),
              '[Score %.2f]' % _score,
              '[average_score %.2f]' % avg_score,
              '[epsilon %.2f]' % _epsilon)

    model_a.save_model()
    model_b.save_model()


if __name__ == '__main__':
    if not play_ai:
        train()
    else:
        play()