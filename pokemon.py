# John Meents & Daniel Moore
# TODO
# fitness funcion
# function to get the state of the game that returns every relevant variable

import time
import random
import numpy as np
import sys
import math

# -------------------------------------Global Data--------------------------------------
pokemon_types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                 "Fighting", "Poison", "Ground", "Flying", "Psychic",
                 "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

# Used to map string values of pokemon_types to an integer for machine learning
pokemon_types_dict = {'Normal': 1, 'Fire': 2, 'Water': 3, 'Electric': 4, 'Grass': 5, 'Ice': 6, 'Fighting': 7,
                      'Poison': 8, 'Ground': 9, 'Flying': 10, 'Psychic': 11, 'Bug': 12, 'Rock': 13, 'Ghost': 14,
                      'Dragon': 15, 'Dark': 16, 'Steel': 17, 'Fairy': 18}

pokemon_conditions_dict = {
    "None": 0, "Paralyzed": 1, "Asleep": 2, "Burned": 3, "Poisoned": 4, "Confused": 6, "Frozen": 7,
}

# A 2 Dimensional Numpy Array Of Damage Multipliers For Attacking Pokemon:
# Origin of this table is at the top left and the sequence follows the list above
# the row represents the attacking pokemon, the column represents the defending pokemon
# the value represents the type damage modifier
# 0: immune
# 1: normal effectiveness
# 2: super effective
# 1/2: not very effective
# for reference, see damage type chart on discord

damage_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 / 2, 0, 1, 1, 1 / 2, 1],
                         [1, 1 / 2, 1 / 2, 1, 2, 2, 1, 1, 1, 1,
                          1, 2, 1 / 2, 1, 1 / 2, 1, 2, 1],
                         [1, 2, 1 / 2, 1, 1 / 2, 1, 1, 1, 2,
                          1, 1, 1, 2, 1, 1 / 2, 1, 1, 1],
                         [1, 1, 2, 1 / 2, 1 / 2, 1, 1, 1, 0,
                          2, 1, 1, 1, 1, 1 / 2, 1, 1, 1],
                         [1, 1 / 2, 2, 1, 1 / 2, 1, 1, 1 / 2, 2, 1 /
                          2, 1, 1 / 2, 2, 1, 1 / 2, 1, 1 / 2, 1],
                         [1, 1 / 2, 1 / 2, 1, 2, 1 / 2, 1, 1, 2,
                          2, 1, 1, 1, 1, 2, 1, 1 / 2, 1],
                         [2, 1, 1, 1, 1, 2, 1, 1 / 2, 1, 1 / 2,
                          1 / 2, 1 / 2, 2, 0, 1, 2, 2, 1 / 2],
                         [1, 1, 1, 1, 2, 1, 1, 1 / 2, 1 / 2, 1,
                          1, 1, 1 / 2, 1 / 2, 1, 1, 0, 2],
                         [1, 2, 1, 2, 1 / 2, 1, 1, 2, 1, 0,
                          1, 1 / 2, 2, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1 / 2, 2, 1, 2, 1, 1, 1,
                          1, 2, 1 / 2, 1, 1, 1, 1 / 2, 1],
                         [1, 1, 1, 1, 1, 1, 2, 2, 1, 1,
                          1 / 2, 1, 1, 1, 1, 0, 1 / 2, 1],
                         [1, 1 / 2, 1, 1, 2, 1, 1 / 2, 1 / 2, 1, 1 /
                          2, 2, 1, 1, 1 / 2, 1, 2, 1 / 2, 1 / 2],
                         [1, 2, 1, 1, 1, 2, 1 / 2, 1, 1 / 2,
                          2, 1, 2, 1, 1, 1, 1, 1 / 2, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          2, 1, 1, 2, 1, 1 / 2, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 2, 1, 1 / 2, 0],
                         [1, 1, 1, 1, 1, 1, 1 / 2, 1, 1, 1,
                          2, 1, 1, 2, 1, 1 / 2, 1, 1 / 2],
                         [1, 1 / 2, 1 / 2, 1 / 2, 1, 2, 1, 1, 1,
                          1, 1, 1, 2, 1, 1, 1, 1 / 2, 2],
                         [1, 1 / 2, 1, 1, 1, 1, 2, 1 / 2, 1, 1, 1, 1, 1, 1, 2, 2, 1 / 2, 1]])


# The highest round that a game will last
max_round = 50

# ------------- Parameters for Rewards -------------

# Reward for if the AI won the round
reward_round_win = 1.0
# Discount for if the AI lost the round
reward_round_loss = 1.0
# Reward multiplier based on what percentage of opponent's hp was taken
reward_damage_multiplier = 0.8
# Discount multiplier based on how many rounds have been played.
reward_round_n_multiplier = 0.01
# Metric that aims for the AI to target this maximum number of rounds
round_n_goal = 20
# Reward/Discount for if a pokemon faints.
reward_pokemon_faint = 0.4
# Discount for if the chosen action is to switch Pokemon.
# The intention is to minimize switching repeatedly
reward_punish_switch = 0.05



# -------------------------------------------------------------------------------------


# Delay Printing like a Gameboy
def delayPrint(s):
    for c in s:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(0.01)  # How fast to print


# Team Class
class Team:
    def __init__(self, teamName, Pokemon1, Pokemon2, Pokemon3):
        self.teamName = teamName
        self.Pokemon1 = Pokemon1
        self.Pokemon2 = Pokemon2
        self.Pokemon3 = Pokemon3
        self.activePokemon = Pokemon1
        self.activePokemonN = 1
        self.hasAvailablePokemon = True
        self.reward = 0
        self.roundNumber = 0
        # Set to true within fightSim if the previous Pokemon fainted prior to an automatic switch
        self.faintedFlag = False

    # Returns an array that represents parameters about this object
    def toArray(self):
        return (self.Pokemon1.toArray() + self.Pokemon2.toArray() +
                self.Pokemon3.toArray() +
                [self.activePokemonN,
                 int(self.hasAvailablePokemon), self.roundNumber])


# Pokemon Class
class Pokemon:
    def __init__(self, name, type1, type2, healthPercentage, moves, status, hp, attack, spattack, defense, spdefense,
                 speed, ability, item, ):
        self.name = name
        self.type1 = type1  # changed this variable name from types to type1, updated in the toArray
        self.type2 = type2
        self.healthPercentage = healthPercentage
        self.moves = moves
        self.status = status
        self.hp = hp
        self.attack = attack
        self.spattack = spattack
        self.defense = defense
        self.spdefense = spdefense
        self.speed = speed
        self.maxHp = hp
        self.ability = ability  # implement last
        self.item = item  # implement last
        self.level = 100.0  # pokemon are automatically set to level 100

    # Returns an array that represents parameters about this object
    def toArray(self):
        # Generate integer values for these strings

        return [
                   self.type1,
                   self.type2 if self.type2 is not None else 0,
                   self.healthPercentage,
                   self.attack,
                   self.spattack,
                   self.defense,
                   self.spdefense,
                   self.speed,
                   self.maxHp,
               ] + [item for items in self.moves for item in items.toArray()]


# Moves Class
class Move:
    def __init__(self, moveName, moveType, physpc, basePower, accuracy,
                 effect):
        self.moveName = moveName
        self.moveType = moveType
        self.physpc = physpc
        self.basePower = basePower
        self.accuracy = accuracy  # moves with 101 accuracy cannot be lowered
        self.effect = effect  # implement last

    # Returns an array that represents parameters about this object
    def toArray(self):
        physpc_int = (0 if self.physpc is None else 1 if self.physpc
                                                         == "special" else 2 if self.physpc == "status" else -1)
        basePower_int = self.basePower if self.basePower is not None else -1
        return [
            self.moveType,
            physpc_int,
            basePower_int,
            self.accuracy,
        ]


# a function that takes an integer as an action, an int/bool as team and pokemon info
def fightSim(Team1, Team2, team1Action, team2Action):
    Team1.reward = 0
    Team2.reward = 0
    Team1.faintedFlag = False
    Team2.faintedFlag = False

    # if they switch pokemon, that action happens first before the opponent moves
    # switching Team1 pokemon
    if team1Action == 5 or team1Action == 6:
        Team1.reward -= reward_punish_switch
        if (team1Action == 5 and Team1.Pokemon1.hp > 0 and Team1.activePokemon != Team1.Pokemon1):
            Team1.activePokemon = Team1.Pokemon1
            Team1.activePokemonN = 1
        elif (team1Action == 5 and Team1.Pokemon2.hp > 0 and Team1.activePokemon != Team1.Pokemon2):
            Team1.activePokemon = Team1.Pokemon2
            Team1.activePokemonN = 2
        elif (team1Action == 5 and Team1.Pokemon3.hp > 0 and Team1.activePokemon != Team1.Pokemon3):
            Team1.activePokemon = Team1.Pokemon3
            Team1.activePokemonN = 3

        elif (team1Action == 6 and Team1.Pokemon3.hp > 0 and Team1.activePokemon != Team1.Pokemon3):
            Team1.activePokemon = Team1.Pokemon3
            Team1.activePokemonN = 3
        elif (team1Action == 6 and Team1.Pokemon2.hp > 0 and Team1.activePokemon != Team1.Pokemon2):
            Team1.activePokemon = Team1.Pokemon2
            Team1.activePokemonN = 2
        elif (team1Action == 6 and Team1.Pokemon1.hp > 0 and Team1.activePokemon != Team1.Pokemon1):
            Team1.activePokemon = Team1.Pokemon1
            Team1.activePokemonN = 1
        else:
            # Attempt to select a Pokemon that has fainted
            pass

    # switching Team2 pokemon
    if team2Action == 5 or team2Action == 6:
        Team2.reward -= reward_punish_switch
        if (team2Action == 5 and Team2.Pokemon1.hp > 0 and Team2.activePokemon != Team2.Pokemon1):
            Team2.activePokemon = Team2.Pokemon1
            Team2.activePokemonN = 1
        elif (team2Action == 5 and Team2.Pokemon2.hp > 0 and Team2.activePokemon != Team2.Pokemon2):
            Team2.activePokemon = Team2.Pokemon2
            Team2.activePokemonN = 2
        elif (team2Action == 5 and Team2.Pokemon3.hp > 0 and Team2.activePokemon != Team2.Pokemon3):
            Team2.activePokemon = Team2.Pokemon3
            Team2.activePokemonN = 3

        elif (team2Action == 6 and Team2.Pokemon3.hp > 0 and Team2.activePokemon != Team2.Pokemon3):
            Team2.activePokemon = Team2.Pokemon3
            Team2.activePokemonN = 3
        elif (team2Action == 6 and Team2.Pokemon2.hp > 0 and Team2.activePokemon != Team2.Pokemon2):
            Team2.activePokemon = Team2.Pokemon2
            Team2.activePokemonN = 2
        elif (team2Action == 6 and Team2.Pokemon1.hp > 0 and Team2.activePokemon != Team2.Pokemon1):
            Team2.activePokemon = Team2.Pokemon1
            Team2.activePokemonN = 1
        else:
            # Attempt to select a Pokemon that has fainted
            pass

    # if Team1 moves first
    if Team1.activePokemon.speed >= Team2.activePokemon.speed:
        # do calc
        damageCalc(Team1, Team2, team1Action)
        # other pokemon attacks if they didn't just faint
        if Team1.activePokemon.hp > 0 and Team2.activePokemon.hp > 0:
            damageCalc(Team2, Team1, team2Action)
    # if Team2 moves first
    else:
        # do calc
        damageCalc(Team2, Team1, team2Action)
        # other pokemon attacks if they didn't just faint
        if Team1.activePokemon.hp > 0 and Team2.activePokemon.hp > 0:
            damageCalc(Team1, Team2, team1Action)

    # Automatically pick an available Pokemon if one of the teams' active pokemon fainted
    if (Team1.activePokemon.hp <= 0 and Team2.activePokemon.hp > 0 and Team1.hasAvailablePokemon):
        Team1.faintedFlag = True
        # A Pokemon has fainted, punish the team
        Team1.reward -= reward_pokemon_faint
        Team2.reward += reward_pokemon_faint
        # Automatically pick a Pokemon for Team 1
        if Team1.Pokemon1.hp > 0:
            Team1.activePokemon = Team1.Pokemon1
            Team1.activePokemonN = 1
        elif Team1.Pokemon2.hp > 0:
            Team1.activePokemon = Team1.Pokemon2
            Team1.activePokemonN = 2
        elif Team1.Pokemon3.hp > 0:
            Team1.activePokemon = Team1.Pokemon3
            Team1.activePokemonN = 3
        else:
            # All pokemon are fainted, this shouldn't happen from the above check
            pass
    elif (Team1.activePokemon.hp > 0 and Team2.activePokemon.hp <= 0 and Team2.hasAvailablePokemon):
        Team2.faintedFlag = True
        # A Pokemon has fainted, punish the team
        Team1.reward += reward_pokemon_faint
        Team2.reward -= reward_pokemon_faint
        # Automatically pick a Pokemon for Team 2
        if Team2.Pokemon1.hp > 0:
            Team2.activePokemon = Team2.Pokemon1
            Team2.activePokemonN = 1
        elif Team2.Pokemon2.hp > 0:
            Team2.activePokemon = Team2.Pokemon2
            Team2.activePokemonN = 2
        elif Team2.Pokemon3.hp > 0:
            Team2.activePokemon = Team2.Pokemon3
            Team2.activePokemonN = 3
        else:
            # All pokemon are fainted, this shouldn't happen from the above check
            pass
    else:
        # I don't think both pokemon will faint on the same turn in this limited simulation,
        # therefore assume they're both alive
        pass

    # Check if max round has been hit
    if Team1.roundNumber >= max_round:
        Team1.hasAvailablePokemon = False
        Team2.hasAvailablePokemon = False
        # Determine who has the most health by sum of percentages
        team1_sum = Team1.Pokemon1.healthPercentage + Team1.Pokemon2.healthPercentage + Team1.Pokemon3.healthPercentage
        team2_sum = Team2.Pokemon1.healthPercentage + Team2.Pokemon2.healthPercentage + Team2.Pokemon3.healthPercentage

        Team1.reward += reward_round_win if team1_sum > team2_sum else -reward_round_loss if team1_sum < team2_sum else 0
        Team2.reward += reward_round_win if team2_sum > team1_sum else -reward_round_loss if team2_sum < team1_sum else 0

    # check if each team has available pokemon
    if Team1.Pokemon1.hp <= 0 and Team1.Pokemon2.hp <= 0 and Team1.Pokemon3.hp <= 0:
        Team1.hasAvailablePokemon = False
        # Team1 lost, punish the AI
        Team1.reward -= reward_round_loss
        # By far the best heuristic metric is if an AI won the game
        Team2.reward += reward_round_win
    if Team2.Pokemon1.hp <= 0 and Team2.Pokemon2.hp <= 0 and Team2.Pokemon3.hp <= 0:
        # Team2 lost, punish the AI
        Team2.reward -= reward_round_loss
        # By far the best heuristic metric is if an AI won the game
        Team1.reward += reward_round_win
        Team2.hasAvailablePokemon = False

    # Discourage the AI from playing long games by linearly increasing a punishment
    roundFactor = float(Team1.roundNumber) / round_n_goal # Denominator represents number of desired rounds
    roundDiscount = max(1.0, roundFactor)
    Team1.reward -= roundDiscount * reward_round_n_multiplier
    Team2.reward -= roundDiscount * reward_round_n_multiplier

    Team1.roundNumber += 1
    Team2.roundNumber += 1


# Returns an array of parameters
def getState(team1: Team, team2: Team):
    return team1.toArray() + team2.toArray()





def battleSim(Team1, Team2, ai=None, ai_is_a=True):
    txt = "default"
    chooseTeam1Move = ""
    chooseTeam2Move = ""
    # players start out with a predetermined pokemon1
    # If both teams have usable pokemon: loop:
    while Team1.hasAvailablePokemon and Team2.hasAvailablePokemon:
        # if both active pokemon have health: loop:
        while Team1.activePokemon.hp >= 0 and Team2.activePokemon.hp >= 0:
            # print out fight info
            delayPrint("Team 1's active Pokemon: ")
            delayPrint(Team1.activePokemon.name)
            delayPrint("\n")
            delayPrint("Health percentage: ")
            print(Team1.activePokemon.healthPercentage)
            delayPrint("Team 2's active Pokemon: ")
            delayPrint(Team2.activePokemon.name)
            delayPrint("\n")
            delayPrint("Health percentage: ")
            print(Team2.activePokemon.healthPercentage)

            if ai is not None and ai_is_a:
                # Playing against AI, and AI is player 1
                chooseTeam1Move = ai.choose_action(getState(Team1, Team2)) + 1
                delayPrint('[AI chose %i]\n' % chooseTeam1Move)
            else:
                validInput1 = False
                # moves are chosen for Team 1
                while validInput1 == False:
                    delayPrint(
                        "Team 1, Enter an integer for the following command\n" +
                        "1: " + Team1.activePokemon.moves[0].moveName +
                        "   2: " + Team1.activePokemon.moves[1].moveName +
                        "   3: " + Team1.activePokemon.moves[2].moveName +
                        "   4: " + Team1.activePokemon.moves[3].moveName +
                        "\n5: Switch to first available Pokemon   6: Switch to second available Pokemon\n"
                    )
                    chooseTeam1Move = input()
                    # If they chose 1-4
                    if (chooseTeam1Move == "1" or chooseTeam1Move == "2"
                            or chooseTeam1Move == "3" or chooseTeam1Move == "4"):
                        validInput1 = True
                    # make sure "5" is a valid input (first benched pokemon)"
                    elif chooseTeam1Move == "5" and Team1.Pokemon1.hp > 0:
                        validInput1 = True
                    elif chooseTeam1Move == "5" and Team1.Pokemon2.hp > 0:
                        validInput1 = True
                    elif chooseTeam1Move == "5" and Team1.Pokemon3.hp > 0:
                        validInput1 = True
                    # make sure "6" is a valid input (second benched pokemon)
                    elif (chooseTeam1Move == "6" and Team1.Pokemon1.hp > 0
                          and Team1.Pokemon2.hp > 0 and Team1.Pokemon2.hp > 0):
                        validInput1 = True
                    elif chooseTeam1Move == "6" and Team1.Pokemon2.hp > 0:
                        validInput1 = True
                    elif chooseTeam1Move == "6" and Team1.Pokemon3.hp > 0:
                        validInput1 = True
                    else:
                        validInput1 = False  # remains false

            if ai is not None and not ai_is_a:
                # Playing against AI and AI is Player 2
                chooseTeam2Move = ai.choose_action(getState(Team1, Team2)) + 1
                delayPrint('[AI chose %i]\n' % chooseTeam2Move)
            else:
                validInput2 = False
                # moves are chosen for Team 2
                while validInput2 == False:
                    delayPrint(
                        "Team 2, Enter an integer for the following command\n" +
                        "1: " + Team2.activePokemon.moves[0].moveName +
                        "   2: " + Team2.activePokemon.moves[1].moveName +
                        "   3: " + Team2.activePokemon.moves[2].moveName +
                        "   4: " + Team2.activePokemon.moves[3].moveName +
                        "\n5: Switch to first available Pokemon   6: Switch to second available Pokemon\n"
                    )
                    chooseTeam2Move = input()
                    # If they choose 1-4
                    if (chooseTeam2Move == "1" or chooseTeam2Move == "2"
                            or chooseTeam2Move == "3" or chooseTeam2Move == "4"):
                        validInput2 = True
                    # make sure "5" is a valid input (first benched pokemon)"
                    elif chooseTeam2Move == "5" and Team2.Pokemon1.hp > 0:
                        validInput2 = True
                    elif chooseTeam2Move == "5" and Team2.Pokemon2.hp > 0:
                        validInput2 = True
                    elif chooseTeam2Move == "5" and Team2.Pokemon3.hp > 0:
                        validInput2 = True
                    # make sure "6" is a valid input (second benched pokemon)
                    elif (chooseTeam2Move == "6" and Team2.Pokemon1.hp > 0
                          and Team2.Pokemon2.hp > 0 and Team2.Pokemon2.hp > 0):
                        validInput2 = True
                    elif chooseTeam2Move == "6" and Team2.Pokemon2.hp > 0:
                        validInput2 = True
                    elif chooseTeam2Move == "6" and Team2.Pokemon3.hp > 0:
                        validInput2 = True
                    else:
                        validInput2 = False  # remains false

            # convert move choices from strings to ints
            chooseTeam1Move = int(chooseTeam1Move)
            chooseTeam2Move = int(chooseTeam2Move)

            # turn happens
            fightSim(Team1, Team2, chooseTeam1Move, chooseTeam2Move)

            if chooseTeam1Move > 4:
                delayPrint("Team 1 switched to %s!\n" % Team1.activePokemon.name)
            if chooseTeam2Move > 4:
                delayPrint("Team 2 switched to %s!\n" % Team2.activePokemon.name)

            # print out action info
            if (chooseTeam1Move < 5 and Team1.activePokemon.hp > 0 and not Team1.faintedFlag):
                delayPrint(Team1.activePokemon.name)
                delayPrint(" used ")
                delayPrint(Team1.activePokemon.moves[(
                        chooseTeam1Move - 1)].moveName)
                delayPrint("!\n")
                if (Team2.activePokemon.hp > 0):
                    delayPrint(Team2.activePokemon.name + " has ")
                    print(Team2.activePokemon.healthPercentage, end='')
                    delayPrint(" percent health\n")
            if (chooseTeam2Move < 5 and Team2.activePokemon.hp > 0 and not Team2.faintedFlag):
                delayPrint(Team2.activePokemon.name)
                delayPrint(" used ")
                delayPrint(Team2.activePokemon.moves[(
                        chooseTeam2Move - 1)].moveName)
                delayPrint("!\n")
                if (Team1.activePokemon.hp > 0):
                    delayPrint(Team1.activePokemon.name + " has ")
                    print(Team1.activePokemon.healthPercentage, end='')
                    delayPrint(" percent health\n")

            # if a pokemon faints:
            if Team1.faintedFlag:
                delayPrint("Team 1's pokemon fainted! Automatically switched to %s\n" % Team1.activePokemon.name)
            if Team2.faintedFlag:
                delayPrint("Team 2's pokemon fainted! Automatically switched to %s\n" % Team2.activePokemon.name)

    # print winning team
    if Team1.hasAvailablePokemon and (Team2.hasAvailablePokemon == False):
        delayPrint("Team 1 wins!\n")
    else:
        delayPrint("Team 2 wins!\n")


def isGameOver(team1: Team, team2: Team):
    return not team1.hasAvailablePokemon or not team2.hasAvailablePokemon


# Perform a step in the game simulation. This is primarily used by the AI
def step(team1: Team, team2: Team, team1_action, team2_action):
    # Perform the turn/round
    fightSim(team1, team2, team1_action, team2_action)

    # Determine observation, or new state space
    observation = getState(team1, team2)

    # Clamp rewards between [-1.0, 1.0]
    team1_reward = max(-1.0, min(1.0, team1.reward))
    team2_reward = max(-1.0, min(1.0, team2.reward))

    # Determine if game (episode) is over
    gameOver = isGameOver(team1, team2)
    # Return observation, rewards, gameOver
    return observation, [team1_reward, team2_reward], gameOver


def damageCalc(TeamAttacker, TeamDefender, move):  # damage calculation function
    Pokemon1 = TeamAttacker.activePokemon
    Pokemon2 = TeamDefender.activePokemon

    # Pokemon1 is the attacking pokemon, Pokemon2 is the defending pokemon
    # do nothing if the move is 5 or 6 (a switch)
    if move == 5 or move == 6:
        return

    # accuracy check
    accuracyGenerator = random.randint(1, 100)
    if (Pokemon1.moves[(move - 1)].accuracy < accuracyGenerator):
        return

    # if Pokemon1 uses a damage dealing move
    if (Pokemon1.moves[(move - 1)].physpc == "physical"
            or Pokemon1.moves[(move - 1)].physpc == "special"):
        # get the base power of the move
        bp = Pokemon1.moves[(move - 1)].basePower

        # determine if it's a critical hit
        criticalGenerator = random.randint(0, 15)
        criticalMultiplier = 1.0
        isCriticalHit = False  # For printing later
        if criticalGenerator == 15:
            criticalMultiplier = 1.5
            isCriticalHit = True

        # determine the roll (high or low damage roll between .85 and 1.0)
        roll = random.uniform(0.85, 1.0)

        # determine if the move gets STAB (Same Type Attack Bonus) (happens if the move's type matches one of the Pokemon's type)
        stab = 1.0
        if (Pokemon1.moves[(move - 1)].moveType == Pokemon1.type1
                or Pokemon1.moves[(move - 1)].moveType == Pokemon1.type2):
            stab = 1.5

        # determine the type effectiveness
        if Pokemon2.type2 == None:  # If the defending pokemon only has one type
            typeEffectiveness = damage_array[Pokemon1.moves[(
                    move - 1)].moveType][Pokemon2.type1]
        else:  # if the defending pokemon is dual type
            typeEffectiveness = (damage_array[Pokemon1.moves[(
                    move - 1)].moveType][Pokemon2.type1] * damage_array[Pokemon1.moves[(move - 1)].moveType][
                                     Pokemon2.type2])

        # Determine if the Attacker gets a damage reduction for being burned and using a physical move
        burn = 1.0
        if (Pokemon1.status == pokemon_conditions_dict["Burned"]
                and Pokemon1.moves[(move - 1)].physpc == "physical"):
            burn = 0.5

        # Possibly implement later: damage boosts from items or abilities
        other = 1.0

        # combine all previous variables into a single variable
        modifier = criticalMultiplier * roll * stab * typeEffectiveness * burn * other

        # determine final damage number
        # if the move is physical
        if Pokemon1.moves[(move - 1)].physpc == "physical":
            damage = (((
                               ((((2.0 * Pokemon1.level) / 5.0) + 2.0) * bp *
                                (Pokemon1.attack / Pokemon2.defense)) / 50.0) + 2.0) * modifier)
        # if the move is special
        else:
            damage = ((((
                                (((2.0 * Pokemon1.level) / 5.0) + 2.0) * bp *
                                (Pokemon1.spattack / Pokemon2.spdefense)) / 50.0) + 2.0) * modifier)

        # apply damage
        Pokemon2.hp -= damage
        rewardFactor = float(damage) / float(Pokemon2.maxHp) * reward_damage_multiplier
        TeamAttacker.reward += rewardFactor

        # Pokemon2.healthPercentage = (Pokemon2.hp / Pokemon2.maxHp) * 100
        Pokemon2.healthPercentage = math.ceil(
            ((Pokemon2.hp / Pokemon2.maxHp) * 100))
    # Pokemon1 used a non-damaging move
    else:
        # temp -----------------------------------------------------
        damage = 100
        Pokemon2.hp -= damage
        Pokemon2.healthPercentage = math.ceil(
            ((Pokemon2.hp / Pokemon2.maxHp) * 100))
        # temp -----------------------------------------------------

    # criticalMultiplier = 1/16th chance it equals 1.5, otherwise it's 1.0
    # roll = Random number between 0.85 and 1.0 (inclusive)
    # stab = (Same type attack bonus) = 1.5 if the move's type matches any of the user's types, 1.0 otherwise
    # typeEffectiveness = This can be 0 (ineffective); 0.25, 0.5 (not very effective); 1 (normally effective);
    #        2, or 4 (super effective), depending on both the move's and target's types.
    # burn = 0.5 if the user is burned and is using a physical move
    # other = this is where we could include damage boosts from items or abilities
    # modifier = critical * random * stab * type * burn * other
    # damage = [( [ (([2*Pokemon1.level]/5)+2) *pokemon1.moves.basePower * (Pokemon1.attackOrSpecialAttack/Pokemon2.defenseOrSpecialDefense)] /50) +2] * Modifier


# Generates a Team object for Team 1 and returns it
def generate_team_1():
    # -------------------------creating moves---------------------------------------
    # pikachu moves
    Thunderbolt = Move(
        "Thunderbolt",
        pokemon_types_dict["Electric"],
        "special",
        90,
        100,
        "15 percent chance to paralyze IMPLEMENT LATER",
    )
    SignalBeam = Move(
        "Signal Beam",
        pokemon_types_dict["Bug"],
        "special",
        75,
        100,
        "15 percent chance to confuse IMPLEMENT LATER",
    )
    NastyPlot = Move(
        "Nasty Plot",
        pokemon_types_dict["Dark"],
        "status",
        None,
        101,
        "Raises secial attack by 2 stages",
    )
    ThunderWave = Move(
        "Thunder Wave",
        pokemon_types_dict["Electric"],
        "status",
        None,
        100,
        "Paralyzes opposing pokemon",
    )
    Surf = Move(
        "Surf",
        pokemon_types_dict["Water"],
        "special",
        95,
        100,
        "None",
    )
    # snorlax moves
    BodySlam = Move(
        "Body Slam",
        pokemon_types_dict["Normal"],
        "physical",
        85,
        100,
        "30 percent chance to paralyze",
    )
    Rest = Move(
        "Rest",
        pokemon_types_dict["Psychic"],
        "status",
        None,
        101,
        "completely restore health, you now have the sleep status condition",
    )
    Yawn = Move(
        "Yawn",
        pokemon_types_dict["Normal"],
        "status",
        None,
        101,
        "puts opposing pokemon asleep after the next turn",
    )
    SleepTalk = Move(
        "Sleep Talk",
        pokemon_types_dict["Normal"],
        "status",
        None,
        101,
        "uses one of the other 3 moves randomly if you are asleep",
    )
    Crunch = Move(
        "Crunch",
        pokemon_types_dict["Dark"],
        "physical",
        80,
        100,
        "20 percent chance to lower the target's defense by one stage"
    )
    BrickBreak = Move(
        "Brick Break",
        pokemon_types_dict["Fighting"],
        "physical",
        75,
        100,
        "None"
    )
    Earthquake = Move("Earthquake", pokemon_types_dict["Ground"], "physical",
                      100, 100, None)
    # Wishcash moves
    HydroPump = Move("Hydro Pump", pokemon_types_dict["Water"], "special", 120,
                     80, None)
    EarthPower = Move(
        "Earth Power",
        pokemon_types_dict["Ground"],
        "special",
        90,
        100,
        "10 percent chance to lower the target's spdef by 1 stage",
    )
    Blizzard = Move(
        "Blizzard",
        pokemon_types_dict["Ice"],
        "special",
        120,
        70,
        "10 percent chance to freeze target",
    )
    HyperBeam = Move(
        "HyperBeam",
        pokemon_types_dict["Normal"],
        "special",
        150,
        90,
        "User cannot move next turn",
    )
    ZenHeadbutt = Move(
        "Zen HeadButt",
        pokemon_types_dict["Psychic"],
        "physical",
        80,
        90,
        "20 percent chance to flinch the target"
    )

    # create Team1
    pikachuMoves = [Thunderbolt, SignalBeam, BodySlam, Surf]
    snorlaxMoves = [BodySlam, Crunch, Earthquake, BrickBreak]
    wishcashMoves = [HydroPump, EarthPower, Blizzard, ZenHeadbutt]
    Pikachu = Pokemon(
        "Pikachu",
        pokemon_types_dict["Electric"],
        None,
        100,
        pikachuMoves,
        pokemon_conditions_dict["None"],
        211,
        103,
        218,
        96,
        117,
        279,
        "Static",
        "Light Ball",
    )
    Snorlax = Pokemon(
        "Snorlax",
        pokemon_types_dict["Normal"],
        None,
        100,
        snorlaxMoves,
        pokemon_conditions_dict["None"],
        462,
        319,
        149,
        166,
        350,
        96,
        "Thick Fat (reduce incoming ice and fire damage my 50 percent)",
        "Chesto Berry (immediately cure yourself from sleep, one time use, can still attack that turn)",
    )
    WishCash = Pokemon(
        "WishCash",
        pokemon_types_dict["Water"],
        pokemon_types_dict["Ground"],
        100,
        wishcashMoves,
        pokemon_conditions_dict["None"],
        361,
        144,
        276,
        182,
        179,
        219,
        "Oblivious: does nothing useful, feel free to make up your own ability",
        "Halves damag taken from a supereffective grass type attack, single use",
    )
    return Team("Team1", Pikachu, Snorlax, WishCash)


# Generates a Team object for Team 2 and returns it
def generate_team_2():
    # -------------------------creating moves---------------------------------------
    # Charizard Moves
    Flamethrower = Move(
        "Flamethrower",
        pokemon_types_dict["Fire"],
        "special",
        95,
        100,
        "10 percent chance to burn target",
    )
    SolarBeam = Move(
        "SolarBeam",
        pokemon_types_dict["Grass"],
        "special",
        120,
        100,
        "Charges turn 1, hits turn 2",
    )
    Earthquake = Move("Earthquake", pokemon_types_dict["Ground"], "physical",
                      100, 100, None)
    FocusBlast = Move(
        "FocusBlast",
        pokemon_types_dict["Fighting"],
        "special",
        120,
        70,
        "10 percent chance to lower target's spdef my 1 stage",
    )
    # Blastoise Moves
    HydroPump = Move("Hydro Pump", pokemon_types_dict["Water"], "special", 120,
                     80, None)
    HiddenPowerGrass = Move("Hidden Power Grass", pokemon_types_dict["Grass"],
                            "special", 70, 100, None)
    # Venusaur Moves
    GigaDrain = Move(
        "Giga Drain",
        pokemon_types_dict["Grass"],
        "special",
        70,
        100,
        "User recovers 50 percent of damage dealt in hp by number, not percentage",
    )
    SludgeBomb = Move(
        "Sludge Bomb",
        pokemon_types_dict["Poison"],
        "special",
        90,
        100,
        "30 percent chance to poison the target",
    )
    SleepPowder = Move(
        "Sleep Powder",
        pokemon_types_dict["Grass"],
        "status",
        None,
        75,
        "Causes the target to fall asleep",
    )
    Synthesis = Move(
        "Synthesis",
        pokemon_types_dict["Grass"],
        "status",
        None,
        101,
        "Heals the user by 50 percent max hp",
    )
    LeafStorm = Move(
        "Leaf Storm",
        pokemon_types_dict["Grass"],
        "special",
        140,
        90,
        "Lowers the user's special attack by two stages"
    )

    # ------------------------------------------------------------------------------
    # create Team2
    charizardMoves = [Flamethrower, SolarBeam, Earthquake, FocusBlast]
    blastoiseMoves = [HydroPump, Earthquake, FocusBlast, HiddenPowerGrass]
    venusaurMoves = [GigaDrain, SludgeBomb, Earthquake, LeafStorm]
    Charizard = Pokemon(
        "Charizard",
        pokemon_types_dict["Fire"],
        pokemon_types_dict["Flying"],
        100,
        charizardMoves,
        pokemon_conditions_dict["None"],
        297,
        225,
        317,
        192,
        185,
        299,
        "Blaze: when under 1/3 health, your fire moves do 1.5 damage",
        "Power Herb: 2 turn attacks skip charging turn. 1 time use",
    )
    Blastoise = Pokemon(
        "Blastoise",
        pokemon_types_dict["Water"],
        None,
        100,
        blastoiseMoves,
        pokemon_conditions_dict["None"],
        299,
        180,
        294,
        236,
        247,
        255,
        "Torrent: when under 1/3 hp, your water attacks do 1.5 damage",
        "Sitrus Berry: restores 1/4 max hp when at or under 1/2 max hp",
    )
    Venusaur = Pokemon(
        "Venusaur",
        pokemon_types_dict["Grass"],
        pokemon_types_dict["Poison"],
        100,
        venusaurMoves,
        pokemon_conditions_dict["None"],
        301,
        152,
        299,
        203,
        328,
        196,
        "Overgrow: when  under 1/3 health, your grass moves do 1.5 damage",
        "At the end of every turn, the user restores 1/16 of its maximum hp",
    )
    return Team("Team2", Charizard, Blastoise, Venusaur)

# Returns the (n - 1) move that does the highest damage, from 0 to 3.
def evaluator_highest_damage_action(team: Team):
    active = team.activePokemon
    best = 0
    highest_damage_move = active.moves[0].basePower \
                          * float(active.moves[0].accuracy) / 100.0

    for i in range(1, 3):
        tmp = active.moves[i].basePower * float(active.moves[i].accuracy) / 100.0

        if tmp > highest_damage_move:
            highest_damage_move = tmp
            best = i

    return i


# Pokemon simulation game when running pokemon.py
# During machine learning team generation and battle simulation will happen independently of battleSim() function
def main():
    # Generate teams
    Team1 = generate_team_1()
    Team2 = generate_team_2()

    # test print
    # delayPrint("This pokemon AI wants to be the very best, like no one ever was\n")

    # battle the teams
    battleSim(Team1, Team2)


if __name__ == "__main__":
    main()
