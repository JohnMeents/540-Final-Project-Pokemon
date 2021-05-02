# John Meents & Daniel Moore
# TODO
# fitness funcion
# function to get the state of the game that returns every relevant variable

import time
import numpy as np
import sys

# -------------------------------------Global Data--------------------------------------
pokemon_types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                 "Fighting", "Poison", "Ground", "Flying", "Psychic",
                 "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

# Used to map string values of pokemon_types to an integer for machine learning
pokemon_types_dict = {'Normal': 1, 'Fire': 2, 'Water': 3, 'Electric': 4, 'Grass': 5, 'Ice': 6, 'Fighting': 7,
                      'Poison': 8, 'Ground': 9, 'Flying': 10, 'Psychic': 11, 'Bug': 12, 'Rock': 13, 'Ghost': 14,
                      'Dragon': 15, 'Dark': 16, 'Steel': 17, 'Fairy': 18}


# A 2 Dimenstional Numpy Array Of Damage Multipliers For Attacking Pokemon:
# Origin of this table is at the top left and the sequence follows the list above
# the row represents the attacking pokemon, the column represents the defending pokemon
# the value represents the type damage modifier
# 0: immune
# 1: normal effectiveness
# 2: super effective
# 1/2: not very effective
# for reference, see damage type chart on discord

damage_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1/2, 0, 1, 1, 1/2, 1],
                        [1, 1/2, 1/2, 1, 2, 2, 1, 1, 1, 1,
                            1, 2, 1/2, 1, 1/2, 1, 2, 1],
                        [1, 2, 1/2, 1, 1/2, 1, 1, 1, 2,
                            1, 1, 1, 2, 1, 1/2, 1, 1, 1],
                        [1, 1, 2, 1/2, 1/2, 1, 1, 1, 0,
                            2, 1, 1, 1, 1, 1/2, 1, 1, 1],
                        [1, 1/2, 2, 1, 1/2, 1, 1, 1/2, 2, 1 /
                            2, 1, 1/2, 2, 1, 1/2, 1, 1/2, 1],
                        [1, 1/2, 1/2, 1, 2, 1/2, 1, 1, 2,
                            2, 1, 1, 1, 1, 2, 1, 1/2, 1],
                        [2, 1, 1, 1, 1, 2, 1, 1/2, 1, 1/2,
                            1/2, 1/2, 2, 0, 1, 2, 2, 1/2],
                        [1, 1, 1, 1, 2, 1, 1, 1/2, 1/2, 1,
                            1, 1, 1/2, 1/2, 1, 1, 0, 2],
                        [1, 2, 1, 2, 1/2, 1, 1, 2, 1, 0,
                            1, 1/2, 2, 1, 1, 1, 2, 1],
                        [1, 1, 1, 1/2, 2, 1, 2, 1, 1, 1,
                            1, 2, 1/2, 1, 1, 1, 1/2, 1],
                        [1, 1, 1, 1, 1, 1, 2, 2, 1, 1,
                            1/2, 1, 1, 1, 1, 0, 1/2, 1],
                        [1, 1/2, 1, 1, 2, 1, 1/2, 1/2, 1, 1 /
                            2, 2, 1, 1, 1/2, 1, 2, 1/2, 1/2],
                        [1, 2, 1, 1, 1, 2, 1/2, 1, 1/2,
                            2, 1, 2, 1, 1, 1, 1, 1/2, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            2, 1, 1, 2, 1, 1/2, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 2, 1, 1/2, 0],
                        [1, 1, 1, 1, 1, 1, 1/2, 1, 1, 1,
                            2, 1, 1, 2, 1, 1/2, 1, 1/2],
                        [1, 1/2, 1/2, 1/2, 1, 2, 1, 1, 1,
                            1, 1, 1, 2, 1, 1, 1, 1/2, 2],
                        [1, 1/2, 1, 1, 1, 1, 2, 1/2, 1, 1, 1, 1, 1, 1, 2, 2, 1/2, 1]])

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

    # Returns an array that represents parameters about this object
    def toArray(self):
        return self.Pokemon1.toArray() + \
            self.Pokemon2.toArray() + \
            self.Pokemon3.toArray() + \
            [self.activePokemonN, int(self.hasAvailablePokemon)]


# Pokemon Class
class Pokemon:
    def __init__(self, name, type1, type2, healthPercentage, moves, status, hp, attack, spattack, defense, spdefense, speed, ability, item):
        self.name = name
        self.types = type1
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
        self.level = 100  # pokemon are automatically set to level 100

    # Returns an array that represents parameters about this object
    def toArray(self):
        # Generate integer values for these strings
        types_int = pokemon_types_dict[self.types] if self.types is not None else 0
        type2_int = pokemon_types_dict[self.type2] if self.type2 is not None else 0

        return [types_int, type2_int, self.healthPercentage, self.attack,
                self.spattack, self.defense, self.spdefense, self.speed, self.maxHp] \
               + [item for items in self.moves for item in items.toArray()]

# Moves Class
class Move:
    def __init__(self, moveName, moveType, physpc, basePower, accuracy, effect):
        self.moveName = moveName
        self.moveType = moveType
        self.physpc = physpc
        self.basePower = basePower
        self.accuracy = accuracy  # moves with 101 accuracy cannot be lowered
        self.effect = effect  # implement last

    # Returns an array that represents parameters about this object
    def toArray(self):
        physpc_int = 0 if self.physpc is None \
            else 1 if self.physpc == 'special' \
            else 2 if self.physpc == 'status' \
            else -1
        basePower_int = self.basePower if self.basePower is not None else -1
        return [pokemon_types_dict[self.moveType], physpc_int, basePower_int, self.accuracy]


# a function that takes an integer as an action, an int/bool as team and pokemon info
def fightSim(Team1, Team2, team1Action, team2Action, headless = False):
    Team1.reward = 0
    Team2.reward = 0
    # if they switch pokemon, that action happens first before the opponent moves
    # switching Team1 pokemon
    if(team1Action == 5 or team1Action == 6):
        if(team1Action == 5 and Team1.Pokemon1.hp > 0 and Team1.activePokemon != Team1.Pokemon1):
            Team1.activePokemon = Team1.Pokemon1
            Team1.activePokemonN = 1
        elif(team1Action == 5 and Team1.Pokemon2.hp > 0 and Team1.activePokemon != Team1.Pokemon2):
            Team1.activePokemon = Team1.Pokemon2
            Team1.activePokemonN = 2
        elif(team1Action == 5 and Team1.Pokemon3.hp > 0 and Team1.activePokemon != Team1.Pokemon3):
            Team1.activePokemon = Team1.Pokemon3
            Team1.activePokemonN = 3

        elif(team1Action == 6 and Team1.Pokemon3.hp > 0 and Team1.activePokemon != Team1.Pokemon3):
            Team1.activePokemon = Team1.Pokemon3
            Team1.activePokemonN = 3
        elif(team1Action == 6 and Team1.Pokemon2.hp > 0 and Team1.activePokemon != Team1.Pokemon2):
            Team1.activePokemon = Team1.Pokemon2
            Team1.activePokemonN = 2
        elif(team1Action == 6 and Team1.Pokemon1.hp > 0 and Team1.activePokemon != Team1.Pokemon1):
            Team1.activePokemon = Team1.Pokemon1
            Team1.activePokemonN = 1
        else:
            pass

    # switching Team2 pokemon
    if(team2Action == 5 or team2Action == 6):
        if(team2Action == 5 and Team2.Pokemon1.hp > 0 and Team2.activePokemon != Team2.Pokemon1):
            Team2.activePokemon = Team2.Pokemon1
            Team2.activePokemonN = 1
        elif(team2Action == 5 and Team2.Pokemon2.hp > 0 and Team2.activePokemon != Team2.Pokemon2):
            Team2.activePokemon = Team2.Pokemon2
            Team2.activePokemonN = 2
        elif(team2Action == 5 and Team2.Pokemon3.hp > 0 and Team2.activePokemon != Team2.Pokemon3):
            Team2.activePokemon = Team2.Pokemon3
            Team2.activePokemonN = 3

        elif(team2Action == 6 and Team2.Pokemon3.hp > 0 and Team2.activePokemon != Team2.Pokemon3):
            Team2.activePokemon = Team2.Pokemon3
            Team2.activePokemonN = 3
        elif(team2Action == 6 and Team2.Pokemon2.hp > 0 and Team2.activePokemon != Team2.Pokemon2):
            Team2.activePokemon = Team2.Pokemon2
            Team2.activePokemonN = 2
        elif(team2Action == 6 and Team2.Pokemon1.hp > 0 and Team2.activePokemon != Team2.Pokemon1):
            Team2.activePokemon = Team2.Pokemon1
            Team2.activePokemonN = 1
        else:
            pass

    # if Team1 moves first
    if(Team1.activePokemon.speed >= Team2.activePokemon.speed):
        # do calc
        damageCalc(Team1.activePokemon, Team2.activePokemon, team1Action, headless=headless)
        Team1.reward += 3
        # other pokemon attacks if they didn't just faint
        if(Team1.activePokemon.hp > 0 and Team2.activePokemon.hp > 0):
            damageCalc(Team2.activePokemon, Team1.activePokemon, team2Action, headless=headless)
            Team2.reward += 3
    # if Team2 moves first
    else:
        # do calc
        damageCalc(Team2.activePokemon, Team1.activePokemon, team2Action, headless=headless)
        Team2.reward += 3
        # other pokemon attacks if they didn't just faint
        if(Team1.activePokemon.hp > 0 and Team2.activePokemon.hp > 0):
            damageCalc(Team1.activePokemon, Team2.activePokemon, team1Action, headless=headless)
            Team1.reward += 3

    # if a pokemon faints:
    if(Team1.activePokemon.hp <= 0):
        Team1.reward -= 20
        if (True):
            print(Team1.activePokemon.name +
                       " fainted!")

    if(Team2.activePokemon.hp <= 0):
        Team2.reward -= 20
        if(True):
            print(Team2.activePokemon.name +
                       " fainted!")

    if (Team1.activePokemon.hp <= 0 and Team2.activePokemon.hp > 0 and Team1.hasAvailablePokemon):
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

    Team1.reward -= Team1.roundNumber / 10
    Team2.reward -= Team2.roundNumber / 10

    Team1.roundNumber += 1
    Team2.roundNumber += 1


# Returns an array of parameters
def getState(team1: Team, team2: Team):
    return team1.toArray() + team2.toArray()


def battleSim(Team1, Team2):
    txt = "default"
    chooseTeam1Move = ""
    chooseTeam2Move = ""
    # players start out with a predetermined pokemon1
    # If both teams have usable pokemon: loop:
    while (Team1.hasAvailablePokemon and Team2.hasAvailablePokemon):
        # if both active pokemon have health: loop:
        while (Team1.activePokemon.hp >= 0 and Team2.activePokemon.hp >= 0):
            # print out fight info
            delayPrint(Team1.activePokemon.name)
            delayPrint("\n")
            delayPrint(Team2.activePokemon.name)
            delayPrint("\n")

            validInput1 = False
            validInput2 = False
            # moves are chosen for Team 1
            while(validInput1 == False):
                delayPrint("Team 1, Enter and integer for the following command\n1: Move 1\n2: Move 2\n3: Move 3\n4: Move 4\n5: Switch to first available pokemon\n6: Switch to second available Pokemon\n")
                chooseTeam1Move = input()
                # If they chose 1-4
                if(chooseTeam1Move == "1" or chooseTeam1Move == "2" or chooseTeam1Move == "3" or chooseTeam1Move == "4"):
                    validInput1 = True
                # make sure "5" is a valid input (first benched pokemon)"
                elif(chooseTeam1Move == "5" and Team1.Pokemon1.hp > 0):
                    validInput1 = True
                elif(chooseTeam1Move == "5" and Team1.Pokemon2.hp > 0):
                    validInput1 = True
                elif(chooseTeam1Move == "5" and Team1.Pokemon3.hp > 0):
                    validInput1 = True
                # make sure "6" is a valid input (second benched pokemon)
                elif(chooseTeam1Move == "6" and Team1.Pokemon1.hp > 0 and Team1.Pokemon2.hp > 0 and Team1.Pokemon2.hp > 0):
                    validInput1 = True
                elif(chooseTeam1Move == "6" and Team1.Pokemon2.hp > 0):
                    validInput1 = True
                elif(chooseTeam1Move == "6" and Team1.Pokemon3.hp > 0):
                    validInput1 = True
                else:
                    validInput1 = False  # remains false

            # moves are chosen for Team 2
            while(validInput2 == False):
                delayPrint("Team 2, Enter and integer for the following command\n1: Move 1\n2: Move 2\n3: Move 3\n4: Move 4\n5: Switch to first available pokemon\n6: Switch to second available Pokemon\n")
                chooseTeam2Move = input()
                # If they choose 1-4
                if(chooseTeam2Move == "1" or chooseTeam2Move == "2" or chooseTeam2Move == "3" or chooseTeam2Move == "4"):
                    validInput2 = True
                # make sure "5" is a valid input (first benched pokemon)"
                elif(chooseTeam2Move == "5" and Team2.Pokemon1.hp > 0):
                    validInput2 = True
                elif(chooseTeam2Move == "5" and Team2.Pokemon2.hp > 0):
                    validInput2 = True
                elif(chooseTeam2Move == "5" and Team2.Pokemon3.hp > 0):
                    validInput2 = True
                # make sure "6" is a valid input (second benched pokemon)
                elif(chooseTeam2Move == "6" and Team2.Pokemon1.hp > 0 and Team2.Pokemon2.hp > 0 and Team2.Pokemon2.hp > 0):
                    validInput2 = True
                elif(chooseTeam2Move == "6" and Team2.Pokemon2.hp > 0):
                    validInput2 = True
                elif(chooseTeam2Move == "6" and Team2.Pokemon3.hp > 0):
                    validInput2 = True
                else:
                    validInput2 = False  # remains false

            # convert move choices from strings to ints
            chooseTeam1Move = int(chooseTeam1Move)
            chooseTeam2Move = int(chooseTeam2Move)

            # turn happens
            fightSim(Team1, Team2, chooseTeam1Move, chooseTeam2Move)

            # check if each team has available pokemon
            if(Team1.Pokemon1.hp <= 0 and Team1.Pokemon2.hp <= 0 and Team1.Pokemon3.hp <= 0):
                Team1.hasAvailablePokemon = False
            if(Team2.Pokemon1.hp <= 0 and Team2.Pokemon2.hp <= 0 and Team2.Pokemon3.hp <= 0):
                Team2.hasAvailablePokemon = False


    # print winning team
    if (Team1.hasAvailablePokemon and (Team2.hasAvailablePokemon is False)):
        delayPrint("Team 1 wins!\n")
    else:
        delayPrint("Team 2 wins!\n")

def isGameOver(team1: Team, team2: Team):
    return (team1.hasAvailablePokemon is False or team2.hasAvailablePokemon is False) \
           or (team1.hasAvailablePokemon and team2.hasAvailablePokemon is False) \
           or (team2.hasAvailablePokemon and team1.hasAvailablePokemon is False)


# Perform a step in the game simulation. This is primarily used by the AI
def step(team1: Team, team2: Team, team1_action, team2_action, headless=True):

    # Perform the turn/round
    fightSim(team1, team2, team1_action, team2_action, headless=headless)

    # Determine observation, or new state space
    observation = getState(team1, team2)

    reward_a = team1.reward
    reward_b = team2.reward

    # Determine if game (episode) is over
    gameOver = isGameOver(team1, team2)

    # Return observation, rewards, gameOver
    # TODO: Is there any other info the AI needs?
    return observation, [reward_a, reward_b], gameOver


def damageCalc(Pokemon1, Pokemon2, move, headless = False):  # damage calculation function
    # do nothing if the move is 5 or 6 (a switch)
    if(move == 5 or move == 6):
        return

    # temp
    damage = 100
    Pokemon2.hp -= damage
    Pokemon2.healthPercentage = Pokemon2.hp/Pokemon2.maxHp
    # delayPrint(damage)
    # delayPrint("\n")
    if not headless:
        delayPrint(Pokemon1.name)
        delayPrint(" used ")
        delayPrint(Pokemon1.moves[(move - 1)].moveName)
        delayPrint("!\n")
        delayPrint(Pokemon2.name + " has ")
        print(Pokemon2.healthPercentage)
        delayPrint(" remaining health\n")

    # critical = 1/16th chance it equals 1.5, otherwise it's 1.0
    # random = Random number between 0.85 and 1.0 (inclusive)
    # stab = (Same type attack bonus) = 1.5 if the move's type matches any of the user's types, 1.0 otherwise
    # type = This can be 0 (ineffective); 0.25, 0.5 (not very effective); 1 (normally effective);
    #        2, or 4 (super effective), depending on both the move's and target's types.
    # burn = 0.5 if the user is burned and is using a physical move
    # other = this is where we could include damage boosts from items or abilities
    # modifier = critical * random * stab * type * burn * other
    # damage = [( [ (([2*Pokemon1.level]/5)+2) *pokemon1.moves.basePower * (Pokemon1.attackOrSpecialAttack/Pokemon2.defenseOrSpecialDefense)] /50) +2] * Modifier


# Generates a Team object for Team 1 and returns it
def generate_team_1():
    # -------------------------creating moves---------------------------------------
    # pikachu moves
    Thunderbolt = Move("Thunderbolt", pokemon_types[3], "special",
                       90, 100, "15 percent chance to paralyze IMPLEMENT LATER")
    SignalBeam = Move("Signal Beam", pokemon_types[11], "special", 75, 100,
                      "15 percent chance to confuse IMPLEMENT LATER")
    NastyPlot = Move("Nasty Plot", pokemon_types[15], "status", None, 101,
                     "Raises secial attack by 2 stages")
    ThunderWave = Move("Thunder Wave", pokemon_types[3], "status", None,
                       100, "Paralyzes opposing pokemon")
    # snorlax moves
    BodySlam = Move(
        "Body Slam", pokemon_types[0], "physical", 85, 100, "30 percent chance to paralyze")
    Rest = Move("Rest", pokemon_types[10], "status", None, 101,
                "completely restore health, you now have the sleep status condition")
    Yawn = Move("Yawn", pokemon_types[0], "status", None, 101,
                "puts opposing pokemon asleep after the next turn")
    SleepTalk = Move("Sleep Talk", pokemon_types[0], "status", None, 101,
                     "uses one of the other 3 moves randomly if you are asleep")
    # Wishcash moves
    HydroPump = Move("Hydro Pump", pokemon_types[2], "special", 120, 80, None)
    EarthPower = Move("Earth Power", pokemon_types[8], "special", 90,
                      100, "10 percent chance to lower the target's spdef by 1 stage")
    Blizzard = Move(
        "Blizzard", pokemon_types[5], "special", 120, 70, "10 percent chance to freeze target")
    HyperBeam = Move(
        "HyperBeam", pokemon_types[0], "special", 150, 90, "User cannot move next turn")

    # create Team1
    pikachuMoves = [Thunderbolt, SignalBeam, NastyPlot, ThunderWave]
    snorlaxMoves = [BodySlam, Rest, Yawn, SleepTalk]
    wishcashMoves = [HydroPump, EarthPower, Blizzard, HyperBeam]
    Pikachu = Pokemon("Pikachu", pokemon_types[3], None, 100, pikachuMoves, None,
                      211, 103, 218, 96, 117, 279, "Static", "Light Ball")
    Snorlax = Pokemon("Snorlax", pokemon_types[0], None, 100, snorlaxMoves, None, 462, 319, 149, 166, 350, 96,
                      "Thick Fat (reduce incoming ice and fire damage my 50 percent)",
                      "Chesto Berry (immediately cure yourself from sleep, one time use, can still attack that turn)")
    WishCash = Pokemon("WishCash", pokemon_types[2], pokemon_types[8], 100, wishcashMoves, None, 361, 144, 276, 182,
                       179, 219,
                       "Oblivious: does nothing useful, feel free to make up your own ability",
                       "Halves damag taken from a supereffective grass type attack, single use")
    return Team("Team1", Pikachu, Snorlax, WishCash)

# Generates a Team object for Team 2 and returns it
def generate_team_2():
    # -------------------------creating moves---------------------------------------
    # Charizard Moves
    Flamethrower = Move(
        "Flamethrower", pokemon_types[1], "special", 95, 100, "10 percent chance to burn target")
    SolarBeam = Move(
        "SolarBeam", pokemon_types[4], "special", 120, 100, "Charges turn 1, hits turn 2")
    Earthquake = Move(
        "EarthQuake", pokemon_types[8], "physical", 100, 100, None)
    FocusBlast = Move("FocusBlast", pokemon_types[6], "special", 120,
                      70, "10 percent chance to lower target's spdef my 1 stage")
    # Blastoise Moves
    HydroPump = Move("Hydro Pump", pokemon_types[2], "special", 120, 80, None)
    HiddenPowerGrass = Move("Hidden Power Grass",
                            pokemon_types[4], "special", 70, 100, None)
    # Venusaur Moves
    GigaDrain = Move("Giga Drain", pokemon_types[4], "special", 70, 100,
                     "User recovers 50 percent of damage dealt in hp by number, not percentage")
    SludgeBomb = Move(
        "Sludge Bomb", pokemon_types[7], "special", 90, 100, "30 percent chance to poison the target")
    SleepPowder = Move(
        "Sleep Powder", pokemon_types[4], "status", None, 75, "Causes the target to fall asleep")
    Synthesis = Move(
        "Synthesis", pokemon_types[4], "status", None, 101, "Heals the user by 50 percent max hp")

    # ------------------------------------------------------------------------------
    # create Team2
    charizardMoves = [Flamethrower, SolarBeam, Earthquake, FocusBlast]
    blastoiseMoves = [HydroPump, Earthquake, FocusBlast, HiddenPowerGrass]
    venusaurMoves = [GigaDrain, SludgeBomb, SleepPowder, Synthesis]
    Charizard = Pokemon("Charizard", pokemon_types[1], pokemon_types[9], 100, charizardMoves, None, 297, 225, 317, 192,
                        185, 299,
                        "Blaze: when under 1/3 health, your fire moves do 1.5 damage",
                        "Power Herb: 2 turn attacks skip charging turn. 1 time use")
    Blastoise = Pokemon("Blastoise", pokemon_types[2], None, 100, blastoiseMoves, None, 299, 180, 294, 236, 247, 255,
                        "Torrent: when under 1/3 hp, your water attacks do 1.5 damage",
                        "Sitrus Berry: restores 1/4 max hp when at or under 1/2 max hp")
    Venusaur = Pokemon("Venusaur", pokemon_types[4], pokemon_types[7], 100, venusaurMoves, None, 301, 152, 299, 203,
                       328, 196,
                       "Overgrow: when  under 1/3 health, your grass moves do 1.5 damage",
                       "At the end of every turn, the user restores 1/16 of its maximum hp")
    return Team("Team2", Charizard, Blastoise, Venusaur)


# Pokemon simulation game when running pokemon.py
# During machine learning team generation and battle simulation will happen independently of battleSim() function
def main():

    # Generate teams
    Team1 = generate_team_1()
    Team2 = generate_team_2()


    # test print
    #delayPrint("This pokemon AI wants to be the very best, like no one ever was\n")

    # battle the teams
    battleSim(Team1, Team2)


if __name__ == "__main__":
    main()
