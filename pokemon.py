# John Meents & Daniel Moore

import time
import numpy as np
import sys

# -------------------------------------Global Data--------------------------------------
pokemon_types = ["Normal", "Fire", "Water", "Electric", "Grass", "Ice",
                 "Fighting", "Poison", "Ground", "Flying", "Psychic",
                 "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"]

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
        time.sleep(0.03)  # How fast to print


# Team Class
class Team:
    def __init__(self, teamName, Pokemon1, Pokemon2, Pokemon3):
        self.teamName = teamName
        self.Pokemon1 = Pokemon1
        self.Pokemon2 = Pokemon2
        self.Pokemon3 = Pokemon3
        self.acitvePokemon = Pokemon1
        self.hasAvailablePokemon = True


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
        self.ability = ability  # implement last
        self.item = item  # implement last
        self.level = 100  # pokemon are automatically set to level 100


# Moves Class
class Move:
    def __init__(self, moveName, moveType, physpc, basePower, accuracy, effect):
        self.moveName = moveName
        self.moveType = moveType
        self.physpc = physpc
        self.basePower = basePower
        self.accuracy = accuracy  # moves with 101 accuracy cannot be lowered
        self.effect = effect


# battle the two teams
def battleSim(Team1, Team2):
    # players start out with a predetermined pokemon1
    # If both teams have usable pokemon: loop:
    while (Team1.hasAvailablePokemon and Team2.hasAvailablePokemon):
        # if both active pokemon have health: loop:
        while (Team1.acitvePokemon.hp != 0 and Team2.acitvePokemon.hp != 0):
            # TO DO

            # print out fight info
            delayPrint(Team1.activePokemon.name)
            delayPrint(Team2.activePokemon.name)
            # moves are chosen
            # if they swith pokemon, that action happens first before the opponent moves
            # determine who moves first
            # do damage calculations for turn
            # if a pokemon faints:
            # print pokemonName fainted!
            # chose an available pokemon
            # print Go PokemonName!

            # TO DO
        if (Team1.acitvePokemon.hp == 0 and Team2.acitvePokemon.hp != 0):
            delayPrint("Enter an integer for a command: ")
            txt = input()
        elif (Team1.acitvePokemon.hp != 0 and Team2.acitvePokemon.hp == 0):
            delayPrint("Enter an integer for a command: ")
            txt = input()
        else:
            print("I don't think both pokemon will faint on the same turn?")

    if (Team1.hasAvailablePokemon and (Team2.hasAvailablePokemon == false)):
        delayPrint("Team 1 wins!\n")
    else:
        delayPrint("Team 2 wins!\n")


def damageCalc(Pokemon1, Pokemon2, move):  # damage calculation function
    damage = 0

    # critical = 1/16th chance it equals 1.5, otherwise it's 1.0
    # random = Random number between 0.85 and 1.0 (inclusive)
    # stab = (Same type attack bonus) = 1.5 if the move's type matches any of the user's types, 1.0 otherwise
    # type = This can be 0 (ineffective); 0.25, 0.5 (not very effective); 1 (normally effective);
    #        2, or 4 (super effective), depending on both the move's and target's types.
    # burn = 0.5 if the user is burned and is using a physical move
    # other = this is where we could include damage boosts from items or abilities
    # modifier = critical * random * stab * type * burn * other
    # damage = [( [ (([2*Pokemon1.level]/5)+2) *pokemon1.moves.basePower * (Pokemon1.attackOrSpecialAttack/Pokemon2.defenseOrSpecialDefense)] /50) +2] * Modifier


def main():
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
    # Charizard Moves
    Flamethrower = Move(
        "Flamethrower", pokemon_types[1], "special", 95, 100, "10 percent chance to burn target")
    SolarBeam = Move(
        "SolarBeam", pokemon_types[4], "special", 120, 100, "Charges turn 1, hits turn 2")
    Earthquake = Move(
        "EarthQuake", pokemon_types[8], "physical", 100, 100, None)
    FocusBlast = Move("FocusBlast", pokemon_types[6], "special", 120,
                      70, "10 percent chance to lower target's spdef my 1 stage")
    # Blastoise Moves (he also has moves already implemented above)
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

    # test print
    delayPrint("This pokemon AI wants to be the very best, like no one ever was\n")

    # create Team1
    pikachuMoves = [Thunderbolt, SignalBeam, NastyPlot, ThunderWave]
    snorlaxMoves = [BodySlam, Rest, Yawn, SleepTalk]
    wishcashMoves = [HydroPump, EarthPower, Blizzard, HyperBeam]
    Pikachu = Pokemon("Pikachu", pokemon_types[3], None, 100, pikachuMoves, None,
                      211, 103, 218, 96, 117, 279, "Static", "Light Ball")
    Snorlax = Pokemon("Snorlax", pokemon_types[0], None, 100, snorlaxMoves, None, 462, 319, 149, 166, 350, 96,
                      "Thick Fat (reduce incoming ice and fire damage my 50 percent)", "Chesto Berry (immediately cure yourself from sleep, one time use, can still attack that turn)")
    WishCash = Pokemon("WishCash", pokemon_types[2], pokemon_types[8], 100, wishcashMoves, None, 361, 144, 276, 182, 179, 219,
                       "Oblivious: does nothing useful, feel free to make up your own ability", "Halves damag taken from a supereffective grass type attack, single use")
    Team1 = Team("Team1", Pikachu, Snorlax, WishCash)

    # create Team2
    charizardMoves = [Flamethrower, SolarBeam, Earthquake, FocusBlast]
    blastoiseMoves = [HydroPump, Earthquake, FocusBlast, HiddenPowerGrass]
    venusaurMoves = [GigaDrain, SludgeBomb, SleepPowder, Synthesis]
    Charizard = Pokemon("Charizard", pokemon_types[1], pokemon_types[9], 100, charizardMoves, None, 297, 225, 317, 192, 185, 299,
                        "Blaze: when under 1/3 health, your fire moves do 1.5 damage", "Power Herb: 2 turn attacks skip charging turn. 1 time use")
    Blastoise = Pokemon("Blastoise", pokemon_types[2], None, 100, blastoiseMoves, None, 299, 180, 294, 236, 247, 255,
                        "Torrent: when under 1/3 hp, your water attacks do 1.5 damage", "Sitrus Berry: restores 1/4 max hp when at or under 1/2 max hp")
    Venusaur = Pokemon("Venusaur", pokemon_types[4], pokemon_types[7], 100, venusaurMoves, None, 301, 152, 299, 203, 328, 196,
                       "Overgrow: when  under 1/3 health, your grass moves do 1.5 damage", "At the end of every turn, the user restores 1/16 of its maximum hp")
    Team2 = Team("Team2", Charizard, Blastoise, Venusaur)

    # test print
    delayPrint(Team1.Pokemon1.ability)
    delayPrint("\n")


if __name__ == "__main__":
    main()
