import pyspiel

print(pyspiel.registered_names(),'\n')

# Load Game
game = pyspiel.load_game('tic_tac_toe')
print('Game: ',game)

# Various properties of games
print('Num Players: ',game.num_players(),'\nMax Utility: ',game.max_utility(),'\nMin Utility: ',game.min_utility(),'\nNumber of Distinct Actions: ',game.num_distinct_actions())

# Initial States
state = game.new_initial_state()
print('====State Information====')
print('Initial State: ',state, '\nCurrent Player: ',state.current_player(),'\nIs Terminal: ',state.is_terminal(),'\nReturns: ',state.returns(),'\nLegal Actions: ',state.legal_actions())

# Playing game
state = game.new_initial_state()
state.apply_action(1)
print('Next state: ',state)
print('Player turn: ',state.current_player())
state.apply_action(2)
state.apply_action(4)
state.apply_action(0)
state.apply_action(7)
print('Next state: ',state)
print('Is Terminal: ',state.is_terminal())
print('Return of P0: ',state.player_return(0))
print('Current Player: ',state.current_player())

# Breakthrough with default parameters
game = pyspiel.load_game('breakthrough')
state = game.new_initial_state()
print(state)
for action in state.legal_actions():
    print(f"{action}{state.action_to_string(action)}")


# Normal-form games; Evolutionary Dynamics
print('\nNORMAL-FORM; EVOLUTIONARY DYNAMICS')
game = pyspiel.create_matrix_game([[1,-1],[-1,1]],[[-1,1],[1,-1]])
print(game)
state = game.new_initial_state()
print('State: ',state,'\nCurrent Player: ',state.current_player(),'\nLegal Actions for P0: ',state.legal_actions(0),'\nLegal Actions for P1: ',state.legal_actions(1),'\nIs Terminal: ',state.is_terminal())

# Joint Action
state.apply_actions([0,0])
print('Returns: ',state.returns())

# Evolutionary Dynamics
from open_spiel.python.egt import dynamics
from open_spiel.python.egt.utils import game_payoffs_array
import numpy as np
game = pyspiel.load_matrix_game('matrix_rps')   # RPS
payoff_matrix = game_payoffs_array(game)        # Normal-form to np
dyn = dynamics.SinglePopulationDynamics(payoff_matrix,dynamics.replicator)
x = np.array([0.2,0.2,0.6])
dyn(x)

# Choose step size and apply dynamic
alpha = 0.01
x += alpha*dyn(x)
print('Evolutionary step: ',x)
x += alpha*dyn(x)
print('Evolutionary step: ',x)
x += alpha*dyn(x)
x += alpha*dyn(x)
x += alpha*dyn(x)
x += alpha*dyn(x)
print('Evolutionary step: ',x)


# Chance Node; Partially Observed Games
