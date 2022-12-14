#include "open_spiel/algorithms/minimax.h"
#include <algorithm>
#include <limits>
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
using namespace std;
namespace open_spiel{
namespace algorithms {
namespace {
	/*	Expectiminimax algorithm
	 *	Args:
	 *		state: current game state
	 *		depth: max depth for min/max search
	 *		value_function: maps Spiel 'State' to value
	 *		maximizing_player_id: id of MAX
	 *	Returns:
	 *		Optimal value of sub-game starting in initial state
	 */
double _expectiminimax(const State* state,int depth,function<double(const State&)> value_function,Player maximizing_player,Action* beset_action){
	if(state->IsTerminal()) return state->PlayerReturn(maximizing_player);
	if(depth==0) return value_function(*state);
	Player player = state->CurrentPlayer();
	if(state->IsChanceNode()){
		double value=0;
		for(const auto& p: state->ChanceOutcomes()){
			unique_ptr<State> child_state = state->Child(p.first);
			double child_value = _expectiminimax(child_state.get(), depth,value_function,maximizing_player,nullptr);
			value += p*child_value;
		}
		return value;
	}else if(player==maximizing_player){
		double value = -numeric_limits<double>
		for(Action action:state->LegalActions()){
			unique_ptr<State> child_state = state->Child(action);
			double child_value=_expectiminimax(child_state.get(),depth-1,value_function,maximizing_player,nullptr);
			if(child_value>value){
				value = child_value;
				if(best_action!=nullptr) 
					*best_action=action;
			}
		}
		return value;
	}else{
		double value = numeric_limits<double>
		for(Action action:state->LegalActions()){
			unique_ptr<State> child_state = state->Child(action);
			double child_value=_expectiminimax(child_state.get(),depth-1,value_function,maximizing_player,nullptr);
			if(child_value<value){
				value = child_value;
				if(best_action!=nullptr) 
					*best_action=action;
			}
		}
		return value;
	}
}

pair<double,action> ExpectiminimaxSearch(const game& game,const state* state,function<double(const state&)> value_function,int depth_limit,player maximizing_player){
	// minmax played between two players
	// set configurations for the game state
	SPIEL_CHECK_LE(game.numplayers(),2);
	GameType game_info = game.GetType();
	SPIEL_CHECK_EQ(game_info.chance_mode,GameType::ChanceMode::kExplicitStochastic);
	SPIEL_CHECK_EQ(game_info.information,GameType::Information::kPerfectInformation);
	SPIEL_CHECK_EQ(game_info.dynamics,GameType::Dynamics::kSequential);
	SPIEL_CHECK_EQ(game_info.utility,GameType::Utility::kZeroSum);
	SPIEL_CHECK_EQ(game_info.reward_model,GameType::RewardModel::kTerminal);
	unique_ptr<State> search_root;
	if(state==nullptr) 
		search_root = game.NewInitialState();
	else
		search_root = state->Clone();
	if(maximizing_player==kInvalidPlayer){
		SPIEL_CHECK_FALSE(search_root->IsChanceNode());
		maximizing_player = search_root->CurrentPlayer();
	}
	Action best_action = kInvalidAction;
	double value = _expectiminimax(search_root.get(),depth_limit,value_function,maximizing_player,&best_action);
	return {value,best_action};
}


}	// namespace
}	// algorithm
}	// open_spiel
