#include "open_spiel/algorithms/minimax.h"
#include <algorithm>
#include <limits>
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
using namespace std;
namespace open_spiel{
namespace algorithms {
namespace {
	/*	Minimax alpha-beta algorithms
	 *	Args:
	 *		state: current game state
	 *		depth: max depth for min/max search
	 *		alpha: best value that max player can guarantee
	 *		beta: best value that min player can guarantee
	 *		value_function: maps Spiel 'State' to value
	 *		maximizing_player_id: id of MAX
	 *	Returns:
	 *		Optimal value of sub-game starting in initial state
	 */
double _alpha_beta(State* state, int depth, double alpha, double beta,function<double(const State&)> value_function,Player maximizing_player,Action* best_action){
	if(state->IsTerminal()){
		return state->PlayerReturn(maximizing_player);
	}
	if(depth==0) return value_function(*state);

	Player player = state->CurrentPlayer();
	if(player==maximizing_player){
		// Default value at each state
		double value = -numeric_limits<double>infinity();
		for(Action action:state->LegalActions()){
			state->ApplyAction(action);
			double child_value=_alpha_beta(state,depth-1,alpha,beta,value_function,maximizing_player,nullptr);
			state->UndoAction(player,action);
			if(child_value>value){
				value = child_value;
				if(best_action!=nullptr) 
					*best_action=action;
			}
			alpha = max(alpha,value);
			if(alpha>=beta) break;
		}
		return value;
	}else{			// SECOND player
		double value = numeric_limits<double>::infinity();
		for(Action action: state->LegalActions()){
			state->ApplyAction(action);
			double child_value=_alpha_beta(state,depth-1,alpha,beta,value_function,maximizing_player,nullptr);
			state->UndoAction(player,action);
			if(child_value<value){
				value = child_value;
				if(best_action!=nullptr) 
					*best_action=action;
			}
			beta = min(beta,value);
			if(alpha>=beta) break;
		}
		return value;
	}
}

pair<double,action> alphabetasearch(const game& game,const state* state,function<double(const state&)> value_function,int depth_limit,player maximizing_player){
	// minmax played between two players
	// set configurations for the game state
	SPIEL_CHECK_LE(game.numplayers(),2);
	GameType game_info = game.GetType();
	SPIEL_CHECK_EQ(game_info.chance_mode,GameType::ChanceMode::kDeterministic);
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
		maximizing_player = search_root->CurrentPlayer();
	}
	double infinity = numeric_limits<double>infinity();
	Action best_action = kInvalidAction;
	double value = _alpha_beta(search_root.get(),depth_limit,-infinity,infinity,value_function,maximizing_player,&best_action);
	return {value,best_action};
}

}	// namespace
}	// algorithms
}	// open_spiel
