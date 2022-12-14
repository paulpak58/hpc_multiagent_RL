#include "open_spiel/algorithms/minimax.h"
#include <algorithm>
#include <limits>
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
using namespace std;
namespace open_spiel{
namespace algorithms {
namespace {
	/*	Minmax alpha-beta algorithms
	 *	Args:
	 *		state: current game state
	 *		depth: max depth for min/max search
	 *		alpha: best value that max player can guarantee
	 *		beta: best value that min player can guarantee
	 *		value_function: maps Spiel 'State' to number
	 *		maximizing_player_id: id of MAX
	 *	Returns:
	 *		Optimal value of sub-game starting in initial state
	 */
double _alpha_beta(State* state, int depth, double alpha, double bea,function<double(const State&)> value_function,Player maximizing_player,Action* best_action){
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
}	// namespace

pair<double,Action> AlphaBetaSearch(Game& game,const State* state,function<double(const State&)> value_function,int depth_limit,Player maximizing_player){

}

}
}
