#include "open_spiel/algorithms/minimax.h"
#include <algorithm>
#include <limits>
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

// Computes the minimax alpha-beta algorithm
__global__ void alpha_beta(State* state,int depth,double alpha,double beta,function<double>(const State&)> value_function,Player maximizing_player,Action* best_action,unsigned int n){

	extern volatile __shared__ float s[];
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;

	if(idx<n && tid<blockDim.x){
		if(state->IsTerminal()){
			s[tid] input[idx];
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
	}
}
