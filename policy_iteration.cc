#include <memory>
#include <string>
#include "open_spiel/algorithms/policy_iteration.h"
#include "open_spiel/games/tic_tac_toe.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
using namespace std;

int main(int argc, char **argv){
	shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("tic_tac_toe");
	absl::flat_hash_map<string,double> solution = open_spiel::algorithms::PolicyIteration(*game,-1,0,0.01);
	for(const auto& kv: solution){
		cerr << "State: " << endl 
			 << kv.first << endl
			 << "Value: " << kv.second << endl;
	}
	std::string initial_state = "...\n...\n...";
	std::string cross_win_state = "...\n...\n.ox";
	std::string naught_win_state = "x..\noo.\nxx.";
	SPIEL_CHECK_EQ(solution[initial_state], 0);
	SPIEL_CHECK_EQ(solution[cross_win_state], 1);
	SPIEL_CHECK_EQ(solution[naught_win_state], -1);
	return 0;
}
