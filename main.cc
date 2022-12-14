#include <memory>
#include <random>
#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
using namespace std;
using namespace open_spiel;

void PrintLegalActions(const State& state,Player player,vector<Action>& movelist){
	cout << "Legal moves for player " << player << ":" << endl;
	for(Action action: movelist){
		cout << " " << state.ActionToString(player,action) << endl;
	}
}

int main(int argc,char** argv){
	if(argc<-2) return -1;
	
	// Print out registered games
	cout << "Registered games: " << std::endl;
	vector<string> names = RegisteredGames();
	for(const string& name:names){
		cout << name << endl;
	}
	mt19937 rng(time(0));

	// Load Game
	cout << "Loading Game..\n" << endl;
	shared_ptr<const Game>game = LoadGame(argv[1]);
	if(!game) return -1;

	cout << "Starting Game..." << endl;
	unique_ptr<State> state = game->NewInitialState();
	cout << "Initial State: " << endl;
	cout << "State: " << endl << state->ToString() << endl;

	while(!state->IsTerminal()){
		cout << "Player " << state->CurrentPlayer() << endl;
		if(state->IsChanceNode()){
			vector<pair<Action,double>> outcomes = state->ChanceOutcomes();
			Action action = SampleAction(outcomes,rng).first;
			cout << "Sampled Outcome: " << state->ActionToString(kChancePlayerId,action) << endl;
			state->ApplyAction(action);
		}else if(state->IsSimultaneousNode()){
			vector<Action> joint_action;
			vector<float> infostate(game->InformationStateTensorSize());
			for(auto player=Player{0};player<game->NumPlayers();++player){
				vector<Action> actions = state->LegalActions(player);
				PrintLegalActions(*state,player,actions);
				uniform_int_distribution<> dis(0,actions.size()-1);
				Action action=actions[dis(rng)];
				joint_action.push_back(action);
				cout << "Player " << player << " chose " << state->ActionToString(player,action) << endl;
			}
			state->ApplyActions(joint_action);
		}else{
			auto player=state->CurrentPlayer();
			vector<Action> actions = state->LegalActions();
			PrintLegalActions(*state,player,actions);
			uniform_int_distribution<> dis(0,actions.size()-1);
			auto action = actions[dis(rng)];
			cout << "Chose action: " << state->ActionToString(player,action) << endl;
			state->ApplyAction(action);
		}
		cout << "State: " << endl << state->ToString() << endl;
	}
	auto returns = state->Returns();
	for(auto p=Player{0}; p<game->NumPlayers(); p++){
		cout << "Final return to player " << p << " is " << returns[p] << endl;
	}
}
