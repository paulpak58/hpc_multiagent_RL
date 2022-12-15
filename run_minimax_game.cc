#include <memory>
#include "open_spiel/algorithms/minimax.h"
#include "open_spiel/games/breakthrough.h"
#include "open_spiel/games/pig.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
inline constexpr int kSearchDepth = 2;
inline constexpr int kSearchDepthPig = 10;
inline constexpr int kWinscorePig = 30;
inline constexpr int kDiceoutcomesPig = 2;
inline constexpr int kSeed = 726345721;
using namespace std;
using namespace open_spiel::algorithms;

namespace open_spiel {
namespace {
int BlackPieceAdvantage(const State& state) {
  const auto& bstate = down_cast<const breakthrough::BreakthroughState&>(state);
  return bstate.pieces(breakthrough::kBlackPlayerId) -
         bstate.pieces(breakthrough::kWhitePlayerId);
}

int FirstPlayerAdvantage(const State& state){
	const auto& pstate = down_cast<const pig::PigState&>(state);
	return pstate.score(0)-pstate.score(1);
}

void PlayBreakthrough() {
	shared_ptr<const Game> game = LoadGame("breakthrough",{{"rows",GameParameter(6)},{"columns",GameParameter(6)}});
	unique_ptr<State> state = game->NewInitialState();
	while(!state->IsTerminal()){
		cout << endl << state->ToString() << endl;
		Player player = state->CurrentPlayer();
		pair<double,Action> value_action = AlphaBetaSearch(*game,state.get(),[player](const State& state) {return (player==breakthrough::kBlackPlayerId ? BlackPieceAdvantage(state) : -BlackPieceAdvantage(state)); },kSearchDepth,player);
		cout << endl << "Player " << player << " choosing action " << state->ActionToString(player,value_action.second) << " wit heuristic value (to black) " << value_action.first << endl;
		state->ApplyAction(value_action.second);
  }
  	cout << "Terminal state: " << endl;
  	cout << state->ToString() << endl;
}

void PlayPig(mt19937& rng) {
	shared_ptr<const Game> game = LoadGame("pig", {{"winscore", GameParameter(kWinscorePig)},{"diceoutcomes", GameParameter(kDiceoutcomesPig)}});
	unique_ptr<State> state = game->NewInitialState();
	while(!state->IsTerminal()){
		cout << endl << state->ToString() << endl;
		Player player = state->CurrentPlayer();
		if(state->IsChanceNode()){
			ActionsAndProbs outcomes = state->ChanceOutcomes();
			Action action = open_spiel::SampleAction(outcomes,rng).first;
			cout << "Sampled action: " << state->ActionToString(player,action) << endl;
			state->ApplyAction(action);
		}else{
			pair<double,Action> value_action = ExpectiminimaxSearch(*game,state.get(),[player](const State& state){return (player==Player{0} ? FirstPlayerAdvantage(state) : -FirstPlayerAdvantage(state));},kSearchDepthPig,player);
			cout << endl << "Player " << player << " choosing action " << state->ActionToString(player,value_action.second) << " with heuristic value " << value_action.first << endl;
			state->ApplyAction(value_action.second);
		}
	}
  std::cout << "Terminal state: " << std::endl;
  std::cout << state->ToString() << std::endl;
}

}  // namespace
}  // namespace open_spiel
int main(int argc, char **argv) {
  mt19937 rng(kSeed);
  open_spiel::PlayBreakthrough();
  open_spiel::PlayPig(rng);
}
