#ifndef OPEN_SPIEL_ALGORITHMS_MINMAX_H_
#define OPEN_SPIEL_ALGORITHMS_MINMAX_H_
#include <memory>
#include <utility>
#include "open_spiel/spiel.h"

namespace open_spiel {
namespace algorithms {

std::pair<double, Action> AlphaBetaSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player);

std::pair<double, Action> ExpectiminimaxSearch(
    const Game& game, const State* state,
    std::function<double(const State&)> value_function, int depth_limit,
    Player maximizing_player);

}
} 
#endif
