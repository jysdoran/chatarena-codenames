"""Python classes for ranking different language models based on their chatarena outcomes"""
import random
from typing import Dict, Generator, List, Optional, Tuple, Type
from itertools import combinations, permutations


from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import IntelligenceBackend
from chatarena.environments.base import Environment, TimeStep
from chatarena.message import Message


from codenames import Codenames


def arena_run_generator(arena, max_steps) -> Generator[TimeStep, None, None]:
    """
    run the game for num_steps while yielding timesteps
    """
    while max_steps > 0:
        max_steps -= 1
        timestep = arena.step()
        yield timestep
        if timestep.terminal:
            return


class CustomProgramming(IntelligenceBackend):
    stateful = False
    type_name = "custom-programming"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def query(
        self,
        agent_name: str,
        role_desc: str,
        history_messages: List[Message],
        global_prompt: str = None,
        request_msg: Message = None,
        *args,
        **kwargs,
    ) -> str:
        return str(random.randrange(*self.kwargs["range"]))


class BiggerNumber(Environment):
    """Chat environment where the goal is to choose a bigger number than the opponent"""

    type_name = "bigger-number"

    def __init__(self, player_names, **kwargs):
        super().__init__(player_names=player_names, **kwargs)

        # Game states
        self._current_turn = 0
        self._current_player_idx = 0
        self._number_buffer = []

        self._initialized = False
        self.reset()  # To initialize the game (select topic, code, chameleon)

    def get_next_player(self) -> str:
        """
        get the next player
        """
        return self.player_names[self._current_player_idx]

    def reset(self):
        """
        sample topic, code and chameleon code
        """

        self._current_turn = 1
        self._current_player_idx = 0

        self._initialized = True
        init_timestep = TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

        return init_timestep

    def get_observation(self, player_name=None) -> List[Message]:
        """
        get observation for the player
        """
        return [
            Message(p, str(n), self._current_turn)
            for n, p in zip(self._number_buffer, self.player_names)
        ]

    def check_action(self, action: str, player_name: str) -> bool:
        """
        check whether the action is valid
        """
        # Check whether the player is the next player
        if player_name != self.get_next_player():
            return False

        # Check whether the action is a number
        n = int(action)

        return True

    def is_terminal(self) -> bool:
        """
        check if the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        return self._current_turn > 10

    def step(self, player_name: str, action: str) -> TimeStep:
        """Take a step in the game."""
        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()}'s turn."

        rewards = self.get_zero_rewards()

        # Submit number to the buffer
        n = int(action)
        self._number_buffer.append(n)

        timestep = TimeStep(
            observation=self.get_observation(),
            reward=rewards,
            terminal=self.is_terminal(),
        )

        if self._current_player_idx == len(self.player_names) - 1:
            # If it is the last player, then the turn is over
            self._current_turn += 1

            numbers = sorted(zip(self._number_buffer, self.player_names), reverse=True)
            biggest = numbers[0][0]
            winners = [name for n, name in numbers if n == biggest]
            for w in winners:
                rewards[w] = 1 / len(winners)
            self._number_buffer.clear()

        # cycle through players
        self._current_player_idx = (self._current_player_idx + 1) % len(
            self.player_names
        )

        return timestep


import numpy as np


class PlayerRanker:
    def __init__(
        self,
        env_type: Type[Environment],
        env_kwargs: Optional[Dict] = None,
        max_steps: Optional[int] = 100,
    ):
        self.env_type = env_type
        self.env_kwargs = env_kwargs or {}
        self.max_steps = max_steps or float("inf")

    def agg_rewards(self, rewards: List[float]) -> float:
        # Override with discounting / mean here
        return sum(rewards)

    def run_game(self, players: List[Player]) -> List[float]:
        # Create the arena
        env = self.env_type(player_names=[p.name for p in players], **self.env_kwargs)
        arena = Arena(players, env)

        # Run the games and record the rewards
        arena.reset()

        p_rewards = {p.name: [] for p in players}
        for timestep in arena_run_generator(arena, self.max_steps):
            for p in players:
                p_rewards[p.name].append(timestep.reward[p.name])

        return [self.agg_rewards(p_rewards[p.name]) for p in players]

    def matchup_table(
        self, players: List[Player], players_per_game, repeats: int = 5
    ) -> np.ndarray:
        """Compute the mean game result for every combination of players
        Assumes that the player positions are not important
        Starts to get a bit wasteful with memory when more than 2 players per game"""
        n = len(players)
        results = np.zeros((n,) * players_per_game)
        # player_combos = np.triu_indices(n,)
        player_combos = combinations(range(n), players_per_game)

        for player_idxs in player_combos:
            player_rewards = np.zeros(players_per_game)
            game_players = [players[i] for i in player_idxs]
            for _ in range(repeats):
                # Run the game
                game_result = self.run_game(game_players)
                player_rewards += np.array(game_result)
            # Normalize
            player_rewards /= repeats
            for perm in permutations(range(players_per_game)):
                results[tuple(player_idxs[i] for i in perm)] = player_rewards[perm[0]]

        return results

    def rank_players(
        self, players: List[Player], players_per_game
    ) -> List[Tuple[Player, float]]:
        """Rank the players by running the matchup table"""
        results = self.matchup_table(players, players_per_game=players_per_game)

        # Compute the mean score for each player
        player_scores = results.mean(axis=tuple(range(1, players_per_game)))

        # Sort the players by their score
        sorted_idxs = np.argsort(player_scores)

        return [(players[i], player_scores[i]) for i in sorted_idxs]


def run_bigger_number():
    player1 = Player(
        name="1-10", backend=CustomProgramming(range=(0, 10)), role_desc=""
    )
    player2 = Player(
        name="5-15", backend=CustomProgramming(range=(5, 15)), role_desc=""
    )
    players = [player1, player2]

    # Kinda think it should get these names from the arena
    env = BiggerNumber(player_names=[player.name for player in players])
    arena = Arena(
        players=players,
        environment=env,
    )
    # Run the game for 10 steps
    for timestep in arena_run_generator(arena, 10):
        print(timestep)


def rank_bigger_number():
    player1 = Player(
        name="1-10", backend=CustomProgramming(range=(0, 10)), role_desc=""
    )
    player2 = Player(
        name="5-15", backend=CustomProgramming(range=(5, 15)), role_desc=""
    )
    player3 = Player(
        name="10-20", backend=CustomProgramming(range=(10, 20)), role_desc=""
    )
    players = [player1, player2, player3]

    ranker = PlayerRanker(BiggerNumber)
    ranked_players = ranker.rank_players(players, players_per_game=2)
    for player, score in ranked_players:
        print(f"{player.name}: {score}")

    table = ranker.matchup_table(players, players_per_game=2, repeats=500)
    print(table)


if __name__ == "__main__":
    rank_bigger_number()
