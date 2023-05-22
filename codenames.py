from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Dict, Union, Optional, Tuple
import random
import re

from chatarena.environments import Environment, TimeStep
from chatarena.message import Message, MessagePool
from chatarena.agent import SIGNAL_END_OF_CONVERSATION
from chatarena.config import EnvironmentConfig

with open("words.txt", "r") as f:
    DEFAULT_WORD_LIST = [line.strip() for line in f.readlines()]

with open("codenames_description.txt", "r") as f:
    CODENAMES_DESCRIPTION = f.read()


class Codenames(Environment):
    """Chat environment for two language agents to play Codenames.
    Moderated by another language agent."""

    type_name = "codenames"
    team_names = ["Red", "Blue"]

    class Phase(Enum):
        GIVE_CLUE = 0
        GUESS_CLUE = 1

    @dataclass
    class Clue:
        word: str
        count: int

        def __str__(self) -> str:
            return f"{self.word.upper()} {self.count}"

    class Team:
        def __init__(self, name, idx):
            self.name = name
            self.idx = idx
            self.spymaster = f"{name} Spymaster"
            self.operative = f"{name} Operative"

        @cached_property
        def members(self):
            return [self.spymaster, self.operative]

        def __hash__(self) -> int:
            return hash(self.idx)

    class GameBoard:
        @dataclass
        class Card:
            class CardType(Enum):
                # Could/Should be refactored to be more general (use environment.team_names)
                RED = 0
                BLUE = 1
                ASSASSIN = 2
                BYSTANDER = 3

                def __str__(self):
                    return self.name.capitalize()

            word: str
            type: CardType
            revealed: bool = False

            def __str__(self):
                return f"{self.word}"

        def __init__(self, board_size: int = 25, cards_each: int = 9):
            self.board_size = board_size
            self.word_list = DEFAULT_WORD_LIST
            self._board: Dict[str, Codenames.GameBoard.Card] = {}
            self.cards_each = cards_each
            self.reset()

        def reset(self):
            deck = self.word_list.copy()
            random.shuffle(deck)
            board = []
            board.extend(
                [
                    self.Card(word=word, type=self.Card.CardType.RED)
                    for word in deck[: self.cards_each]
                ]
            )
            board.extend(
                [
                    self.Card(word=word, type=self.Card.CardType.BLUE)
                    for word in deck[self.cards_each : 2 * self.cards_each - 1]
                ]
            )
            board.extend(
                [
                    self.Card(word=word, type=self.Card.CardType.BYSTANDER)
                    for word in deck[2 * self.cards_each - 1 : self.board_size - 1]
                ]
            )
            board.append(
                self.Card(
                    word=deck[self.board_size - 1], type=self.Card.CardType.ASSASSIN
                )
            )
            # This will be the display order so it's important to shuffle
            random.shuffle(board)

            self._board = {card.word: card for card in board}
            del deck

        def reveal_card(self, word: str):
            card = self._board.get(word)
            if card is None:
                raise ValueError(f"Card '{word}' not found.")
            elif card.revealed:
                raise ValueError(f"Card '{word}' has already been revealed.")
            card.revealed = True
            return card

        def display_board(self, spymaster: bool = False):
            """Return a string representation of the board."""
            board = []
            for i, card in enumerate(self._board.values()):
                # card_type = card.type.name if card.revealed or spymaster else "???"
                # board.append(f"{card_type}: {card.word}")
                if spymaster:
                    board.append(f"{card.type}: {card.word}")
                else:
                    board.append(card.word)

            return "\n".join(board)

    def __init__(self, word_list: Optional[List[str]] = None, **kwargs):
        self.teams = [
            Codenames.Team(name=name, idx=i) for i, name in enumerate(team_names)
        ]
        player_names = self.teams[0].members + self.teams[1].members
        super().__init__(player_names=player_names, word_list=word_list, **kwargs)

        # The "state" of the environment is maintained by the message pool
        self.message_pool = MessagePool()

        # Randomly sample a topic, code and chameleon player
        self.topic = None
        self.cards_each = 9
        self.remaining_cards = {
            self.teams[0]: self.cards_each,
            self.teams[1]: self.cards_each - 1,
        }

        word_list = DEFAULT_WORD_LIST if word_list is None else word_list
        self.board = self.GameBoard(board_size=25, cards_each=self.cards_each)

        # Game states
        self._current_turn = 0
        self._current_team_idx = 0
        self._remaining_guesses = 0
        self._clue_history = []
        self._current_phase = Codenames.Phase.GIVE_CLUE

        self._initialized = False
        self.reset()  # To initialize the game (select topic, code, chameleon)

    def get_next_player(self) -> str:
        """
        get the next player
        """
        if self._current_phase == Codenames.Phase.GIVE_CLUE:
            return self.teams[self._current_team_idx].spymaster
        else:
            # Phase is GUESS_CLUE
            return self.teams[self._current_team_idx].operative

    def reset(self):
        """
        sample topic, code and chameleon code
        """
        self.board.reset()
        self.remaining_cards = {
            self.teams[0]: self.cards_each,
            self.teams[1]: self.cards_each - 1,
        }

        self._remaining_guesses = 0
        self._current_turn = 0
        self._current_team_idx = 0
        self._current_phase = Codenames.Phase.GIVE_CLUE
        self._clue_history = []

        self.message_pool.reset()

        self._moderator_speak("Welcome to Codenames!")
        self._moderator_speak(CODENAMES_DESCRIPTION)
        self._moderator_speak(f"Now the game starts!")
        # Inform spymasters of the board
        for team in self.teams:
            self._moderator_speak(
                f"You are the {team.name} team spymaster.", visible_to=team.spymaster
            )
        self._moderator_speak(
            f"Here is your key card:\n{self.board.display_board(spymaster=True)}",
            visible_to=[team.spymaster for team in self.teams],
        )

        # inform operatives
        for team in self.teams:
            self._moderator_speak(
                f"You are the {team.name} team operative.", visible_to=team.operative
            )
        self._moderator_speak(
            f"Here is the list of codenames:\n{self.board.display_board(spymaster=False)}",
            visible_to=[team.operative for team in self.teams],
        )

        self._moderator_speak(
            f"Now the {self.teams[0].name} team spymaster gives their first clue."
        )

        self._current_turn = 1

        self._initialized = True
        init_timestep = TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

        return init_timestep

    def print(self):
        self.message_pool.print()

    def get_observation(self, player_name=None) -> List[Message]:
        """
        get observation for the player
        """
        if player_name is None:
            return self.message_pool.get_all_messages()
        else:
            return self.message_pool.get_visible_messages(
                player_name, turn=self._current_turn
            )

    def _parse_clue(self, action) -> Optional[Clue]:
        """
        Check whether the clue given by the spymaster is valid
        """
        # Get the word enclosed by quote marks with regex
        match = re.search(r"(\w+)(:)? (\d+)", action)
        # Maybe findall is better
        if match:
            word, number = match.groups(1)
            word = str(word)
            number = int(number)
        else:
            return None

        if word.upper() in self.board.word_list:
            "The word in the word list (illegal) or the number is not a digit"
            return None

        return Codenames.Clue(word, number)

    def _parse_guess(self, action) -> Optional[str]:
        # Get the word enclosed by quote marks or is capitalized
        # Requires at least two capital letters to avoid false positives "I"
        match = re.search(r"([A-Z][A-Z]+)|(\"\w+\")", action)
        # Maybe findall is better
        if match:
            for group in match.groups():
                if group is not None:
                    return group.strip('"').upper()

        return None

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        """
        moderator say something
        """
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def is_terminal(self) -> bool:
        """
        check if the conversation is over
        """
        # If the last message is the signal, then the conversation is over
        return (
            not self.message_pool.last_message
            or self.message_pool.last_message.content == SIGNAL_END_OF_CONVERSATION
        )

    def _game_state_potential(self) -> Dict[Team, float]:
        # Calculate the potential of the current game state (normalised relative words left)
        total_remaining = sum(self.remaining_cards.values())
        normalised_remaining = {
            team: self.remaining_cards[team] / total_remaining for team in self.teams
        }
        # Could combine into one line, but this is more readable
        potential = {team: 1 - 2 * x for team, x in normalised_remaining.items()}
        return potential

    def step(self, player_name: str, action: str) -> TimeStep:
        """Take a step in the game."""
        # If not initialized, reset the environment
        if not self._initialized:
            self.reset()

        assert (
            player_name == self.get_next_player()
        ), f"Wrong player! It is {self.get_next_player()}'s turn."

        current_team = self.teams[self._current_team_idx]
        rewards = self.get_zero_rewards()
        terminal = False

        if self._current_phase == Codenames.Phase.GIVE_CLUE:
            # Parse the clue
            clue = self._parse_clue(action)
            if clue is None:
                raise ValueError(f"No clue detected in action: {action}")

            message = Message(
                agent_name=player_name,
                content=f"The clue for the {current_team.name} team is: {str(clue)}",
                turn=self._current_turn,
            )

            self.message_pool.append_message(message)
            self._clue_history.append(clue)
            self._current_phase = Codenames.Phase.GUESS_CLUE
            self._remaining_guesses = clue.count + 1

        elif self._current_phase == Codenames.Phase.GUESS_CLUE:
            # Parse the guess
            guess = self._parse_guess(action)
            if guess is None:
                raise ValueError(f"No guess detected in action: {action}")

            card = self.board.reveal_card(guess)

            message = Message(
                agent_name=player_name,
                content=f"The guess for the {current_team.name} team is: {str(card)}",
                turn=self._current_turn,
            )

            self.message_pool.append_message(message)

            if card.type == Codenames.GameBoard.Card.CardType.ASSASSIN:
                self._moderator_speak(
                    f"{player_name} guessed the assassin! {current_team.name} team lost!"
                )
                for team in self.teams:
                    for player in team.members:
                        rewards[player] = -1 if team == current_team else 1
                terminal = True
            else:
                # Card is red or blue
                self._moderator_speak(
                    f"The card {player_name} guessed was a {card.type} card!"
                )

                if card.type.value == self._current_team_idx:
                    self._remaining_guesses -= 1
                else:
                    self._remaining_guesses = 0

                previous_potential = self._game_state_potential()
                self.remaining_cards[self.teams[card.type.value]] -= 1
                current_potential = self._game_state_potential()

                if min(self.remaining_cards.values()) <= 0:
                    winning_team = min(
                        self.remaining_cards.keys(),
                        key=self.remaining_cards.__getitem__,
                    )
                    self._moderator_speak(
                        f"{winning_team.name} team has no more cards left! {winning_team.name} team won!"
                    )
                    for team in self.teams:
                        for player in team.members:
                            rewards[player] = 1 if team == winning_team else -1
                    terminal = True
                else:
                    for team in self.teams:
                        potential_diff = (
                            current_potential[team] - previous_potential[team]
                        )
                        for player in team.members:
                            rewards[player] = potential_diff

                    if self._remaining_guesses <= 0:
                        self._moderator_speak(
                            f"{current_team.name} team has no more guesses left! It is now the {self.teams[1 - self._current_team_idx].name} team's turn."
                        )
                        self._current_team_idx = 1 - self._current_team_idx
                        self._current_phase = Codenames.Phase.GIVE_CLUE
                        self._current_turn += 1

        else:
            raise ValueError(f"Unknown phase: {self._current_phase}")

        timestep = TimeStep(
            observation=self.get_observation(),
            reward=rewards,
            terminal=terminal,
        )

        # Check if the player signals the end of the conversation
        if self.is_terminal():
            timestep.terminal = True

        return timestep
