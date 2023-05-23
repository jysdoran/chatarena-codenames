from chatarena.arena import Arena
from chatarena.agent import Player
from chatarena.backends import OpenAIChat

from codenames import Codenames


def run_codenames():
    with open("description.txt", "r") as f:
        environment_description = f.read()

    spymaster_description = "Your role is to give clues to the operative of your team. On your turn give only the clue and the number of words on the list that correspond to it (e.g. FRUIT 2)."
    operative_description = "Your role is to guess the words that correspond to the clue given by your spymaster. On your turn give only the words from the list that you want to guess (e.g. APPLE, ORANGE)."

    player1 = Player(
        name="Red Spymaster",
        backend=OpenAIChat(),
        role_desc="You are the spymaster of the red team. " + spymaster_description ,
        global_prompt=environment_description,
    )
    player2 = Player(
        name="Red Operative",
        backend=OpenAIChat(),
        role_desc="You are the operative of the red team. " + operative_description,
        global_prompt=environment_description,
    )
    player3 = Player(
        name="Blue Spymaster",
        backend=OpenAIChat(),
        role_desc="You are the spymaster of the blue team. " + spymaster_description,
        global_prompt=environment_description,
    )
    player4 = Player(
        name="Blue Operative",
        backend=OpenAIChat(),
        role_desc="You are the operative of the blue team. " + operative_description,
        global_prompt=environment_description,
    )

    env = Codenames()
    arena = Arena(
        players=[player1, player2, player3, player4],
        environment=env,
        global_prompt=environment_description,
    )
    # Run the game for 10 steps
    try:
        arena.run(num_steps=40)
    finally:
        arena.save_history(path="history.csv")


def config_example():
    # Tic-tac-toe example
    # Arena.from_config("tic-tac-toe.json").launch_cli()

    # # Rock-paper-scissors example
    arena = Arena.from_config("rock-paper-scissors.json")
    arena.run(num_steps=5)
    arena.save_history(path="history.json")


def human_example():
    pass


def main():
    run_codenames()
    # run_interview()


if __name__ == "__main__":
    main()
