"""Plays benchmark games; runs in the worker processes of `__main__`.

A module of its own because worker processes (forkserver on Python 3.14)
can't import functions defined in `__main__`.
"""

COLORS = ["RED", "BLUE", "ORANGE", "WHITE"]


def play_games(checkpoint: str, opponent: str, seats: int, seeds: list[int]):
    """Plays one game per seed; returns (winner_is_ours, our VPs, their VPs,
    turns) per game. Runs in a worker process."""
    import torch

    torch.set_num_threads(1)
    import catanatron.players  # noqa: F401  (registers the built-in bots)
    from catanatron import Color, Game
    from catanatron.registry import REGISTRY

    from bench.catanatron.player import RcatanPlayer
    from train.ppo import load_actor

    actor = load_actor(checkpoint, device="cpu")
    colors = [Color[c] for c in COLORS]
    results = []
    for seed in seeds:
        ours = [RcatanPlayer(c, actor) for c in colors[:seats]]
        theirs = [REGISTRY.build(opponent, c) for c in colors[seats:]]
        game = Game(ours + theirs, seed=seed)
        winner = game.play()
        ps = game.state.player_state

        def vp(color):
            index = game.state.color_to_index[color]
            return ps[f"P{index}_ACTUAL_VICTORY_POINTS"]

        results.append(
            (
                None if winner is None else winner in colors[:seats],
                [vp(c) for c in colors[:seats]],
                [vp(c) for c in colors[seats:]],
                game.state.num_turns,
            )
        )
    return results
