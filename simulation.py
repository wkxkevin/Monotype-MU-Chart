import numpy as np

chart = np.load("matchup_1630.npy")
chart = (chart * 100).astype(np.int32)
print(chart.shape)
player_freq = np.full(18, 100, dtype=np.int32)
types = np.array(
    [
        "normal",
        "fighting",
        "flying",
        "poison",
        "ground",
        "rock",
        "bug",
        "ghost",
        "steel",
        "fire",
        "water",
        "grass",
        "electric",
        "psychic",
        "ice",
        "dragon",
        "dark",
        "fairy",
    ]
)


def softmax(x, temperature=0.2):
    e = np.exp((x - x.max()) / temperature)
    return e / e.sum()


for generation in range(10000):
    # new matchup chart for this gen
    curr_mu_chart = np.random.randint(chart[..., 0], chart[..., 1] + 1)
    # print(curr_mu_chart)

    # declare list of players and array to track wins and losses
    players = np.repeat(np.arange(18), player_freq)
    player_wins = np.zeros(len(players), dtype=np.int32)
    player_losses = np.zeros(len(players), dtype=np.int32)

    # matchups that will be played
    player_1 = np.random.randint(0, len(players), size=100_000, dtype=np.int32)
    player_2 = np.random.randint(0, len(players), size=100_000, dtype=np.int32)

    # computing the win loss record
    odds = np.random.randint(0, 10001, size=100_000, dtype=np.int32)
    winrates = curr_mu_chart[players[player_1], players[player_2]]
    wins = odds < winrates
    player_wins = np.bincount(player_1[wins], minlength=len(players)) + np.bincount(
        player_2[~wins], minlength=len(players)
    )
    player_losses = np.bincount(player_2[wins], minlength=len(players)) + np.bincount(
        player_1[~wins], minlength=len(players)
    )

    # computing win percent of each type
    type_wins = np.bincount(players[player_1[wins]], minlength=18) + np.bincount(
        players[player_2[~wins]], minlength=18
    )
    type_games = np.bincount(players[player_1], minlength=18) + np.bincount(
        players[player_2], minlength=18
    )
    type_winrate = np.divide(
        type_wins, type_games, out=np.full(18, 0.5), where=type_games != 0
    )

    # compute proability of switching to a new type
    epsilon = 0.05
    pick_probs = (1 - epsilon) * softmax(type_winrate) + epsilon * (1 / 18)
    # print(pick_probs)

    # find players with low winrates
    total = player_wins + player_losses
    winrate = np.divide(
        player_wins, total, out=np.full(len(players), 0.5), where=total != 0
    )
    switching = np.where(winrate < 0.30)[0]
    choose_switch = np.random.randint(0, 100, size=len(switching), dtype=np.int32)

    # have players switch types
    bad_types = np.where(player_freq < 10)[0]
    if len(bad_types) == 0:
        bad_types = np.argsort(player_freq)[:1]
    is_contrarian = np.random.random(len(switching)) < 0.05

    new_types = np.random.choice(
        18, size=len(switching), p=type_winrate / type_winrate.sum()
    )
    contrarian_picks = np.random.choice(bad_types, size=len(switching))
    new_types[is_contrarian] = contrarian_picks[is_contrarian]

    # update player per type
    players[switching] = new_types
    player_freq = np.bincount(players, minlength=18)
    if generation % 100 == 0:
        # print(player_freq)
        print(",".join(map(str, player_freq)))
