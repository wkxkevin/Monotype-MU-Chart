import os
import re

import numpy as np
import requests
from scipy import stats
from scipy.optimize import linprog


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - margin) * 100, (center + margin) * 100


def solve_nash(best_chart):
    A = np.asarray(best_chart, dtype=float)
    n, m = A.shape

    # Row player's LP: maximize guaranteed win rate
    # Variables: p[0:n], v
    c = np.zeros(n + 1)
    c[-1] = -1  # maximize v

    # A.T @ p >= v  =>  -A.T @ p + v <= 0
    A_ub = np.hstack([-A.T, np.ones((m, 1))])

    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1

    b_eq = [1]
    b_ub = np.zeros(m)

    bounds = [(0, None)] * n + [(None, None)]

    row_result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    # Column player's LP: minimize maximum row payoff
    # Variables: q[0:m], v
    c = np.zeros(m + 1)
    c[-1] = 1

    A_ub = np.hstack([A, -np.ones((n, 1))])

    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1

    col_result = linprog(
        c,
        A_ub=A_ub,
        b_ub=np.zeros(n),
        A_eq=A_eq,
        b_eq=[1],
        bounds=[(0, None)] * m + [(None, None)],
        method="highs",
    )

    if not row_result.success or not col_result.success:
        raise ValueError("Optimization failed")

    p = row_result.x[:n]
    q = col_result.x[:m]
    value = row_result.x[-1]

    return p, q, value


def solve_banded_row(A, usage, band=0.5):
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    lo = usage * (1 - band)
    hi = usage * (1 + band)

    # Variables: p[0:n], u[0:m], w[0:m], t
    N = n + 2 * m + 1
    c = np.zeros(N)
    c[n : n + m] = -lo
    c[n + m : n + 2 * m] = hi
    c[-1] = -1

    # -A.T @ p + u - w + t <= 0
    A_ub = np.zeros((m, N))
    A_ub[:, :n] = -A.T
    A_ub[:, n : n + m] = np.eye(m)
    A_ub[:, n + m : n + 2 * m] = -np.eye(m)
    A_ub[:, -1] = 1

    A_eq = np.zeros((1, N))
    A_eq[0, :n] = 1  # sum(p) = 1

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=np.zeros(m),
        A_eq=A_eq,
        b_eq=[1],
        bounds=[(0, None)] * (n + 2 * m) + [(None, None)],
        method="highs",
    )
    if not res.success:
        raise ValueError("Banded optimization failed")
    return res.x[:n], -res.fun


def sample_matrices(counts, n_samples, seed=0):
    """Sample zero-sum win-rate matrices from Beta posteriors of pooled counts."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(18, 1)
    samples = []
    for _ in range(n_samples):
        A = np.full((18, 18), 0.5)
        for r, c in zip(*iu):
            k_rc, k_cr = counts[r, c], counts[c, r]
            s = rng.beta(k_rc + 1, k_cr + 1) if k_rc + k_cr > 0 else 0.5
            A[r, c] = s
            A[c, r] = 1 - s
        samples.append(A)
    return samples


def format_mix(title, names, probs, cap=10, line_width=80, threshold=1e-4):
    """Title, then name row and percent row, wrapped to line_width."""
    keep = [(n, p) for n, p in zip(names, probs) if p > threshold]
    per_line = max(1, line_width // (cap + 1))
    out = [title]
    for start in range(0, len(keep), per_line):
        chunk = keep[start : start + per_line]
        out.append("".join(n[:cap].ljust(cap) + " " for n, _ in chunk).rstrip())
        out.append("".join(f"{p:.2%}".ljust(cap) + " " for _, p in chunk).rstrip())
        out.append("")
    return "\n".join(out)


dates = [
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
    "2026-07",
    "2026-08",
    "2026-09",
    "2026-10",
]
charts = [
    "gen9monotype-matchup_chart-0.txt",
    "gen9monotype-matchup_chart-1500.txt",
    "gen9monotype-matchup_chart-1630.txt",
    "gen9monotype-matchup_chart-1760.txt",
    "gen9nationaldexmonotype-matchup_chart-0.txt",
    "gen9nationaldexmonotype-matchup_chart-1500.txt",
    "gen9nationaldexmonotype-matchup_chart-1630.txt",
    "gen9nationaldexmonotype-matchup_chart-1760.txt",
]
types = [
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
gliscor_ban = True  # True: use only 2026-08 onward
BAND = 0  # usage band for the ladder-usage experiments
N_SAMPLES = 1000  # matrices sampled for the CI experiments
CELL_CAP = 7  # max characters per type name in the output
LINE_WIDTH = 200  # wrap output lines at this width

START_IDX = dates.index("2026-08") if gliscor_ban else 0

print("Usage")
monthly_games = np.zeros((8, len(dates)))
total_games = np.zeros(8)
monthly_usage = np.zeros((8, len(dates), 18))
for i, chart in enumerate(charts):
    for j, date in enumerate(dates):
        response = requests.get(
            "https://www.smogon.com/stats/"
            + date
            + "/metagame/"
            + chart.replace("-matchup_chart", ""),
            stream=True,
        )
        response2 = requests.get(
            "https://www.smogon.com/stats/"
            + date
            + "/"
            + chart.replace("-matchup_chart", ""),
            stream=True,
        )
        lines = response2.iter_lines(decode_unicode=True)
        monthly_games[i][j] = np.round(
            float(next(lines)[15:]) * float(next(lines)[18:])
        )
        for line in response.iter_lines(decode_unicode=True):
            if len(line) < 5:
                break
            if line[4] != "t" and line[:4] == "mono":
                match = re.match(r"^(.*?)\.+(\d{1,2}\.\d{5})%$", line[4:])
                if match:
                    monthly_usage[i][j][types.index(match.group(1))] = np.round(
                        float(match.group(2)) * monthly_games[i][j]
                    )
monthly_usage = monthly_usage.astype(int)
monthly_games = monthly_games.astype(int)
total_games = np.sum(monthly_games, axis=1)

print("Aggregate Fetch")
aggregate_charts = np.zeros((8, 18, 18))
all_months = np.zeros((8, 18, 18))  # counts summed over every month
post_ban = np.zeros((8, 18, 18))  # counts summed over months from START_IDX on
for d, date in enumerate(dates):
    os.makedirs(date, exist_ok=True)
    print(date)
    for i, chart in enumerate(charts):
        response = requests.get(
            "https://www.smogon.com/stats/" + date + "/monotype/matchupcharts/" + chart
        )
        with open(date + "/" + chart, "w") as file:
            mod_chart = [
                line[10:-1]
                for line in response.text.splitlines()[4:57]
                if "+" not in line
            ]
            mod_chart = [mod_chart[i] for i in range(len(mod_chart)) if i % 2 == 0]
            arr = np.array(
                [
                    [float(cell.strip()) for cell in line.split("|")]
                    for line in mod_chart
                ]
            )
            all_months[i] += arr
            if d >= START_IDX:
                post_ban[i] += arr
            file.write("\n".join(mod_chart))

affected = [types.index("flying"), types.index("ground")]
mask = np.zeros((18, 18), dtype=bool)
mask[affected, :] = True
mask[:, affected] = True  # any pair involving flying or ground
aggregate_charts = np.where(mask, post_ban, all_months)

os.makedirs("aggregate", exist_ok=True)
print("Aggregate")
col_w = 9
sep = "+" + (("-" * col_w + "+") * 19)
best_chart = np.zeros((18, 18))
for i in range(8):
    chart = aggregate_charts[i]
    total = chart + chart.T
    percent = (
        np.divide(chart, total, out=np.full_like(chart, 50.0), where=total != 0) * 100
    )
    if i == 2:
        best_chart = percent / 100

    lines = [sep]
    header = "|" + f"{'':<{col_w}}|" + "".join(f"{t:<{col_w}}|" for t in types)
    lines.append(header)
    lines.append(sep)

    for r in range(18):
        raw_row = (
            "|"
            + f"{types[r]:<{col_w}}|"
            + "".join(f"{chart[r, c]:<9.2f}|" for c in range(18))
        )
        pct_row = (
            "|"
            + f"{'':<{col_w}}|"
            + "".join(f"{f'{percent[r, c]:.3f}%':<9}|" for c in range(18))
        )
        lines.append(raw_row)
        lines.append(pct_row)
        lines.append(sep)

    text = "\n".join(lines)

    with open("aggregate/" + charts[i], "w") as file:
        file.write(text)

os.makedirs("ConfInterv", exist_ok=True)
col_w = 9
sep = "+" + (("-" * col_w + "+") * 19)
print("Confidence Intereval")
for i in range(8):
    chart = aggregate_charts[i]
    total = chart + chart.T

    lines = [sep]
    header = "|" + f"{'':<{col_w}}|" + "".join(f"{t:<{col_w}}|" for t in types)
    lines.append(header)
    lines.append(sep)

    for r in range(18):
        ci_row = "|" + f"{types[r]:<{col_w}}|"
        ci_row2 = "|" + f"{'':<{col_w}}|"

        for c in range(18):
            n = total[r, c]
            k = chart[r, c]
            if n > 0:
                ci = wilson_ci(k, n)
                ci_row += f"{f'{ci[0]:.3f}%':<9}|"
                ci_row2 += f"{f'{ci[1]:.3f}%':<9}|"
            else:
                ci_row += f"{'N/A':<9}|"
                ci_row2 += f"{'N/A':<9}|"

        lines.append(ci_row)
        lines.append(ci_row2)
        lines.append(sep)

    with open("ConfInterv/" + charts[i], "w") as file:
        file.write("\n".join(lines))


def bucket(pct):
    if pct < 45:
        return -int((55 - pct) // 10)
    elif pct > 55:
        return int((pct - 45) // 10)
    else:
        return 0


def bucket_str(lo, hi):
    b_lo = bucket(lo)
    b_hi = bucket(hi)
    return f"{b_lo} | {b_hi}"


print("Simplified")
for i in range(8):
    chart = aggregate_charts[i]
    total = chart + chart.T

    lines = []
    header = "|" + f"{'':<{col_w}}|" + "".join(f"{t:<{col_w}}||" for t in types)
    lines.append(header)

    for r in range(18):
        ci_row = f"{types[r]:<{col_w}}|"

        for c in range(18):
            n = total[r, c]
            k = chart[r, c]
            if n > 0:
                ci = wilson_ci(k, n)
                val = bucket_str(ci[0], ci[1])
                ci_row += f"{val:<9}|"
            else:
                ci_row += f"{'N/A':<9}|"

        lines.append(ci_row)

    with open("ConfInterv/simple_" + charts[i], "w") as file:
        file.write("\n".join(lines))

print("Timeline")
cell_w = col_w  # width per tier within a cell
sep = "+" + (("-" * (cell_w * 3 + 2) + "+") * 18)  # 4 tiers + 3 dividers per cell
lines = []
header = "|" + "".join(f"{t:^{cell_w * 3 + 2}}|||" for t in types)
lines.append(header)

global_chart = np.zeros((18, 18))
for r in range(18):
    hi_row = f"{types[r]:<{cell_w}}|"
    lo_row = f"{'':<{cell_w}}|"
    av_row = f"{'':<{cell_w}}|"

    for c in range(18):
        tier_av = []
        tier_hi = []
        tier_lo = []
        for tier_idx in range(3):
            chart = aggregate_charts[tier_idx]
            total = chart + chart.T
            n = total[r, c]
            k = chart[r, c]
            if n > 0:
                ci = wilson_ci(k, n)
                if tier_idx == 2:
                    i = bucket(ci[0])
                    j = bucket(ci[1])
                    if i == 0 and j == 0 or r == c:
                        global_chart[r][c] = 0
                    elif i > 2 and j > 2:
                        global_chart[r][c] = 2
                    elif i < -2 and j < -2:
                        global_chart[r][c] = -2
                    elif i > 0 and j > 0:
                        global_chart[r][c] = 1
                    elif i < 0 and j < 0:
                        global_chart[r][c] = -1
                    else:
                        global_chart[r][c] = 0.5 * np.sign((i + j) / 2)

                tier_hi.append(f"{bucket(ci[1]):^{cell_w}}")
                tier_lo.append(f"{bucket(ci[0]):^{cell_w}}")
                tier_av.append(
                    f"{((bucket(ci[0]) + bucket(ci[1])) / 2.0):^{cell_w}.2f}"
                )
            else:
                tier_hi.append(f"{'N/A':^{cell_w}}")
                tier_lo.append(f"{'N/A':^{cell_w}}")
        hi_row += "|".join(tier_hi) + "|"
        lo_row += "|".join(tier_lo) + "|"
        av_row += "|".join(tier_av) + "|"
    lines.append(hi_row)
    lines.append(lo_row)
    lines.append(av_row)

for i in range(18):
    for j in range(18):
        if (global_chart[i][j] + global_chart[j][i]) != 0:
            if abs(global_chart[i][j]) != 0.5 and abs(global_chart[j][i]) == 0.5:
                global_chart[j][i] = -global_chart[i][j]
            elif abs(global_chart[j][i]) != 0.5 and abs(global_chart[i][j]) == 0.5:
                global_chart[i][j] = -global_chart[j][i]
            elif abs(global_chart[i][j]) > abs(global_chart[j][i]):
                global_chart[j][i] = -global_chart[i][j]
            else:
                global_chart[i][j] = -global_chart[j][i]

print("Nash")
usage_totals = monthly_usage[2, START_IDX:].sum(axis=0)
usage = usage_totals / usage_totals.sum()

# Experiment 1: uniform usage, point estimate
uniform_p, _, uniform_value = solve_nash(best_chart)

# Experiment 2: ladder usage, point estimate
trend_p, trend_value = solve_banded_row(best_chart, usage, band=BAND)

# Experiments 3 and 4: average over sampled matrices
samples = sample_matrices(aggregate_charts[2], N_SAMPLES)
ci_uniform_p = np.mean([solve_nash(A)[0] for A in samples], axis=0)
ci_trend_p = np.mean([solve_banded_row(A, usage, BAND)[0] for A in samples], axis=0)

results = [
    format_mix(
        f"1. Uniform usage (value {uniform_value:.4f})",
        types,
        uniform_p,
        CELL_CAP,
        LINE_WIDTH,
    ),
    format_mix(
        f"2. Ladder usage, band {BAND} (value {trend_value:.4f})",
        types,
        trend_p,
        CELL_CAP,
        LINE_WIDTH,
    ),
    format_mix(
        f"3. Confidence interval, uniform usage ({N_SAMPLES} samples)",
        types,
        ci_uniform_p,
        CELL_CAP,
        LINE_WIDTH,
    ),
    format_mix(
        f"4. Confidence interval, ladder usage, band {BAND} ({N_SAMPLES} samples)",
        types,
        ci_trend_p,
        CELL_CAP,
        LINE_WIDTH,
    ),
]

# ---- Tournament usage: number of times each type was brought ----
tour_data = [
    {
        "dragon": 34,
        "steel": 32,
        "water": 29,
        "flying": 25,
        "dark": 24,
        "fairy": 23,
        "ground": 20,
        "fighting": 19,
        "ghost": 13,
        "fire": 10,
        "bug": 6,
        "rock": 5,
        "electric": 4,
        "normal": 3,
        "poison": 3,
        "grass": 1,
        "psychic": 1,
        "ice": 0,
    },
    {
        "dragon": 32,
        "flying": 26,
        "dark": 25,
        "water": 22,
        "ground": 21,
        "fairy": 20,
        "fire": 18,
        "steel": 17,
        "fighting": 16,
        "ghost": 10,
        "poison": 4,
        "normal": 2,
        "grass": 1,
        "bug": 0,
        "electric": 0,
        "ice": 0,
        "psychic": 0,
        "rock": 0,
    },
    {
        "dragon": 30,
        "steel": 22,
        "flying": 21,
        "water": 18,
        "dark": 18,
        "ground": 14,
        "fighting": 13,
        "fire": 11,
        "fairy": 11,
        "ghost": 7,
        "poison": 4,
        "normal": 3,
        "bug": 3,
        "grass": 1,
        "electric": 1,
        "ice": 0,
        "psychic": 0,
        "rock": 0,
    },
]
tour_names = ["MWP VII", "MFPL III", "MPL"]
for i in range(3):
    results += [tour_names[i] + ": "]
    TOURNEY_COUNTS = tour_data[i]
    tourney_counts = np.array([TOURNEY_COUNTS[t] for t in types], dtype=float)
    tourney_usage = tourney_counts / tourney_counts.sum()
    PRIOR = 0.5  # Dirichlet pseudo-count per type, so rare or unused types aren't "impossible"

    # Experiment 5, set in stone: point-estimate matrix and point-estimate usage
    stone_p, stone_value = solve_banded_row(best_chart, tourney_usage, band=BAND)

    # Experiment 6, confidence interval: resample the matrix and the usage
    rng = np.random.default_rng(1)
    usage_samples = rng.dirichlet(tourney_counts + PRIOR, size=N_SAMPLES)
    ci_stone_p = np.mean(
        [solve_banded_row(A, u, BAND)[0] for A, u in zip(samples, usage_samples)],
        axis=0,
    )

    results += [
        format_mix(
            f"5. Tournament usage, band {BAND}, fixed (value {stone_value:.4f})",
            types,
            stone_p,
            CELL_CAP,
            LINE_WIDTH,
        ),
        format_mix(
            f"6. Confidence interval, tournament usage, band {BAND} ({N_SAMPLES} samples)",
            types,
            ci_stone_p,
            CELL_CAP,
            LINE_WIDTH,
        ),
    ]

text = "\n".join(results)
with open("nash_chart.txt", "w") as file:
    file.write(text)

with open("overall_chart.txt", "w") as file:
    for i in global_chart:
        stri = ""
        for j in i:
            stri = stri + str(j) + ","
        file.write(stri[:-1] + "\n")

with open("timeline.txt", "w") as file:
    file.write("\n".join(lines))
