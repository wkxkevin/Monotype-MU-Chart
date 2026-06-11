import os

import numpy as np
import requests
from scipy import stats


def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - margin) * 100, (center + margin) * 100


dates = ["2026-03", "2026-04", "2026-05"]
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

aggregate_charts = np.zeros((8, 18, 18))
for date in dates:
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
            aggregate_charts[i] += np.array(
                [
                    [float(cell.strip()) for cell in line.split("|")]
                    for line in mod_chart
                ]
            )
            file.write("\n".join(mod_chart))

os.makedirs("aggregate", exist_ok=True)
print("Aggregate")
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

col_w = 9
sep = "+" + (("-" * col_w + "+") * 19)

for i in range(8):
    chart = aggregate_charts[i]
    total = chart + chart.T
    percent = np.divide(chart, total, out=np.zeros_like(chart), where=total != 0) * 100

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
        return -int((55 - pct) // 10) - 1
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

with open("overall_chart.txt", "w") as file:
    for i in global_chart:
        stri = ""
        for j in i:
            stri = stri + str(j) + ","
        file.write(stri[:-1] + "\n")

with open("timeline.txt", "w") as file:
    file.write("\n".join(lines))
