# Monotype Matchup Analysis for Pokémon Showdown

Statistical analysis of Gen 9 Monotype matchup data from Smogon's public monthly stats: confidence intervals on every type-vs-type matchup, a simplified matchup scale, equilibrium analysis of the matchup game, and a metagame simulation.

> **Status: work in progress.** The equilibrium analysis is recent and the simulation inputs have not yet been regenerated from the latest data. See [Limitations](#limitations) and [Possible next steps](#possible-next-steps).

## The question

In Monotype, each team is built around one of 18 types, so the metagame is close to a game with 18 strategies. If you know how often each type beats each other type, which types should be played, and how much of that is real signal versus sampling noise?

## Data

- **Source:** Smogon's public stats (`smogon.com/stats`): the monthly Monotype matchup charts and metagame usage stats.
- **Coverage:** March to October 2026, for `gen9monotype` and `gen9nationaldexmonotype`, at four rating cutoffs (0, 1500, 1630, 1760).
- **What is analyzed in depth:** `gen9monotype` at the 1630 cutoff. Higher cutoffs have far fewer games per matchup. At 1630 there were about 16,000 games in March and about 6,000 per month more recently.
- **Gliscor ban.** Gliscor was banned on August 7, 2026 (effective a few days later). It can only appear on Ground and Flying teams, so every matchup involving Flying or Ground uses only post-ban months (August onward). All other matchups use all months, since more data reduces noise. Set `gliscor_ban = False` in `muchart.py` to use all months for everything.

## Method

1. **Aggregate** each month's matchup counts into one matrix per chart (`aggregate/`).
2. **Confidence intervals.** A 95% Wilson score interval for every matchup win rate (`ConfInterv/`).
3. **Simplified scale.** Win rates are bucketed in steps of 10 percentage points around an "even" band of 45 to 55%, and the interval bounds are combined into a single scale from -2 to +2 (`overall_chart.txt`; see `bucket()` and the `global_chart` logic in `muchart.py`).
4. **Equilibrium of the matchup game.** The 1630 win-rate matrix defines a symmetric zero-sum game. The equilibrium mix is solved with linear programming (`scipy.optimize.linprog`, HiGHS). The game value comes out at 0.5000, as it must for a symmetric game.
5. **Best response to a usage distribution.** `solve_banded_row` finds the mix that maximizes expected win rate against an opponent whose type usage may vary within a band (`BAND`) around an observed distribution. With `BAND = 0` this is the best response to the observed usage exactly.
6. **Uncertainty.** Each matchup has a Beta posterior from its win and loss counts. The analysis draws 1,000 matrices (`N_SAMPLES`), solves each one, and reports the average mix. For tournament usage, the usage distribution is also resampled from a Dirichlet posterior.

`muchart.py` writes every variant to `nash_chart.txt`: equilibrium and best-response analyses, each as a point estimate and as a resampled average, with the best-response ones run against ladder usage and against each tournament's usage.

Usage distributions come from two places: Smogon's monthly ladder usage for the post-ban months, and type counts from three tournaments (MWP VII, MFPL III and MPL), entered by hand in `tour_data`.

### Simulation

`simulation.go` is an evolutionary model of the metagame:

- 18 types with 100 players each. Every generation draws a matchup matrix uniformly inside the Wilson intervals, plays 100,000 random games, and measures each type's win rate.
- Players below a 30% win rate switch types. Most pick a new type by a softmax over type win rates (temperature 0.2, plus 5% uniform exploration). A further 5% are "contrarians" who pick an under-used type.
- It runs for 100,000 generations and writes the type counts per generation to `history.csv`. `plot.py` plots them.

`simulation.py` is an earlier prototype of the same model. It differs in how switching players choose a new type.

## Limitations

- **Ladder, not tournaments.** The matchup data comes from ladder games at the 1630 cutoff. Tournament play involves preparation, scouting and team building that type-level win rates do not capture.
- **Types, not teams.** A type's win rate mixes strong and weak teams, and the analysis cannot see individual Pokémon.
- **Thin post-ban data.** Matchups involving Flying or Ground rest on far fewer games than the rest of the matrix, so their estimates are noisier. Point-estimate optimization can latch onto that noise, so the resampled analyses are more trustworthy than the point estimates.
- **Not everything is an equilibrium.** Only the uniform-usage analyses solve for an equilibrium. The ladder-usage and tournament-usage analyses are best responses to an assumed opponent distribution.
- **Hand-entered tournament data.** The tournament counts are typed into `muchart.py` and not sourced in the repository.
- **Simulation.** The parameters (softmax temperature, 5% exploration, 30% switching threshold) are reasonable choices but are not calibrated against data. Its inputs, `chart.txt` and `matchup_1630.npy`, were built from March to May 2026 data (before the Gliscor ban) and have not been regenerated. The script that exports them is not in this repository.

## Repository contents

| Path | Contents |
|---|---|
| `muchart.py` | Fetches the stats, builds the aggregates, intervals, scale and equilibrium analysis, and writes the outputs below |
| `2026-03/` to `2026-10/` | Raw monthly matchup charts as fetched |
| `aggregate/` | Summed win counts and win percentages per chart |
| `ConfInterv/` | Wilson intervals per matchup, plus bucketed `simple_` versions |
| `overall_chart.txt` | The simplified -2 to +2 matchup scale for 1630 |
| `timeline.txt` | Bucketed interval bounds across the 0, 1500 and 1630 cutoffs |
| `nash_chart.txt` | Output of all equilibrium and best-response analyses |
| `simulation.go` | Metagame simulation (reference implementation) |
| `simulation.py` | Earlier Python prototype of the simulation |
| `chart.txt`, `matchup_1630.npy` | Lower and upper interval bounds used as simulation input |
| `history.csv` | Simulation output: type counts per generation |
| `plot.py` | Plots `history.csv` |
| `MU_chart.ipynb` | The original exploratory notebook (viability scores, a ±4 binned chart, Wilson intervals). `muchart.py` supersedes it. |

## Running it

```
pip install numpy scipy requests matplotlib adjustText
python muchart.py       # needs internet; rewrites the month folders, aggregate/, ConfInterv/, and the output text files
go run simulation.go    # reads chart.txt, writes history.csv (100,000 generations, can take a long time)
python plot.py          # plots history.csv
```

Settings at the top of the analysis section of `muchart.py`:

| Setting | Meaning |
|---|---|
| `gliscor_ban` | Use only post-ban months for Flying and Ground matchups |
| `BAND` | How far the usage distribution may vary from the observed one |
| `N_SAMPLES` | Number of resampled matrices |

## Possible next steps

- Regenerate the simulation inputs from the current data window, then compare the simulation's long-run type frequencies with the equilibrium mix.
- Validate the model against external rankings and tournament results. A community type ranking on the [Smogon forums](https://www.smogon.com/forums/threads/sv-monotype-metagame-discussion-indigo-disk.3732581/post-11178694) places Dragon and Steel above Flying, which is an open question for the model.
- Shrink the thin post-ban Flying and Ground estimates toward the older data instead of discarding it.
