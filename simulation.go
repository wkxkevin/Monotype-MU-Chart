package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
)

const TYPES = 18
const GAMES = 100_000
const GENS = 100_000

var typeNames = [TYPES]string{
	"normal", "fighting", "flying", "poison", "ground", "rock",
	"bug", "ghost", "steel", "fire", "water", "grass",
	"electric", "psychic", "ice", "dragon", "dark", "fairy",
}

func loadChart(filename string) [TYPES][TYPES][2]int {
	var chart [TYPES][TYPES][2]int
	f, _ := os.Open(filename)
	scanner := bufio.NewScanner(f)
	row := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		vals := strings.Split(line, ",")
		parsed := []float64{}
		for _, v := range vals {
			v = strings.TrimSpace(v)
			if v == "" {
				continue
			}
			f, _ := strconv.ParseFloat(v, 64)
			parsed = append(parsed, f)
		}
		which := row / 2 // 0=lo, 1=hi within pair
		typeRow := row / 2
		if row%2 == 0 {
			for col, v := range parsed {
				chart[typeRow][col][0] = int(v * 100)
			}
		} else {
			for col, v := range parsed {
				chart[typeRow][col][1] = int(v * 100)
			}
		}
		_ = which
		row++
	}
	return chart
}

func softmax(wr [TYPES]float64, temp float64) [TYPES]float64 {
	var result [TYPES]float64
	max := wr[0]
	for _, v := range wr {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for i, v := range wr {
		result[i] = math.Exp((v - max) / temp)
		sum += result[i]
	}
	for i := range result {
		result[i] /= sum
	}
	return result
}

func weightedChoice(probs [TYPES]float64) int {
	r := rand.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return TYPES - 1
}

func main() {
	chart := loadChart("chart.txt")
	playerFreq := [TYPES]int{}
	for i := range playerFreq {
		playerFreq[i] = 100
	}
	file, _ := os.Create("history.csv")
	writer := bufio.NewWriter(file)
	for gen := 0; gen < GENS; gen++ {
		// sample matchup chart for this gen
		if gen%1000 == 0 {
			fmt.Println(gen)
		}
		var currChart [TYPES][TYPES]int
		for i := range currChart {
			for j := range currChart[i] {
				lo := chart[i][j][0]
				hi := chart[i][j][1]
				currChart[i][j] = lo + rand.Intn(hi-lo+1)
			}
		}

		// build players array
		totalPlayers := 0
		for _, f := range playerFreq {
			totalPlayers += f
		}
		players := make([]int, 0, totalPlayers)
		for t, count := range playerFreq {
			for i := 0; i < count; i++ {
				players = append(players, t)
			}
		}

		playerWins := make([]int, totalPlayers)
		playerLosses := make([]int, totalPlayers)

		// run games
		for game := 0; game < GAMES; game++ {
			p1 := rand.Intn(totalPlayers)
			p2 := rand.Intn(totalPlayers)
			winrate := currChart[players[p1]][players[p2]]
			if rand.Intn(10001) < winrate {
				playerWins[p1]++
				playerLosses[p2]++
			} else {
				playerWins[p2]++
				playerLosses[p1]++
			}
		}

		// compute type winrates
		var typeWins, typeGames [TYPES]int
		for i, t := range players {
			typeWins[t] += playerWins[i]
			typeGames[t] += playerWins[i] + playerLosses[i]
		}
		var typeWinrate [TYPES]float64
		for t := range typeWinrate {
			if typeGames[t] > 0 {
				typeWinrate[t] = float64(typeWins[t]) / float64(typeGames[t])
			} else {
				typeWinrate[t] = 0.5
			}
		}

		// pick probs with softmax + epsilon
		sm := softmax(typeWinrate, 0.2)
		var pickProbs [TYPES]float64
		for i := range pickProbs {
			pickProbs[i] = 0.95*sm[i] + 0.05/TYPES
		}

		// find bad types for contrarians
		badTypes := []int{}
		for t, f := range playerFreq {
			if f < 10 {
				badTypes = append(badTypes, t)
			}
		}
		if len(badTypes) == 0 {
			// find worst type
			worst := 0
			for t := range playerFreq {
				if playerFreq[t] < playerFreq[worst] {
					worst = t
				}
			}
			badTypes = []int{worst}
		}

		// switching
		for i := range players {
			total := playerWins[i] + playerLosses[i]
			wr := 0.5
			if total > 0 {
				wr = float64(playerWins[i]) / float64(total)
			}
			if wr < 0.30 {
				var newType int
				if rand.Float64() < 0.05 {
					newType = badTypes[rand.Intn(len(badTypes))]
				} else {
					newType = weightedChoice(pickProbs)
				}
				players[i] = newType
			}
		}

		// update freq
		playerFreq = [TYPES]int{}
		for _, t := range players {
			playerFreq[t]++
		}

		parts := make([]string, TYPES)
		for i, f := range playerFreq {
			parts[i] = strconv.Itoa(f)
		}
		fmt.Fprintln(writer, strings.Join(parts, ","))
	}
	writer.Flush()
	file.Close()
}
