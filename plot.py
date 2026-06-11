import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

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

history = np.loadtxt("history.csv", delimiter=",")
for t in range(18):
    plt.plot(history[:, t], label=types[t])
    plt.text(len(history) - 1, history[-1, t], types[t], fontsize=8, va="center")
plt.tight_layout()
plt.legend()
plt.show()

exclude = {
    "water",
    "flying",
    "steel",
    "fairy",
    "dark",
    "dragon",
    "fighting",
    "ground",
    "fire",
}

plt.figure()
texts = []
for t in range(18):
    if types[t] not in exclude:
        plt.plot(history[:, t], label=types[t])
        texts.append(plt.text(len(history) - 1, history[-1, t], types[t], fontsize=8))
adjust_text(texts)
plt.tight_layout()
plt.show()
