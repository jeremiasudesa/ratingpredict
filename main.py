import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

NUM_PLAYERS = 1000
TRUE_SKILL = truncnorm((0 - 1500), (3000 - 1500), loc=1500, scale=1).rvs(NUM_PLAYERS)
rating = ([(1500,)] * NUM_PLAYERS)
RD = np.array([350] * NUM_PLAYERS)
Q = np.log(10) / 400

def g(RD_): # funcion magica
    return 1 / np.sqrt(1 + 3 * (Q ** 2) * (RD_ ** 2) / (np.pi ** 2))

def E_rating(p, p_j): # resultado esperado de una partida, usando como informacion el rating de cada jugador
    return 1 / (1 + 10 ** (-g(RD[p_j]) * (rating[p][-1] - rating[p_j][-1]) / 400))

def E_real(p, p_j): # esperanza real de la partida, sabiendo el skill real de ambos jugadores
    return 1 / (1 + 10 ** (-g(RD[p_j]) * (TRUE_SKILL[p] - TRUE_SKILL[p_j]) / 400))

def d_squared(p, p_j): # otra funcion magica
    spread = E_rating(p, p_j)
    return 1 / ((Q ** 2) * (g(RD[p_j]) ** 2) * spread * (1 - spread))

def update_rating_singular(p, p_j, s): # actualiza el rating de un jugador
    if(d_squared(p, p_j) == 0 or (RD[p] ** 2) == 0):
        return
    rating[p] += (rating[p][-1] + Q / (1 / (RD[p] ** 2) + 1 / d_squared(p, p_j)) * g(RD[p_j]) * (s - E_rating(p, p_j)), )

def update_RD_singular(p, p_j): # lo mismo pero para la varianza
    if(d_squared(p, p_j) == 0 or (RD[p] ** 2) == 0):
        return
    RD[p] = np.sqrt(1 / (1 / (RD[p] ** 2) + 1 / d_squared(p, p_j)))

def update_rating_from_match(p, p_j, s): # actualiza informacion en base a los resultados de una partida
    update_rating_singular(p, p_j, s)
    update_rating_singular(p_j, p, 1 - s)
    update_RD_singular(p, p_j)
    update_RD_singular(p_j, p)

GAMES = 100000
error = []
for m in range(GAMES):
    p1, p2 = np.random.choice(range(NUM_PLAYERS), size = 2, replace = False)
    result = np.random.binomial(n=1, p=E_real(p1, p2))
    update_rating_from_match(p1, p2, result)
    squared_diff = []
    for i in range(NUM_PLAYERS):
        squared_diff.append((rating[i][-1] - TRUE_SKILL[i]) ** 2)
    error.append(np.mean(squared_diff))
    
# graficar las curvas de rating
for player in range(NUM_PLAYERS): 
    print(TRUE_SKILL[player])
    print(rating[player])
    plt.plot([i for i in range(len(rating[player]))], [TRUE_SKILL[player] for i in range(len(rating[player]))])
    plt.plot(rating[player])
    plt.show()

# MSE a medida que aumentan las partidas
print(np.sqrt(error[-1]))
plt.plot(error)
plt.xlabel("partidas")
plt.ylabel("MSE")
plt.show()

# Este codigo es para visualizar N < 10
# print(rating)
# print(TRUE_SKILL)