import random
import numpy
import math
import time
import copy
import KernighanLinGraphPartitioning
import repair


def load_graph(filename):
    lines = open(filename).readlines()
    n_node = int(lines[0].split()[0])
    n_edge = int(lines[0].split()[1])
    G = [[0 for _ in range(n_node)] for _ in range(n_node)]

    for i in range(1, len(lines)):
        a = int(lines[i].split()[0]) - 1
        b = int(lines[i].split()[1]) - 1
        v = int(lines[i].split()[2])
        G[a][b] = G[b][a] = v

    return G, n_node, n_edge, lines


def adjustment(x, lb, ub):
    for i in range(len(x)):
        if x[i] < lb:
            x[i] = lb
        if x[i] > ub:
            x[i] = ub
    return x


def mutation(rl, x, hawks):
    F = 0.5
    randomHawkIndexList = random.sample(range(0, hawks - 1), 4)
    return rl + F * (x[randomHawkIndexList[0]] - x[randomHawkIndexList[1]]) + (
                x[randomHawkIndexList[2]] - x[randomHawkIndexList[3]])


def random_swap(x):
    rand1 = random.randint(0, dim - 1)
    rand2 = random.randint(0, dim - 1)
    x[rand1], x[rand2] = x[rand1], x[rand2]
    return x


def decomposition(x):
    temp = x.copy()
    for j in range(len(x // 2), len(x)):
        x[j] = numpy.random.randint(0, 2)
    for j in range(len(x) // 2):
        temp[j] = numpy.random.randint(0, 2)

    if objf(temp) > objf(x):
        return temp
    else:
        return x


def synthesis(x, x_rand):
    x_temp = []
    for j in range(len(x) // 2):
        x_temp.append(x[j])
    for j in range(len(x) // 2, len(x)):
        x_temp.append(x_rand[j])
    for j in range(len(x) // 2):
        x[j] = x_rand[j]
    return x_temp, x


def acceptance_criterion(t, x):
    if objf(t) > objf(x):
        x = copy.deepcopy(t)
    return x


def objf(F):
    s = 0

    for i in range(1, len(lines)):
        a = int(lines[i].split()[0]) - 1
        b = int(lines[i].split()[1]) - 1

        if F[a] != F[b]:
            s += int(lines[i].split()[2])

    return s


def HHO(lb, ub, dim, SearchAgents_no, Max_iter):
    Rabbit_Location = numpy.zeros(dim)
    Rabbit_Energy = float("-inf")
    Hawk_Update = [0 for _ in range(SearchAgents_no)]

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)
    X = []

    # Kernighan-Lin
    localResult = KernighanLinGraphPartitioning.graph_partition(filename=filename)

    for i in range(SearchAgents_no):
        X.append(localResult.copy())
        X[i][random.randint(0, len(X[i]) - 1)] = random.randint(0, 1)

    X[0] = copy.deepcopy(localResult)
    X = numpy.asarray(X)
    timerStart = time.time()
    t = 0

    while t < Max_iter:
        for i in range(0, SearchAgents_no):
            X[i, :] = adjustment(X[i], lb, ub)
            fitness = objf(X[i, :])

            if fitness > Rabbit_Energy:
                Hawk_Update[i] = t
                Rabbit_Energy = fitness
                Rabbit_Location = copy.deepcopy(X[i])
            if t - Hawk_Update[i] >= 50:
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]

                # Decomposition
                X[i] = decomposition(X[i])

                # Synthesis
                X_rand, X[i] = synthesis(X, X_rand)

        E1 = 2 * (1 - (t / Max_iter))

        for i in range(0, SearchAgents_no):
            temp = copy.deepcopy(X[i])
            E0 = 2 * random.random() - 1
            Escaping_Energy = E1 * E0

            if abs(Escaping_Energy) >= 1:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    X[i, :] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i, :])

                elif q >= 0.5:
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random.random() * ((ub - lb) * random.random() + lb)
                X[i] = numpy.clip(X[i, :], lb, ub)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                r = random.random()

                if r >= 0.5 and abs(Escaping_Energy) < 0.5:
                    X[i, :] = Rabbit_Location - Escaping_Energy * abs(Rabbit_Location - X[i, :])

                if r >= 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :])

                if r < 0.5 and abs(Escaping_Energy) >= 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X[i, :])
                    X1 = numpy.clip(X1, lb, ub)

                    if objf(X1) > fitness:
                        X[i, :] = X1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - X[i, :]) + numpy.multiply(numpy.random.randn(dim),
                                                                                        Levy(dim))
                        X2 = numpy.clip(X2, lb, ub)
                        if objf(X2) > fitness:
                            X[i, :] = X2.copy()

                if r < 0.5 and abs(Escaping_Energy) < 0.5:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(Jump_strength * Rabbit_Location - X.mean(0))
                    X1 = numpy.clip(X1, lb, ub)

                    if objf(X1) > fitness:
                        X[i, :] = X1.copy()
                    else:
                        X2 = Rabbit_Location - Escaping_Energy * abs(
                            Jump_strength * Rabbit_Location - X.mean(0)) + numpy.multiply(numpy.random.randn(dim),
                                                                                          Levy(dim))
                        X2 = numpy.clip(X2, lb, ub)
                        if objf(X2) > fitness:
                            X[i, :] = X2.copy()

            # Random Swap
            X[i] = random_swap(X[i])

            # Acceptance Criterion
            X[i] = acceptance_criterion(temp, X[i])

            # Repair Operator
            X[i] = repair.repair(filename, X[i])

        print(['At iteration ' + str(t) + ' the best fitness is ' + str(Rabbit_Energy)])
        t = t + 1

    timerEnd = time.time()
    print((timerEnd - timerStart) / Max_iter)
    print(Rabbit_Location)
    return Rabbit_Energy


def Levy(dim):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step


if __name__ == '__main__':
    filename = 'Gset/G2.txt'
    G, dim, edges, lines = load_graph(filename)
    V = [i for i in range(dim)]
    Max_iter = 2
    Search_agent_no = 10
    lower, upper = 0, 1

    finalResult = HHO(lower, upper, dim, Search_agent_no, Max_iter)
    print('Best Fitness Value :', finalResult)
