import random


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


def objf(F):
    s = 0

    for i in range(1, len(lines)):
        a = int(lines[i].split()[0]) - 1
        b = int(lines[i].split()[1]) - 1

        if F[a] != F[b]:
            s += int(lines[i].split()[2])

    return s


def repair(filename, sub):
    G, dim, edges, lines = load_graph(filename)
    lst = [i.count(1) for i in G]
    cut = [0 for i in range(dim)]

    for i in range(1, len(lines)):
        a = int(lines[i].split()[0]) - 1
        b = int(lines[i].split()[1]) - 1

        if sub[a] != sub[b]:
            cut[a] += 1
            cut[b] += 1

    uncut = [lst[i] - cut[i] for i in range(dim)]

    for loop in range(20):
        for i in range(dim):
            if cut[i] < uncut[i]:
                profit = uncut[i] - cut[i]

                for j in range(len(G[i])):
                    if G[i][j] == 1:
                        if sub[i] == sub[j]:
                            profit += 1
                        else:
                            profit -= 1

                if profit > 0:
                    cut[i], uncut[i] = uncut[i], cut[i]

                    for j in range(len(G[i])):
                        if G[i][j] == 1:
                            if sub[i] == sub[j]:
                                cut[j] += 1
                                uncut[j] -= 1
                            else:
                                cut[j] -= 1
                                uncut[j] += 1
                    sub[i] = 1 - sub[i]
    return sub


if __name__ == '__main__':
    filename = 'sample_graph.txt'
    G, dim, edges, lines = load_graph(filename)
    best = 0
    best_sub = []
    t = 0

    while True:
        sub = [0 for i in range(dim)]
        lst = random.sample([i for i in range(dim)], dim//2)

        for i in lst:
            sub[i] = 1

        ans = repair(filename, sub)

        if objf(ans) > best:
            best = objf(ans)
            best_sub = ans.copy()
        print(t, best, objf(ans), best_sub)
        t += 1
