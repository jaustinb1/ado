import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class Node:

    def __init__(self, i, j):
        self.i = float(i) + np.random.rand()
        self.j = float(j) + np.random.rand()

    def dist_to(self, n):
        assert isinstance(n, Node)

        return np.sqrt((n.i - self.i)**2 + (n.j - self.j)**2)

    def __repr__(self):
        return "({},{})".format(self.i, self.j)


class Graph:

    def __init__(self, dim):
        self.nodes = []
        self.dim = dim

        for i in range(dim):
            for j in range(dim):
                self.nodes.append(Node(i, j))

    def as_arr(self):
        return [n.i for n in self.nodes], [n.j for n in self.nodes]


Bunch = set

class ADO:

    def __init__(self, graph, k):
        self.g = graph
        self.A = []
        self.k = k
        self.n = len(self.g.nodes)
        self.p = defaultdict(lambda: {})
        self.bunches = {}
        self.A.append(self.g.nodes)

        for _ in range(self.k-1):
            nodes = []
            for n in self.A[-1]:
                rnd = np.random.rand()

                if rnd < self.n **(-1. / self.k):
                    nodes.append(n)
            self.A.append(nodes)

        self.A.append([])
        for i in range(len(self.A)):
            self.A[i] = set(self.A[i])


        # calculate p
        for v in self.g.nodes:
            self.bunches[v] = Bunch()

            for i in range(len(self.A)):
                delta = float("inf")

                for w in self.A[i]:
                    if v == w:
                        continue
                    if v.dist_to(w) < delta:
                        self.p[i][v] = w
                    delta = min(delta, v.dist_to(w))

                if i >= 1:
                    to_add_to_bunch = [ w for w in self.A[i-1].difference(self.A[i]) if w.dist_to(v) < delta]
                    for x in to_add_to_bunch:
                        self.bunches[v].add(x)

    def estimate(self, x, y):

        plt.figure()
        plt.scatter(*self.g.as_arr(), c='blue')

        xx = [n.i for n in self.bunches[x]]
        yy = [n.j for n in self.bunches[x]]
        plt.scatter(xx, yy, c='black')
        plt.scatter([x.i], [x.j], c='red')

        for i in range(len(self.p)):
            plt.scatter([self.p[i][x].i], [self.p[i][x].j], color='green')

        plt.figure()
        plt.scatter(*self.g.as_arr(), c='blue')

        xx = [n.i for n in self.bunches[y]]
        yy = [n.j for n in self.bunches[y]]
        plt.scatter(xx, yy, c='black')
        plt.scatter([y.i], [y.j], c='red')
        for i in range(len(self.p)):
            plt.scatter([self.p[i][y].i], [self.p[i][y].j], color='green')

        plt.show()


if __name__ == "__main__":

    g = Graph(15)

    a = ADO(g, 5)

    a.estimate(g.nodes[0], g.nodes[10])
