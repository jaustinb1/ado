import random
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class Vertex:
    """
    A representation of a single point in a finite metric space.
    """

    def __init__(self, i: float, j: float, stddev: float = 0.1,
                 idx: Optional[int] = None):
        """
        :param i: The x-coordinate (mean) of the point
        :param j: The y-coordinate (mean) of the point
        :param stddev: Standard deviation of the normal distribution the
        point is sampled from
        :param idx: (Optional) Index of the point in the graph for bookkeeping
        """
        self.i = np.random.normal(loc=i, scale=stddev)
        self.j = np.random.normal(loc=j, scale=stddev)
        if idx is not None:
            self.idx = idx

    def dist_to(self, v: 'Vertex') -> float:
        assert isinstance(v, Vertex)
        return np.sqrt((v.i - self.i) ** 2 + (v.j - self.j) ** 2)

    def __repr__(self) -> str:
        return "({},{})".format(self.i, self.j)


class Graph:
    """
    A representation of a graph (collection of Vertices with distances
    between them) in a finite metric space.
    """

    def __init__(self, vertices_list: List[Vertex]):
        """
        :param vertices_list: A list of points (Vertex objects) to put in the
        graph.
        """
        self.vertices = vertices_list

        # Mark the vertices with their indices for convenience and to
        # distinguish them in the unlikely case of exact duplicates
        for i, v in enumerate(self.vertices):
            v.idx = i

        self.dim = len(vertices_list)

        # Compute finite metric distance matrix (n x n)
        z = np.array([complex(v.i, v.j) for v in self.vertices])
        self.distances = abs(z.T - z)

    @staticmethod
    def create_random_graph(num_vertices: int,
                            x_range: Tuple[float, float] = (-100, 100),
                            y_range: Tuple[float, float] = (-100, 100),
                            seed: int = 0) -> \
            'Graph':
        """
        Construct a graph with num_vertices points by sampling each point
        uniformly from the provided range of x and y values.

        :param num_vertices: The number of vertices to place in the graph.
        :param x_range: The range (low, high) of x coordinates to sample
        points from.
        :param y_range: The range (low, high) of y coordinates to sample
        points from.
        :param seed: The random seed to use for generating the data.

        :return: A Graph object containing num_vertices random points,
        with the corresponding distance matrix constructed internally.
        """
        random.seed(seed)
        np.random.seed(seed)

        x_coords = [random.uniform(x_range[0], x_range[1]) for _ in range(
            num_vertices)]
        y_coords = [random.uniform(y_range[0], y_range[1]) for _ in range(
            num_vertices)]
        random_vertices = [Vertex(i, j) for i, j in zip(x_coords, y_coords)]

        return Graph(vertices_list=random_vertices)

    def as_array(self) -> Tuple[List[float], List[float]]:
        return [v.i for v in self.vertices], [v.j for v in self.vertices]


class ADO:
    """
    An implementation of the approximate distance oracle algorithm/data
    structure.
    """

    def __init__(self, graph: Graph, k: int):
        """
        :param graph: A Graph object that the ADO will preprocess and
        answer queries on.
        :param k: The tunable ADO parameter that controls the data structure
        size, query runtime, and stretch of the estimates.
        """
        self.g = graph
        self.n = self.g.dim
        self.k = k

        # These fields are initialized in preprocess below
        self.A = None
        self.p = None
        self.B = None

        # Preprocess and construct A, p, B
        self.preprocess()

    def preprocess(self) -> None:
        """
        Do all preprocessing on the graph to answer queries.
        Runs in O(n^2) time.
        """
        # Sample k + 1 sets A_0 through A_k
        A_sets = self.__sample_A_sets()
        while not A_sets[self.k - 1]:  # If A_{k-1} is empty, re-sample
            A_sets = self.__sample_A_sets()
        self.A = A_sets

        # Compute distances between each point and each set A_i, saving
        # the closest points p_i(v)
        self.p = self.__compute_p()

        # Compute a bunch and associated distance table for each vertex
        self.B = self.__compute_bunches()

    def __sample_A_sets(self) -> List[Set[Vertex]]:
        """
        Constructs k+1 sets A_0 through A_k by repeated sampling with
        probability n^{-1/k} from the previous set. Note that A_0 = V and A_k
        = {}.

        :return: A list of sets A_0 through A_k in that order, where each set
        is a set of Vertex objects.
        """
        # Add set A_0 = V
        A_sets = [set(self.g.vertices)]

        # Add sets A_1 through A_{k-1} by sampling
        for _ in range(1, self.k):
            vertices = set([v for v in A_sets[-1]
                            if random.random() < (self.n ** (-1. / self.k))])
            A_sets.append(vertices)

        # Add set A_k = {}
        A_sets.append(set())

        return A_sets

    def __compute_p(self) -> DefaultDict[Vertex, Dict[int, Tuple[Vertex,
                                                                 float]]]:
        """
        Computes the distance from each vertex v to each set A_i, and notes
        the nearest point in A_i to v (p_i(v)).

        :return: A dictionary mapping from a Vertex v to a dictionary, which
        in turn maps from an index i (from 0 to k) corresponding to a set A_i to
        a tuple of (p_i(v), distance(v, A_i)).
        """
        p_dict = defaultdict(dict)

        # Compute minimum distance from every point v to every set A_i
        # Save the closest point in the set p_i(v) in the dict p_dict
        for v in self.g.vertices:
            for i, A_i in enumerate(self.A):
                p_i = None
                delta = float("inf")

                for u in A_i:
                    d = v.dist_to(u)
                    if d < delta:
                        p_i = u
                        delta = d

                p_dict[v][i] = (p_i, delta)

        return p_dict

    def __compute_bunches(self) -> DefaultDict[Vertex, Dict[Vertex, float]]:
        """
        Computes the bunch B(v) associated with each vertex v and for each
        vertex w in the bunch saves the distance from v to w in a
        dictionary/lookup table.

        :return: A dictionary mapping from a Vertex v to its associated bunch
        B(v), which is represented as a dictionary where the keys are each of
        the vertices w in B(v) and the values are the corresponding distances
        distance(v, w).
        """
        bunches = defaultdict(dict)
        for v in self.g.vertices:
            for i, A_i in enumerate(self.A[:self.k]):
                A_i_plus_1 = self.A[i + 1]
                candidates = A_i.difference(A_i_plus_1)
                for w in candidates:
                    d = v.dist_to(w)
                    if d < self.p[v][i + 1][1]:
                        bunches[v][w] = d

        return bunches

    def query(self, u: Vertex, v: Vertex) -> float:
        """
        Query the ADO for the estimated distance between two vertices u and v.

        :param u: A vertex in the ADO's graph self.g
        :param v: A different vertex in the ADO's graph self.g
        :return: The estimated distance between u and v.
        """
        # Check that the vertices are actually from this graph
        assert self.g.vertices[u.idx] == u and self.g.vertices[v.idx] == v, \
            "The vertices being queried must be in the graph."

        w = u
        i = 0

        while w not in self.B[v]:
            i += 1
            u, v = v, u
            w = self.p[u][i][0]

        return self.B[u][w] + self.B[v][w]

    """
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
    """

    def animate_query(self, u: Vertex, v: Vertex, timestep: float = 2.0) -> \
            None:
        """
        Query the ADO for the estimated distance between two vertices u and v
        and animate/plot the process with pyplot.

        :param u: A vertex in the ADO's graph self.g
        :param v: A different vertex in the ADO's graph self.g
        :param timestep: The timestep between frames in the animation
        """
        # Check that the vertices are actually from this graph
        assert self.g.vertices[u.idx] == u and self.g.vertices[v.idx] == v, \
            "The vertices being queried must be in the graph."

        fig, ax = plt.subplots()

        w = u
        i = 0

        self.__plot_query_state(fig, ax, u, v, w, i)
        plt.pause(timestep)

        while w not in self.B[v]:
            i += 1
            u, v = v, u
            w = self.p[u][i][0]

            self.__plot_query_state(fig, ax, u, v, w, i)
            plt.pause(timestep)

        self.__plot_query_state(fig, ax, u, v, w, i, final=True)

    def __plot_query_state(self, fig: plt.Figure,
                           ax: plt.Axes,
                           u: Vertex, v: Vertex,
                           w: Vertex, i: int,
                           final: bool = False) -> None:
        """
        Plot a single frame/plot that depicts the current state of the ADO
        query. Clears and overwrites any previous contents of the plot.

        :param fig: A Matplotlib figure object representing the figure being
        modified/displayed.
        :param ax: A Matplotlib Axes object representing the subplot being
        modified/displayed.
        :param u: The current value of u in the ADO query algorithm.
        :param v: The current value of v in the ADO query algorithm.
        :param w: The current value of w in the ADO query algorithm.
        :param i: The iteration of the ADO query algorithm.
        :param final: Whether or not this is the final query state/final
        iteration of the algorithm.
        """
        ax.cla()

        # Plot all the points in the graph
        ax.scatter([v.i for v in self.g.vertices],
                   [v.j for v in self.g.vertices],
                   s=4,
                   color="black",
                   marker=".",
                   label="Points")

        # Plot u, v, w with special symbols/colors
        ax.scatter([u.i], [u.j],
                   s=12,
                   color="red",
                   marker="*",
                   label="u")
        ax.annotate("u", (u.i, u.j), color="red")
        ax.scatter([v.i], [v.j],
                   s=12,
                   color="green",
                   marker="*",
                   label="v")
        ax.annotate("v", (v.i, v.j), color="green")
        ax.scatter([w.i], [w.j],
                   s=5,
                   color="orange",
                   marker="p",
                   label="w")
        ax.annotate("w", (w.i, w.j), color="orange",
                    xytext=(-15, -10),
                    textcoords="offset pixels")

        # For the current u, mark and label its p_i(u)s
        p_i_u = [self.p[u][i] for i in range(self.k)]
        ax.scatter([v[0].i for v in p_i_u],
                   [v[0].j for v in p_i_u],
                   s=4,
                   color="violet",
                   marker="o",
                   label="p_i(u)")
        for j in range(1, self.k):
            ax.annotate("p_{}(u)".format(j), (p_i_u[j][0].i, p_i_u[j][0].j),
                        xytext=(5, 5),
                        textcoords="offset pixels",
                        color="violet")

        # For the current v, highlight its batch B(v) in a different color
        B_v = [w for w in self.B[v]]
        ax.scatter([w.i for w in B_v],
                   [w.j for w in B_v],
                   s=4,
                   color="lime",
                   marker="*",
                   label="B(v)")

        # Draw line from u to current w
        ax.add_line(Line2D([u.i, w.i], [u.j, w.j], color="pink"))

        # For the final plot, draw a line from w to current v as well
        if final:
            ax.add_line(Line2D([w.i, v.i], [w.j, v.j], color="palegreen"))

        title = "Iteration {} (final)".format(i) \
            if final else "Iteration {}".format(i)

        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        fig.show()


if __name__ == "__main__":
    g = Graph.create_random_graph(
        num_vertices=250,
        x_range=(-50, 50),
        y_range=(-50, 50),
        seed=1
    )
    a = ADO(g, 5)
    a.animate_query(g.vertices[0], g.vertices[10])
