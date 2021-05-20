import math
import random
from abc import ABC, abstractmethod
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import time
from tqdm import tqdm


class Vertex(ABC):
    """
    Generalized abstract class representing a vertex (either a point in
    real space or a node in a graph.
    """

    @property
    @abstractmethod
    def idx(self):
        return None

    @abstractmethod
    def __repr__(self):
        return None


class FiniteMetricVertex(Vertex):
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
        self._idx = idx

    def __repr__(self) -> str:
        return "({},{})".format(self.i, self.j)

    @property
    def idx(self):
        return self._idx


class ADO(ABC):
    def __init__(self, k: int):
        self.A = None
        self.p = None
        self.B = None
        self.n = None
        self.k = k

    @property
    @abstractmethod
    def vertices(self):
        pass

    def preprocess(self) -> None:
        """
        Do all preprocessing on the graph to answer queries.
        Runs in O(n^2) time.

        Saves the required preprocessed data in instance variables.
        """

        # Sample k + 1 sets A_0 through A_k
        A_sets = self.__sample_A_sets()
        while not A_sets[self.k - 1]:  # If A_{k-1} is empty, re-sample
            A_sets = self.__sample_A_sets()
        self.A = A_sets

        # Compute distances between each point and each set A_i, saving
        # the closest points p_i(v)
        self.p = self.compute_p()

        # Compute a bunch and associated distance table for each vertex
        self.B = self.compute_bunches()

    def __sample_A_sets(self) -> List[Set[Vertex]]:
        """
        Constructs k+1 sets A_0 through A_k by repeated sampling with
        probability n^{-1/k} from the previous set. Note that A_0 = V and A_k
        = {}.

        :return: A list of sets A_0 through A_k in that order, where each set
        is a set of Vertex objects.
        """
        # Add set A_0 = V
        A_sets = [set(self.vertices)]

        # Add sets A_1 through A_{k-1} by sampling
        for _ in range(1, self.k):
            vertices = set([v for v in A_sets[-1]
                            if random.random() < (self.n ** (-1. / self.k))])
            A_sets.append(vertices)

        # Add set A_k = {}
        A_sets.append(set())

        return A_sets

    @abstractmethod
    def compute_p(self) -> DefaultDict[Vertex, Dict[int, Tuple[Vertex,
                                                               float]]]:
        pass

    @abstractmethod
    def compute_bunches(self) -> DefaultDict[Vertex, Dict[Vertex,
                                                          float]]:
        pass

    def query(self,
              u: Vertex,
              v: Vertex,
              return_w: bool = False) -> Union[float, Tuple[float, Vertex]]:
        """
        Query the ADO for the estimated distance between two vertices u and v.

        :param u: A vertex in the ADO's graph self.g
        :param v: A different vertex in the ADO's graph self.g
        :param return_w: If this is True, in addition to the distance, returns
        the final point w.
        :return: The estimated distance between u and v, plus optionally the
        Vertex w if return_w is True.
        """

        w = u
        i = 0

        while w not in self.B[v]:
            i += 1
            u, v = v, u
            w = self.p[u][i][0]

        distance = self.B[u][w] + self.B[v][w]

        if return_w:
            return distance, w
        else:
            return distance


class PointADO(ADO):
    """
    An implementation of the approximate distance oracle algorithm/data
    structure.
    """

    def __init__(self, vertices: List[FiniteMetricVertex], k: int):
        """
        :param vertices: A list of real 2D points (FiniteMetricVertex objects)
        that form the graph that ADO will run on.
        :param k: The tunable ADO parameter that controls the data structure
        size, query runtime, and stretch of the estimates.
        """
        super().__init__(k=k)
        self._vertices = vertices
        self.n = len(self.vertices)

        # Preprocess and construct A, p, B
        self.preprocess()

    @property
    def vertices(self):
        return self._vertices

    @staticmethod
    def distance(u: FiniteMetricVertex, v: FiniteMetricVertex):
        return math.sqrt((u.i - v.i) ** 2 + (u.j - v.j) ** 2)

    def compute_p(self) -> DefaultDict[Vertex,
                                       Dict[int, Tuple[Vertex,
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
        for v in self.vertices:
            for i, A_i in enumerate(self.A):
                p_i = None
                delta = float("inf")

                for u in A_i:
                    if u == v:
                        continue

                    d = self.distance(v, u)
                    if d < delta:
                        p_i = u
                        delta = d

                p_dict[v][i] = (p_i, delta)

        return p_dict

    def compute_bunches(self) -> DefaultDict[Vertex, Dict[Vertex,
                                                          float]]:
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
        for v in self.vertices:
            for i, A_i in enumerate(self.A[:self.k]):
                A_i_plus_1 = self.A[i + 1]
                candidates = A_i.difference(A_i_plus_1)
                for w in candidates:
                    d = self.distance(v, w)
                    if d < self.p[v][i + 1][1]:
                        bunches[v][w] = d

        return bunches

    def random_query(self):

        u = self.vertices[0] #np.random.choice(self.vertices)
        v = self.vertices[-1] #np.random.choice(self.vertices)

        return self.query(u, v)


    def query(self, u: FiniteMetricVertex, v: FiniteMetricVertex):
        w = u
        i = 0

        while w not in self.B[v]:
            i += 1
            u, v = v, u
            w = self.p[u][i][0]

        return self.distance(u, w) + self.distance(w, v)


    def animate_query(self, u: FiniteMetricVertex,
                      v: FiniteMetricVertex, timestep: float = 2.0) -> None:
        """
        Query the ADO for the estimated distance between two vertices u and v
        and animate/plot the process with pyplot.

        :param u: A vertex in the ADO's graph self.g
        :param v: A different vertex in the ADO's graph self.g
        :param timestep: The timestep between frames in the animation
        """
        # Check that the vertices are actually from this graph
        assert self.vertices[u.idx] == u and self.vertices[v.idx] == v, \
            "The vertices being queried must be in the graph."

        fig, ax = plt.subplots()

        #self.__plot_A_i(fig, ax)
        #fig, ax = plt.subplots()

        #self.__plot_p_i(fig, ax, u)
        #fig, ax = plt.subplots()

        self.__plot_bunches(fig, ax, u)
        fig, ax = plt.subplots()
        self.__plot_bunches(fig, ax, v)
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

    def __plot_A_i(self, fig: plt.Figure, ax: plt.Axes):

        for i, a_i in enumerate(self.A):
            ax.scatter(
                [v.i for v in a_i],
                [v.j for v in a_i],
                label="A_{}".format(i)
            )

        plt.legend(loc=(1.04, 0))
        ax.set_title("A_i")
        plt.savefig("partitions.png", bbox_inches="tight")


    def __plot_p_i(self, fig: plt.Figure, ax: plt.Axes, u: FiniteMetricVertex):

        # Plot all the points in the graph
        for i, a_i in enumerate(self.A):
            ax.scatter(
                [v.i for v in a_i],
                [v.j for v in a_i],
                label="A_{}".format(i)
            )

        ax.scatter([u.i], [u.j], color='red', label='u')

        for i, p_i in self.p[u].items():
            if p_i[0] is None:
                continue

            w = p_i[0]
            d = p_i[1]

            ax.scatter([w.i], [w.j], label='p_{}(u)'.format(i))

            circ = plt.Circle((u.i, u.j), d, fill=False)

            ax.add_patch(circ)

        plt.legend(loc=(1.04, 0))
        ax.set_title("p_i(u)")
        plt.savefig("witnesses.png", bbox_inches="tight")

    def __plot_bunches(self, fig: plt.Figure, ax: plt.Axes, u: FiniteMetricVertex):
        # Plot all the points in the graph
        for i, a_i in enumerate(self.A):
            ax.scatter(
                [v.i for v in a_i],
                [v.j for v in a_i],
                label="A_{}".format(i)
            )

        ax.scatter([u.i], [u.j], color='red', label='u')

        for i, p_i in self.p[u].items():
            if p_i[0] is None:
                continue

            w = p_i[0]
            d = p_i[1]

            ax.scatter([w.i], [w.j], label='p_{}(u)'.format(i))

            circ = plt.Circle((u.i, u.j), d, fill=False)

            ax.add_patch(circ)

        ax.scatter(
            [v.i for v in self.B[u]],
            [v.j for v in self.B[u]],
            label='B(u)', color='black'
        )

        plt.legend(loc=(1.04, 0))
        ax.set_title("B(u)")

        plt.savefig("bunches.png", bbox_inches="tight")
        #plt.show()


    def __plot_query_state(self, fig: plt.Figure,
                           ax: plt.Axes,
                           u: FiniteMetricVertex, v: FiniteMetricVertex,
                           w: FiniteMetricVertex, i: int,
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
        ax.scatter([v.i for v in self.vertices],
                   [v.j for v in self.vertices],
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


def sample_random_real_points(num_vertices: int,
                              x_range: Tuple[float, float] = (-100, 100),
                              y_range: Tuple[float, float] = (-100, 100),
                              seed: int = 0) -> List[FiniteMetricVertex]:
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
    random_vertices = [FiniteMetricVertex(i, j, idx=n) for n, (i,
                                                               j) in enumerate(
        zip(x_coords, y_coords))]

    return random_vertices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime_test", action='store_true')
    parser.add_argument("--num_vertices", type=int, default=250)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    if args.runtime_test:

        for k in range(1, args.k+1):
            times_preproc = []
            times_query = []
            num_pts = []
            for i in tqdm(range(10, args.num_vertices, 4)):
                points = sample_random_real_points(
                    num_vertices=i,
                    x_range=(-50, 50),
                    y_range=(-50, 50),
                    seed=args.seed)
                start = time.time()
                point_ado = PointADO(points, k)
                times_preproc.append(time.time() - start)
                num_pts.append(i)

                #start = time.time()
                #for _ in range(100):
                #    _ = point_ado.random_query()
                #times_query.append((time.time() - start) / 100.)

            #plt.scatter(num_pts, times_preproc, s=3)


            fit = np.polyfit(num_pts, times_preproc, 2)
            best_fit = np.poly1d(fit)

            yhat = best_fit(num_pts)
            ybar = np.mean(times_preproc)
            ssreg = np.sum((yhat-ybar)**2)
            sstot = np.sum((times_preproc - ybar)**2)
            r2 = ssreg / sstot
            print("R Squared Value Preprocess: {}".format(r2))
            print(fit)
            plt.plot(num_pts, best_fit(num_pts), label="k={}".format(k))

        plt.legend()
        plt.title("Preprocessing runtimes for different numbers of points")
        plt.show()

        #plt.scatter(num_pts, times_query)


        #plt.title("Query runtimes for different numbers of points")
        #plt.show()

    else:
        num_graphs = 0
        max_stretch = 0
        while True:
            num_graphs += 1
            points = sample_random_real_points(
                num_vertices=args.num_vertices,
                x_range=(-50, 50),
                y_range=(-50, 50),
                seed=args.seed)
            point_ado = PointADO(points, args.k)

            for ii in range(args.num_vertices):
                for jj in range(args.num_vertices):
                    if ii == jj:
                        continue
                    res = point_ado.query(point_ado.vertices[ii], point_ado.vertices[jj])
                    truth = point_ado.distance(point_ado.vertices[ii], point_ado.vertices[jj])

                    stretch = res / truth

                    if stretch > max_stretch:
                        plt.close('all')
                        max_stretch = stretch
                        print(stretch, num_graphs)
                        point_ado.animate_query(point_ado.vertices[ii], point_ado.vertices[jj])

                        if stretch >= 2. * args.k - 1.0 - 0.01:
                            while True:
                                plt.draw()
                                plt.pause(0.01)
