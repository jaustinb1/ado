import math
import random
from abc import ABC, abstractmethod
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from tqdm import tqdm
from scipy.optimize import curve_fit


class Vertex(ABC):
    """
    Generalized abstract class representing a vertex (either a point in
    real space or a node in a graph.
    """

    def __init__(self, idx: Optional[int] = None):
        self._idx = idx

    @property
    def idx(self):
        """
        Index of the point in its enclosing graph to uniquely identify it.
        """
        return self._idx

    @abstractmethod
    def __repr__(self):
        pass


class FiniteMetricVertex(Vertex):
    """
    A representation of a single point in a finite metric space.
    """

    def __init__(self, i: float, j: float, stddev: float = 0.1,
                 idx: Optional[int] = None):
        """
        :param i: The x-coordinate (mean) of the point.
        :param j: The y-coordinate (mean) of the point.
        :param stddev: Standard deviation of the normal distribution the
        point is sampled from.
        :param idx: (Optional) Index of the point in the graph for bookkeeping.
        """
        super().__init__(idx)
        self.i = np.random.normal(loc=i, scale=stddev)
        self.j = np.random.normal(loc=j, scale=stddev)

    def __repr__(self) -> str:
        return "({},{})".format(self.i, self.j)

    def dist_to(self, v):
        return math.sqrt(
            (self.i - v.i)**2 + (self.j - v.j)**2
        )


class ADO(ABC):
    """
    Abstract class representing a general approximate distance oracle (ADO)
    implementation.

    Actual ADO implementations are concrete classes inheriting from this class.
    """

    def __init__(self, k: int, seed: Optional[int] = None):
        # All variables are from the paper, see method docstrings
        # for detailed descriptions
        self.A = None
        self.p = None
        self.B = None

        self.k = k
        self.seed = seed
        if self.seed:
            random.seed(self.seed)

    @property
    @abstractmethod
    def vertices(self):
        pass

    @property
    def n(self):
        return len(self.vertices)

    def preprocess(self) -> None:
        """
        Do all preprocessing on the graph to answer queries.
        Runs in O(n^2) time.

        Saves the required preprocessed data in instance variables.
        """

        # Sample k + 1 sets A_0 through A_k
        A_sets = self.sample_A_sets()
        while not A_sets[self.k - 1]:  # If A_{k-1} is empty, re-sample
            A_sets = self.sample_A_sets()
        self.A = A_sets
        self.differences = { i: self.A[i].difference(self.A[i+1]) for i in range(self.k)}


        # Compute distances between each point and each set A_i, saving
        # the closest points p_i(v)
        self.p = self.compute_p()

        # Compute a bunch and associated distance table for each vertex
        self.B = self.compute_bunches()

    def sample_A_sets(self) -> List[Set[Vertex]]:
        """
        Constructs k+1 sets A_0 through A_k by repeated sampling with
        probability n^{-1/k} from the previous set. Note that A_0 = V and A_k
        = {}.

        :return: A list of sets A_0 through A_k in that order, where each set
        is a set of Vertex objects.
        """
        # Add set A_0 = V
        A_sets = [set(self.vertices)]

        # Add sets A_1 through A_{k-1} by sampling from the previous
        # set with probability n^{-1/k} for each vertex.
        for _ in range(1, self.k):
            # If we are seeding, we fix the order of set iteration by sorting
            # and then shuffling. This ensures that the entire sampling process
            # is fully deterministic.
            # This also adds an O(n log n) contribution to the runtime,
            # so it should be disabled for accurate runtime measurements
            if self.seed:
                A_prev = list(sorted(A_sets[-1], key=lambda v: v.idx))
                random.shuffle(A_prev)
            else:
                A_prev = A_sets[-1]
            vertices = set([v for v in A_prev
                            if random.random() < (self.n ** (-1. / self.k))])
            A_sets.append(vertices)

        # Add set A_k = {}
        A_sets.append(set())

        return A_sets

    @abstractmethod
    def compute_p(self) -> DefaultDict[Vertex, Dict[int, Tuple[Vertex,
                                                               float]]]:
        """
        Computes the distance from each vertex v to each set A_i, and notes
        the nearest point in A_i to v (p_i(v)).

        :return: A dictionary mapping from a Vertex v to a dictionary, which
        in turn maps from an index i (from 0 to k) corresponding to a set A_i to
        a tuple of (p_i(v), distance(v, A_i)).
        """
        pass

    @abstractmethod
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
        final Vertex w if return_w is True.
        """

        w = u
        i = 0

        while w not in self.B[v]:
            i += 1
            u, v = v, u
            w = self.p[u][i][0]

        distance = self.B[u][w] + self.B[v][w]

        return (distance, w) if return_w else distance


class PointADO(ADO):
    """
    An implementation of the approximate distance oracle algorithm/data
    structure on 2D points.
    """

    def __init__(self, vertices: List[FiniteMetricVertex], k: int,
                 seed: Optional[int] = None):
        """
        :param vertices: A list of real 2D points (FiniteMetricVertex objects)
        that form the graph that ADO will run on.
        :param k: The tunable ADO parameter that controls the data structure
        size, query runtime, and stretch of the estimates.
        """
        super().__init__(k=k, seed=seed)
        self._vertices = vertices

        # Preprocess and construct A, p, B
        self.preprocess()

    @property
    def vertices(self):
        return self._vertices

    @staticmethod
    def distance(u: FiniteMetricVertex, v: FiniteMetricVertex):
        return u.dist_to(v)


    def compute_p(self) -> DefaultDict[Vertex,
                                       Dict[int, Tuple[Vertex,
                                                       float]]]:
        p_dict = {}

        # Compute minimum distance from every point v to every set A_i
        # Save the closest point in the set p_i(v) in the dict p_dict
        for v in self.vertices:
            p_dict[v] = {}
            for i in range(self.k + 1):
                p_i_v = None
                d_v_Ai = float("inf")

                for w in self.A[i]:
                    d_w_v = self.distance(w, v)
                    if d_w_v < d_v_Ai:
                        p_i_v = w
                        d_v_Ai = d_w_v

                p_dict[v][i] = (p_i_v, d_v_Ai)

        return p_dict

    def compute_bunches(self) -> DefaultDict[Vertex, Dict[Vertex,
                                                          float]]:
        bunches = {}
        for v in self.vertices:
            bunches[v] = {}
            for i in range(self.k):
                for w in self.differences[i]:
                    d_v_w = self.distance(v, w)
                    d_v_A_i_plus_1 = self.p[v][i + 1][1]
                    if d_v_w < d_v_A_i_plus_1:
                        bunches[v][w] = d_v_w

        return bunches

    def animate_query(self, u: FiniteMetricVertex,
                      v: FiniteMetricVertex, timestep: float = 2.0,
                      save: bool = False) -> None:
        """
        Query the ADO for the estimated distance between two vertices u and v
        and animate/plot the process with pyplot.

        :param u: A vertex in the ADO's graph self.g
        :param v: A different vertex in the ADO's graph self.g
        :param timestep: The timestep between frames in the animation
        :param save: If true, saves each of the generated images as PNGs.
        """
        # Check that the vertices are actually from this graph
        assert self.vertices[u.idx] == u and self.vertices[v.idx] == v, \
            "The vertices being queried must be in the graph."

        fig, ax = plt.subplots()

        # Plot preprocessing data
        self.__plot_A_i(fig, ax)
        if save:
            fig.savefig('A_i.png')
        plt.pause(timestep)

        self.__plot_p_i(fig, ax, u, name="u")
        if save:
            fig.savefig('p_i_u.png')
        plt.pause(timestep)
        self.__plot_p_i(fig, ax, v, name="v")
        if save:
            fig.savefig('p_i_v.png')
        plt.pause(timestep)

        self.__plot_bunches(fig, ax, u, name="u")
        if save:
            fig.savefig('B_u.png')
        plt.pause(timestep)
        self.__plot_bunches(fig, ax, v, name="v")
        if save:
            fig.savefig('B_v.png')
        plt.pause(timestep)

        # Plot the query steps
        w = u
        i = 0

        self.__plot_query_state(fig, ax, u, v, w, i)
        if save:
            fig.savefig('iter_0.png')
        plt.pause(timestep)

        while w not in self.B[v]:
            i += 1
            u, v = v, u
            w = self.p[u][i][0]

            self.__plot_query_state(fig, ax, u, v, w, i)
            if save:
                fig.savefig('iter_{}.png'.format(i))
            plt.pause(timestep)

        self.__plot_query_state(fig, ax, u, v, w, i, final=True)
        if save:
            fig.savefig('iter_{}_final.png'.format(i))

    def __plot_A_i(self, fig: plt.Figure, ax: plt.Axes) -> None:
        """
        Plot all points and highlight the sampled sets A_i.

        :param fig: The matplotlib figure to plot on.
        :param ax: The matplotlib axes to plot on.
        """
        ax.cla()

        colors = get_cmap("Dark2").colors
        ax.set_prop_cycle(color=colors)
        for i, a_i in enumerate(self.A):
            ax.scatter(
                [v.i for v in a_i],
                [v.j for v in a_i],
                s=8,
                marker="o",
                label="A_{}".format(i))

        ax.set_title("Sampled sets A_i")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        fig.show()

    def __plot_p_i(self, fig: plt.Figure, ax: plt.Axes,
                   point: FiniteMetricVertex, name: str = "u") -> None:
        """
        Plot all points and highlight the witnesses p_i for the given point
        along with corresponding rings on the given figure and axes.

        :param fig: The matplotlib figure to plot on.
        :param ax: The matplotlib axes to plot on.
        :param point: The vertex whose witnesses/rings we wish to plot.
        :param name: The name to use to label the vertex/bunches.
        """
        ax.cla()

        # Plot all points and color by set A_i
        for i, a_i in enumerate(self.A):
            ax.scatter(
                [v.i for v in a_i],
                [v.j for v in a_i],
                s=8,
                marker="o",
                label="A_{}".format(i)
            )

        # Plot and label the point itself
        ax.scatter([point.i], [point.j],
                   s=12,
                   color="red",
                   marker="*",
                   label=name)
        ax.annotate(name, (point.i, point.j), color="red")

        # Force the xlim and ylim to become fixed
        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())

        # For the current point, mark and label its p_i s
        # and add circles
        p_i = [self.p[point][i] for i in range(self.k)]
        for i in range(1, self.k):
            if p_i[i] is None:
                continue
            ax.annotate("p_{}({})".format(i, name), (p_i[i][0].i, p_i[i][
                0].j),
                        xytext=(5, 5),
                        textcoords="offset pixels",
                        color="violet")
            circ = plt.Circle((point.i, point.j), p_i[i][1], fill=False)
            ax.add_patch(circ)

        ax.set_title("Witnesses p_i({}) and rings.".format(name))
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        fig.show()

    def __plot_bunches(self, fig: plt.Figure, ax: plt.Axes,
                       point: FiniteMetricVertex, name: str = "u") -> None:
        """
        Plot all points and highlight the bunches for the given point on
        the provided figure/axes.

        :param fig: The matplotlib figure to plot on.
        :param ax: The matplotlib axes to plot on.
        :param point: The vertex whose bunches we wish to plot.
        :param name: The name to use to label the vertex/bunches.
        """
        ax.cla()

        # Plot all points and color by set A_i
        ax.scatter([v.i for v in self.vertices],
                   [v.j for v in self.vertices],
                   s=4,
                   color="black",
                   marker=".",
                   label="Points")

        # Plot and label the point itself
        ax.scatter([point.i], [point.j],
                   s=12,
                   color="red",
                   marker="*",
                   label=name)
        ax.annotate(name, (point.i, point.j), color="red")

        # Force the xlim and ylim to become fixed
        ax.set_xlim(*ax.get_xlim())
        ax.set_ylim(*ax.get_ylim())

        # For the current point, mark and label its p_i s
        # and add circles
        p_i = [self.p[point][i] for i in range(self.k)]
        for i in range(1, self.k):
            if p_i[i] is None:
                continue
            ax.annotate("p_{}({})".format(i, name), (p_i[i][0].i, p_i[i][
                0].j),
                        xytext=(5, 5),
                        textcoords="offset pixels",
                        color="violet")
            circ = plt.Circle((point.i, point.j), p_i[i][1], fill=False)
            ax.add_patch(circ)

        # Plot the points in the bunch
        B = [w for w in self.B[point]]
        ax.scatter([w.i for w in B],
                   [w.j for w in B],
                   s=12,
                   color="lime",
                   marker="*",
                   label="B({})".format(name))

        ax.set_title("Bunch B({})".format(name))
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        fig.show()

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

    :param num_vertices: The number of vertices (points) to sample.
    :param x_range: The range (low, high) of x coordinates to sample
    points from.
    :param y_range: The range (low, high) of y coordinates to sample
    points from.
    :param seed: The random seed to use for generating the data.

    :return: A list of num_vertices random points as FiniteMetricVertex objects,
    which can be passed as an input to a PointADO instance.
    """
    random.seed(seed)
    np.random.seed(seed)

    x_coords = [random.uniform(x_range[0], x_range[1])
                for _ in range(num_vertices)]
    y_coords = [random.uniform(y_range[0], y_range[1])
                for _ in range(num_vertices)]
    random_vertices = [FiniteMetricVertex(i, j, idx=n)
                       for n, (i, j) in enumerate(zip(x_coords, y_coords))]

    return random_vertices

def sweep_per_k(kk, max=250, stride=64):
    k, ii = kk

    vals = []
    mean_stretch = []
    ns = []
    totals = []

    if k == 2 and ii == 0:
        ff = lambda x: tqdm(range(*x))
    else:
        ff = lambda x: range(*x)

    for n in ff((k, max, stride)):
        all_pairs = []

        points = sample_random_real_points(
            num_vertices=n,
            x_range=(-50, 50),
            y_range=(-50, 50),
            seed=0)
        point_ado = PointADO(points, k)

        seen_pairs = set()
        total = 0
        for u in point_ado.vertices:
            for v in point_ado.vertices:
                if v == u:
                    continue
                if (u,v) in seen_pairs or (v, u) in seen_pairs:
                    continue

                seen_pairs.add((u,v))

                true_distance = u.dist_to(v)

                ado_distance = point_ado.query(u, v)

                stretch = ado_distance / true_distance

                vals.append(stretch)
                all_pairs.append(stretch)
                total += 1
        totals.append(total)
        mean_stretch.append(np.sum(all_pairs))
        ns.append(n)

    #plt.figure()
    #plt.hist(vals)
    #plt.title("k={}".format(k))
    #plt.show()
    return k, np.array(vals), ns, np.array(mean_stretch), np.array(totals)

def fit_fn(x, lam):
    # exponential distribution pdf
    return lam * np.exp(- lam * x)

def fn(k):
    max_val = 2048

    return sweep_per_k(k, max=max_val)


if __name__ == "__main__":

    from multiprocessing import Pool

    """
    max_val = 2048
    v1, nn, m1 = sweep_per_k(2, max=max_val)
    v2, _, m2 = sweep_per_k(3, max=max_val)
    v3, _, m3 = sweep_per_k(4, max=max_val)
    v4, _, m4 = sweep_per_k(5, max=max_val)
    v5, _, m5 = sweep_per_k(6, max=max_val)
    """
    with Pool(15) as pool:
        jobs = []
        for i in range(2):
            for j in [2,3,4,5,6]:
                jobs.append((j, i))
        res = pool.map(fn, jobs)

    counts = defaultdict(lambda: [])
    means = defaultdict(lambda: 0)
    totals = defaultdict(lambda: 0)
    nn = None
    for (k, v, n, m, tots) in res:
        if nn is None:
            nn = n

        counts[k].extend(v)
        means[k] += m
        totals[k] += tots

    for k in means:
        means[k] /= totals[k]

    plt.figure()
    plt.hist([counts[k] for k in counts], label=[k for k in counts], density=True, bins=50)
    plt.legend()
    plt.yscale('log')

    plt.figure()
    for k in means:
        plt.plot(nn, means[k], label=str(k))
    plt.legend()

    for k in counts:
        np_hist, bins = np.histogram(counts[k], density=True)
        fit, _ = curve_fit(lambda x, lam: lam * k * np.exp(- lam * k * x), bins[:-1], np_hist)
        print(k, fit)

    plt.show()

    nn = res[0][1]
    v1 = res[0][0]
    v2 = res[1][0]
    v3 = res[2][0]
    v4 = res[3][0]
    v5 = res[4][0]

    np_hist1, b1 = np.histogram(v1, density=True)
    np_hist2, b2 = np.histogram(v2, density=True)
    np_hist3, b3 = np.histogram(v3, density=True)
    np_hist4, b4 = np.histogram(v4, density=True)
    np_hist5, b5 = np.histogram(v5, density=True)

    fit1, _ = curve_fit(lambda x, lam: lam * 2 * np.exp(- lam * 2. * x), b1[:-1], np_hist1)
    fit2, _ = curve_fit(lambda x, lam: lam * 3 * np.exp(- lam * 3. * x), b2[:-1], np_hist2)
    fit3, _ = curve_fit(lambda x, lam: lam * 4 * np.exp(- lam * 4. * x), b3[:-1], np_hist3)
    fit4, _ = curve_fit(lambda x, lam: lam * 5 * np.exp(- lam * 5. * x), b4[:-1], np_hist4)
    fit5, _ = curve_fit(lambda x, lam: lam * 6 * np.exp(- lam * 6. * x), b5[:-1], np_hist5)

    print(fit1)
    print(fit2)
    print(fit3)
    print(fit4)
    print(fit5)



    plt.figure()
    plt.hist([v1,v2,v3,v4,v5], label=['2','3','4', '5', '6'], density=True)
    plt.legend()
    plt.yscale('log')

    plt.figure()
    plt.plot(nn, m1, label='2')
    plt.plot(nn, m2, label='3')
    plt.plot(nn, m3, label='4')
    plt.plot(nn, m4, label='5')
    plt.plot(nn, m5, label='6')

    plt.legend()
    plt.show()
    exit()

    points = sample_random_real_points(
        num_vertices=250,
        x_range=(-50, 50),
        y_range=(-50, 50),
        seed=0)
    point_ado = PointADO(points, 5, seed=6)
    query_points = random.sample(point_ado.vertices, 2)
    point_ado.animate_query(query_points[0], query_points[1],
                            save=True,
                            timestep=2.0)
