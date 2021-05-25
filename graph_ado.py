import itertools
import random
from collections import defaultdict
from heapq import heappop, heappush
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import networkx as nx
from matplotlib import pyplot as plt

from ado import Vertex, ADO


class GraphVertex(Vertex):
    """
    A representation of a graph node. Essentially a wrapper around a NetworkX
    graph node object, with the addition of an index identifying the node in
    the graph.
    """

    def __init__(self, n: Any, idx: Optional[int] = None):
        """
        :param n: The corresponding NetworkX node identifier.
        :param idx: (Optional) Index of the point in the graph for bookkeeping
        """
        super().__init__(idx)
        self.nx_node = n

    def __repr__(self) -> str:
        return "(idx={}, nx_node={})".format(self.idx, self.nx_node)


class GraphADO(ADO):
    """
    An implementation of the ADO algorithm/data structure on an undirected,
    weighted graph.
    """

    def __init__(self, k: int, graph: nx.Graph):
        """
        :param k: The stretch parameter for the ADO algorithm.
        :param graph: A NetworkX graph to apply the ADO algorithm to.
        """
        super().__init__(k)
        self.g = graph

        # Get all vertices from the NetworkX graph and wrap them as
        # Vertex objects to provide a uniform interface with the non-graph
        # ADO version.
        self._vertices = [GraphVertex(n=nx_node, idx=i) for i, nx_node in
                          enumerate(
                              self.g.nodes)]
        # Add a mapping from each NetworkX node to its corresponding Vertex
        # for easy translation back and forth.
        self._nx_node_to_vertex = {v.nx_node: v for v in self._vertices}

        # The clusters used for calculating the bunches in the ADO algorithm.
        self.C = None

        self.preprocess()

    @property
    def vertices(self):
        return self._vertices

    def distance(self, u: GraphVertex, v: GraphVertex):
        """
        Get the (weighted) distance between two vertices from the NetworkX
        graph.
        """
        return self.g[u.nx_node][v.nx_node]['weight']

    def __custom_dijkstras(self,
                           graph: nx.Graph,
                           source: GraphVertex,
                           A_i_plus_1_dists: Dict[Vertex, float]) -> Tuple[
        Dict[GraphVertex, int],
        Dict[GraphVertex, List[GraphVertex]]
    ]:
        """
        An helper implementation of the modified Dijkstra's single-source
        shortest paths algorithm used as a subroutine of the graph ADO
        pre-processing.

        The modification is that we only relax the distances of nodes on the
        frontier if the new distance is shorter than the distance to A_{i+1}.

        :param graph: The NetworkX graph that the algorithm computes the
        distances for.
        :param source: The source Vertex, which must be in the graph.
        :param A_i_plus_1_dists: A dictionary of distances from each node to
        the set A_{i+1}.
        :return: A tuple consisting of: a distance dictionary mapping from each
        vertex to its distance from the source, and a path dictionary of the
        shortest path from the source to each vertex. Note that both the
        distance and path dictionaries only contain keys for vertices which
        satisfy the custom condition that they are closer to the source than
        to A_{i+1}.
        """

        def backtrace(prev: Dict[GraphVertex, GraphVertex],
                      start: GraphVertex,
                      end: GraphVertex) -> List[GraphVertex]:
            """
            Follows the saved backpointers in the prev dictionary to reconstruct
            the shortest distance path from start to end.

            :param prev: A dictionary of backpointers mapping from every vertex
            to its parent along the shortest path from the source.
            :param start: The start vertex of the path.
            :param end: The end vertex of the path.
            :return: The shortest path found by the custom Dijkstra's algorithm
            as a list of GraphVertices.
            """
            node = end
            path = []
            while node != start:
                path.append(node)
                node = prev[node]
            path.append(node)
            path.reverse()
            return path

        # Priority queue operations, from the Python documentation recommended
        # implementation.
        # See https://docs.python.org/3/library/heapq.html#priority-queue
        # -implementation-notes
        frontier = []
        heap_nodes = {}
        REMOVED = '<removed>'  # marker for a removed entry
        # This is a hack to break ties/ensure stable ordering of elements
        counter = itertools.count()

        def pq_push(pq, node, priority):
            """
            Add a new entry or update the priority of an existing entry.
            """
            if node in heap_nodes:
                pq_remove(node)
            count = next(counter)
            entry = [priority, count, node]
            heap_nodes[node] = entry
            heappush(pq, entry)

        def pq_remove(node):
            """
            Mark an existing entry as REMOVED.
            Raise KeyError if not found.
            """
            entry = heap_nodes.pop(node)
            entry[-1] = REMOVED

        def pq_pop(pq):
            """
            Remove and return the lowest priority entry. Raise KeyError if
            empty.
            """
            while pq:
                priority, count, node = heappop(pq)
                if node is not REMOVED:
                    del heap_nodes[node]
                    return priority, node
            raise KeyError('Pop from an empty priority queue.')

        # Setup for Dijkstra's algorithm

        distances = defaultdict(lambda: float("inf"))
        prev_node = {}
        seen = set()

        distances[source] = 0
        for v in self.vertices:
            pq_push(frontier, v, distances[v])

        # Dijkstra's algorithm
        while heap_nodes:
            current_distance, u = pq_pop(frontier)
            if u in seen:
                continue
            seen.add(u)
            for v in graph.neighbors(u.nx_node):
                v = self._nx_node_to_vertex[v]
                if v in seen:
                    continue
                dist_through_u = distances[u] + self.distance(u, v)
                # The modification from the standard Dijkstra's SSSP algorithm
                # is here: in addition to checking whether the path through u
                # is shorter, we also check that the path through u
                # is shorter than the distance from v to A_{i+1}
                if dist_through_u < distances[v] and dist_through_u < \
                        A_i_plus_1_dists[v]:
                    distances[v] = dist_through_u
                    prev_node[v] = u
                    # Update the existing entry in the priority queue
                    # with the new distance
                    pq_push(frontier, v, distances[v])

        # Get only points with finite distances and reconstruct paths
        distances = {v: distances[v] for v in distances if distances[v] <
                     float("inf")}
        paths = {v: backtrace(prev_node, source, v) for v in distances}
        return distances, paths

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
        # We record the previous distance delta(A_{i+1}, v) for use in checking
        # the condition on pg. 16 in the paper
        self.C = defaultdict(lambda: {})

        for i in range(self.k - 1, -1, -1):
            # d(A_{i+1}, v) for all v
            if i == self.k - 1:
                delta_A_i_plus_1_v = defaultdict(lambda: float("inf"))
            else:
                delta_A_i_plus_1_v = {v: p_dict[v][i + 1][1] for v in p_dict}

            # Compute the distances from A_i to each other vertex v
            distances, paths = nx.multi_source_dijkstra(self.g,
                                                        set(v.nx_node
                                                            for
                                                            v in
                                                            self.A[i]))

            # Compute distances and our witnesses p_i_v
            for v in self.vertices:
                # We check if the distance is the same as to the previous
                # set (A_{i+1}), and if so re-use the same witness p_{i+1}(v)
                delta_A_i_v = distances[v.nx_node]
                if delta_A_i_plus_1_v[v] == delta_A_i_v:
                    p_i_v = p_dict[v][i + 1][0]
                else:
                    p_i_v = self._nx_node_to_vertex[paths[v.nx_node][0]]

                assert p_i_v in self.A[i]

                p_dict[v][i] = (p_i_v, delta_A_i_v)

            # Compute clusters C(w) and corresponding shortest path
            # trees T(w) in the form of a saved shortest path from each
            # w to each vertex in C(w)
            for w in self.A[i].difference(self.A[i + 1]):
                # Do a modified Dijkstra's with the stricter relaxation
                # condition
                # This will give distances to all points in the cluster C(w)
                # and can also be modified to give the tree (paths).
                C_distances, C_paths = self.__custom_dijkstras(
                    self.g,
                    w,
                    delta_A_i_plus_1_v
                )
                self.C[w].update({
                    v: (C_distances[v], C_paths[v]) for v in C_distances
                })

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
            for w in self.vertices:
                if v in self.C[w]:
                    bunches[v][w] = self.C[w][v][0]

        return bunches

    def __get_path(self, u: Vertex, v: Vertex, w: Vertex) -> List[Vertex]:
        """
        Given the source and target vertices (u and v) and the final w vertex
        computed by the query algorithm, construct and return the corresponding
        approximate shortest path between u and v as a list of Vertices.
        """
        path_u_to_w = list(reversed(self.C[w][u][1]))
        path_v_to_w = list(reversed(self.C[w][v][1]))

        # Find the least common ancestor along the two paths (it could be
        # a vertex before w, or it could be w).
        if len(path_u_to_w) <= len(path_v_to_w):
            shorter = path_u_to_w
            longer = path_v_to_w
        else:
            shorter = path_v_to_w
            longer = path_u_to_w

        shorter_dict = {vertex: j for j, vertex in enumerate(shorter)}

        lca = None
        i = 0
        j = 0
        while i < len(longer):
            vertex = longer[i]
            if vertex in shorter_dict:
                j = shorter_dict[vertex]
                lca = vertex
                break
            i += 1

        # Then construct the final path as (u to lca) + (lca to v)
        assert lca is not None
        path = longer[:i + 1] + shorter[j - 1::-1]

        # Depending on which part of the path is shorter, we may need to
        # reverse the order
        if path[0] == v:
            path.reverse()

        assert path[0] == u and path[-1] == v and w in path
        return path

    def query_for_path(self, u: Vertex, v: Vertex) -> List[Vertex]:
        """
        Query the ADO for two points and return the approximate shortest path
        between them as a list of Vertices.
        """
        _, w = self.query(u, v, return_w=True)
        return self.__get_path(u, v, w)

    def animate_query(self, u: GraphVertex,
                      v: GraphVertex, timestep: float = 2.0) -> None:
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

    def __plot_query_state(self,
                           fig: plt.Figure,
                           ax: plt.Axes,
                           u: GraphVertex, v: GraphVertex,
                           w: GraphVertex, i: int,
                           final: bool = False) -> None:
        """
        Plot a single frame/plot that depicts the current state of the ADO
        query. Clears and overwrites any previous contents of the plot.

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

        p_i_u = defaultdict(list)
        for i_val, (p, _) in self.p[u].items():
            p_i_u[p].append(i_val)

        # Color most nodes, black, but give special coloring to:
        # u, v, w, p_i(u)s and B(v)s.
        # Label u, v, w, p_i(u)s, and B(v)s
        node_colors = []
        node_labels = {}
        for n in self.vertices:
            if n == u:
                node_colors.append("red")
                node_labels[n.nx_node] = "u"
            elif n == v:
                node_colors.append("lime")
                node_labels[n.nx_node] = "v"
            elif n == w:
                node_colors.append("orange")
                node_labels[n.nx_node] = "w"
            elif n in p_i_u:
                node_colors.append("violet")
                i_values_str = ",".join(str(i) for i in p_i_u[n])
                node_labels[n.nx_node] = "p_{" + i_values_str + "}(u)"
            elif n in self.B[v]:
                node_colors.append("lime")
                node_labels[n.nx_node] = "B(v)"
            else:
                node_colors.append("black")

        # For the final plot, highlight the found path
        if final:
            path = self.__get_path(u, v, w)
            path_edges = set(
                (path[i], path[i + 1]) for i in range(len(path) - 1))
            edge_colors = ["lime"
                           if (u, v) in path_edges or (v, u) in path_edges
                           else "black"
                           for u, v in self.g.edges()]
        else:
            edge_colors = "black"

        pos = {(x, y): (y, -x) for x, y in self.g.nodes()}
        nx.draw(self.g,
                pos=pos,
                ax=ax,
                node_size=100,
                font_size=10,
                node_color=node_colors,
                edge_color=edge_colors,
                labels=node_labels)

        # Add a plot title
        title = "Iteration {} (final)".format(i) \
            if final else "Iteration {}".format(i)
        ax.set_title(title)
        fig.show()


def generate_random_weighted_2d_grid_graph(m: int = 5, n: int = 5,
                                           min_weight: int = 1,
                                           max_weight: int = 1,
                                           randomize_weights: bool = True,
                                           seed: int = 1) -> nx.Graph:
    """
    :param m: The number of rows in the grid graph.
    :param n: The number of columns in the grid graph.
    :param min_weight: The minimum weight of the uniform weight distribution
    to sample each edge's weight from.
    :param max_weight: The maximum weight of the uniform weight distribution
    to sample each edge's weight from.
    :param randomize_weights: If true, sample the weights from the uniform
    distribution over the range (min_weight, max_weight). If false,
    then min_weight and max_weight must be equal, and every edge's weight is
    set to that value.
    :param seed: The random seed used for the edge weight sampling.
    :return: The fully constructed NetworkX graph.
    """
    random.seed(seed)

    grid_graph = nx.grid_2d_graph(m, n)
    for (u, v, d) in grid_graph.edges(data=True):
        if randomize_weights:
            weight = random.randint(min_weight, max_weight)
        else:
            assert min_weight == max_weight, "If randomize_weights=False, " \
                                             "min_weight and max_weight must " \
                                             "be the same. "
            weight = min_weight
        d['weight'] = weight

    return grid_graph


if __name__ == "__main__":
    test_graph = generate_random_weighted_2d_grid_graph(seed=1)
    graph_ado = GraphADO(k=5, graph=test_graph)
    query_points = random.sample(graph_ado.vertices, 2)
    graph_ado.animate_query(query_points[0], query_points[1])
