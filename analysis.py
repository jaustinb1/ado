import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class DataPoint:

    def __init__(self, f_name, u, v, ado_est, true_dist):
        self.density = float(
            ".".join(f_name.split("=")[-1].split(".")[:2])
        )
        self.u = u
        self.v = v
        self.ado_est = ado_est
        self.true_dist = true_dist
        self.stretch = float(self.ado_est) / max(float(self.true_dist), 1.0)
        if self.stretch == 0:
            self.stretch = 1.0

        self.good = True

        if self.stretch > (2 * K - 1):
            self.good = False
            #print(self.u, self.v, self.stretch, self.ado_est, self.true_dist)
        if self.stretch < 1.0:
            #print(self.u, self.v, self.stretch, self.ado_est, self.true_dist)

            self.good = False
        #print(self.stretch)

    def __repr__(self):
        return "Density: {}\t ADO Estimate: {}\t True Distance: {}\t Stretch: {}".format(
            self.density, self.ado_est, self.true_dist, self.stretch
        )


def parse_file(f_name):
    d = []
    with open(f_name, "r") as infile:
        for l in infile:
            line = l[:-1].strip()
            line = line.split(" ")
            if len(line) < 4:
                continue

            line = [int(x) for x in line]
            d.append(DataPoint(
                f_name, line[0], line[1], line[2], line[3]
            ))

    return d

def parse_all_files(lst):

    all_data = []
    for f in lst:
        all_data.extend(
            parse_file(f)
        )
    return all_data

def reduce_by_density(dd):
    density_results_dict = defaultdict(lambda: [])

    for query in dd:
        if not query.good:
            continue
        density_results_dict[query.density].append(query.stretch)

    return density_results_dict

if __name__ == "__main__":
    from tqdm import tqdm
    base = ["graph_out_2", "graph_out_3", "graph_out_4", "graph_out_5", "graph_out_6"]

    every_k = {}
    for b in tqdm(base):
        K = int(b[-1:])

        files = [os.path.join(b, f) for f in os.listdir(b) if ".out" in f]
        data = {}


        d = parse_all_files(files)
        ddd = reduce_by_density(d)


        densities = [dddd for dddd in sorted(ddd.keys())]

        means = [np.mean(ddd[dddd]) for dddd in sorted(ddd.keys())]
        std = [np.std(ddd[dddd]) for dddd in sorted(ddd.keys())]
        maxes = [np.max(ddd[dddd]) for dddd in sorted(ddd.keys())]

        every_k[K] = {
            'densities': densities,
            'means':  means,
            'stds': std,
            'maxes': maxes
        }

    plt.figure()
    for k in every_k:
        densities = every_k[k]['densities']
        means = every_k[k]['means']
        #plt.plot(densities, means)
        plt.plot(densities[20:-20], smooth(means, 20)[20:-20], label=str(k))
    plt.title("mean stretch")
    plt.legend()

    plt.figure()
    for k in every_k:
        densities = every_k[k]['densities']
        std = every_k[k]['stds']
        plt.plot(densities[20:-20], smooth(std, 20)[20:-20])
        plt.title("std stretch")

    plt.figure()
    for k in every_k:
        densities = every_k[k]['densities']
        maxes = every_k[k]['maxes']
        plt.plot(densities[20:-20], smooth(maxes, 20)[20:-20])
        plt.title("max stretch")

    plt.show()
