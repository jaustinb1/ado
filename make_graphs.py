import numpy as np
import os
import subprocess
import shutil


if os.path.exists("graphs"):
    shutil.rmtree("graphs")

os.chdir("graph_gen")

if os.path.exists("graphs"):
    shutil.rmtree("graphs")

generator = "./dimacs.sh"

N = 100
n_graphs = 1000

space = np.linspace(0., 1., n_graphs)


for i in range(n_graphs):
    subprocess.run([generator, "-n", str(N), "-d", str(space[i]), "-w", "1", "10"])

shutil.move("graphs", "..")
