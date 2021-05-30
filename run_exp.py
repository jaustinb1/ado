import os
import subprocess

K = 6

os.chdir("cpp")

subprocess.run("make")

base = "../graphs"

for f in list(os.listdir(base)):
    if f.endswith(".out"):
        os.remove(os.path.join(base, f))

all_graphs = [os.path.join(base, f) for f in os.listdir(base)]


command_tz = "./bin/main tz {} res.csv {} {} {} 10"
command_bdj = "./bin/main bdj {} res.csv {} {} {} 1"


def run_command(cmd, g, k, u, v):
    c = cmd.format(g,k,u,v).split(" ")
    print(" ".join(c))
    return subprocess.run(
        c, capture_output=True, encoding='utf-8'
    )

def parse_output(out):
    split = output.split("\n")

    preproc_time = float(split[0].split(" ")[-1])
    dist_est = float(split[3].split(" ")[-1])
    true_dist = float(split[7].split(" ")[2].split("=")[1])

    stretch = dist_est / true_dist

    return preproc_time, dist_est, true_dist, stretch

for g_path in all_graphs:
    print(g_path)
    output = run_command(command_tz, g_path, K, 1, 2)
    #print(output.stdout)
    print(output.stdout)
    #print(parse_output(output))
