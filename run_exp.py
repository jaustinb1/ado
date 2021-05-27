import os
import subprocess

os.chdir("cpp")

subprocess.run("make")

base = "../graphs"
all_graphs = [os.path.join(base, f) for f in os.listdir(base)]


command_tz = "./bin/main tz {} res.csv {} {} {} 1"
command_bdj = "./bin/main bdj {} res.csv {} {} {} 1"


def run_command(cmd, g, k, u, v):
    return subprocess.run(
        cmd.format(g,k,u,v).split(" "), capture_output=True, encoding='utf-8'
    ).stdout

def parse_output(out):
    split = output.split("\n")

    preproc_time = float(split[0].split(" ")[-1])
    dist_est = float(split[3].split(" ")[-1])
    true_dist = float(split[7].split(" ")[2].split("=")[1])

    stretch = dist_est / true_dist

    return preproc_time, dist_est, true_dist, stretch

for g_path in all_graphs:
    print(g_path)
    output = run_command(command_tz, g_path, 4, 1, 2)

    print(parse_output(output))
