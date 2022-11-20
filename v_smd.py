import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["f1"]
    return f1


def get_precision(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["precision"]
    return f1


def get_recal(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["recall"]
    return f1


def get_data():
    base_path = "output/SMD"
    upper = [8, 9, 11]
    f1s = []
    groups = []
    for i in range(3):
        for j in range(upper[i]):
            group = f"{i+1}-{j+1}"
            groups.append(group)
            path = base_path+"/"+group
            for k in os.listdir(path):
                if k != "logs" and not k.endswith("after"):
                    file_name = path+"/"+k+"/"+"summary.txt"
                    f1 = get_recal(file_name)
                    f1s.append(f1)
    return groups,f1s


if __name__ == '__main__':
    groups,f1s = get_data()
    print(np.mean(f1s))
    plt.figure()
    plt.plot(groups,f1s,c="r", label=f'f1', marker="^")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    plt.title(f"smd_overall_f1")
    plt.savefig("analysis/smd_overall_f1.pdf")
