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

dataset = "SMAP"

def get_data():
    dir_path = f"data/{dataset}/train"
    end = ".npy"
    base_path = f"output/{dataset}"
    f1s = []
    groups = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(end):
            group = file_name.split(end)[0]

            path = base_path + "/" + group
            file_name2 = path + "/baseline/summary.txt"
            try:
                f1 = get_recal(file_name2)
                groups.append(group)
                f1s.append(f1)
            except Exception as e:
                continue


    return groups,f1s


if __name__ == '__main__':
    groups,f1s = get_data()
    print(np.mean(f1s))
    plt.figure(dpi=300,figsize=(30,8))
    plt.plot(groups,f1s,c="r", label=f'f1', marker="^")
    # plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("group")
    plt.title(f"{dataset}_overall_f1")
    plt.savefig(f"analysis/{dataset}_overall_f1.pdf")
