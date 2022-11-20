import os

dir_path = "datasets/ServerMachineDataset/train"

for file_name in os.listdir(dir_path):
    if file_name.endswith(".txt"):
        dataset = file_name.split(".txt")[0]
        dataset = dataset.replace("machine-", "")
        # print(dataset)
        if dataset == "1-1" or dataset == "1-2":
            continue
        os.system(f"python train.py --dataset smd --group {dataset}")
        #python train.py --dataset smd --group 1-4 --epochs 1
