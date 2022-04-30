import os
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd

# D:0
# M:1
# N:2

def main():

    columns = ["path to patches"]
    SOURCE_ROOT = "D:/MedicalAI/dataset/TCGA/patch"
    SUBSET = sorted(["train", "val", "test"])
    CLASSES = sorted(["LUDA", "LUSC"])

    for subset in SUBSET:
        with open(f"TCGA_{subset}.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            # write the name of column
            writer.writerow(columns)
            for _class in CLASSES:
                source_path = f"{SOURCE_ROOT}/{subset}/{_class}"
                patch_files = os.listdir(source_path)
                for p in patch_files:
                    each_patch = os.path.join(source_path, p)
                    writer.writerow([each_patch])

if __name__ == "__main__":
    main()

