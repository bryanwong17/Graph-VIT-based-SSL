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
    SOURCE_ROOT = "D:/MedicalAI/dataset/CAMELYON16/patch"
    SUBSET = sorted(["train", "val", "test"])

    for subset in SUBSET:
        with open(f"new_{subset}_temp.csv", 'w', newline="") as f:
            writer = csv.writer(f)

            # write the name of column
            writer.writerow(columns)
            source_path = f"{SOURCE_ROOT}/{subset}/group_patches"
            wsi_files = os.listdir(source_path)
            temp_patches = []
            for f in range(len(wsi_files)):
                temp_patches.append([])
                each_wsi = os.listdir(os.path.join(source_path, wsi_files[f]))
                for patch in each_wsi:
                    temp_patches[f].append(os.path.join(source_path, wsi_files[f], patch))
                   
            # make sure that each batch has patch from different slides
            max_len = max([len(i) for i in temp_patches])
            outputs = []
            for i in range(max_len):
                for x in temp_patches:
                    if i < len(x):
                        outputs.append(x[i])
                        writer.writerow([x[i]])


if __name__ == "__main__":
    main()



