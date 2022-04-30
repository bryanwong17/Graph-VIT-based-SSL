import os
import pandas as pd
import shutil

# D:0
# M:1
# N:2

def main():

    CSV_ROOT = "../../dataset/graph_transformer_3_class/colon/03. same_config/"
    FOLDER_ROOT = "../../dataset/graph_transformer_3_class/tiles"
    TARGET_ROOT = "../../dataset/graph_transformer_3_class/new_tiles_2"
    SUBSET = sorted(["train", "val", "test"])
    CLASSES = sorted(["D", "M", "N"])

    for _class in CLASSES:
        for subset in SUBSET:
            print(subset, _class)
            df = pd.read_csv(CSV_ROOT + f"{_class}_{subset}.csv")
            temp_csv = list(df["file"])
            temp_path = os.path.join(FOLDER_ROOT, subset, "group_patches")
            for folder in os.listdir(temp_path):
                for item in temp_csv:
                    if folder == item:
                        for file in os.listdir(os.path.join(temp_path, folder)):
                            temp_target = os.path.join(TARGET_ROOT, subset, folder)
                            if not os.path.exists(temp_target):
                                os.makedirs(temp_target)
                            shutil.copy(os.path.join(temp_path, folder, file), os.path.join(temp_target, file))
        





            
            

if __name__ == "__main__":
    main()