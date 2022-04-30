import os
import shutil

def main():

    SOURCE_ROOT = 'D:/MedicalAI/dataset/TCGA/patch'

    CLASSES = sorted(["LUDA", "LUSC"])
    SUBSET = sorted(["train", "val", "test"])

    for subset in SUBSET:
        index = []
        for _class in CLASSES:
            source_path = f"{SOURCE_ROOT}/{subset}/{_class}"
            print(source_path)
            source_files = os.listdir(source_path)
            for f in source_files:
                f = f.strip()
                f = f.split('-')[0]
                index.append(f)
            
            index = list(set(index))
            for f in source_files:
                f = f.strip()
                for i in index:
                    if f.split("-")[0] == i:
                        target_path = f"{SOURCE_ROOT}/{subset}/group_patches"
                        print(target_path)
                        if not os.path.exists(os.path.join(target_path, i)):
                            os.makedirs(os.path.join(target_path, i))
                        shutil.copy(os.path.join(source_path, f), os.path.join(target_path, i, f))
                        break
            


if __name__ == "__main__":
    main()