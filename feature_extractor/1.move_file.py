import os
import shutil


def main():
    TILE_ROOT = "D:/MedicalAI/dataset/graph_transformer_3_class/tiling_2"
    SUBSET = sorted(["train", "val", "test"])

    for subset in SUBSET:
        tile_path = f"{TILE_ROOT}/{subset}/group_patches"
        for folder in os.listdir(tile_path):
            for file in os.listdir(os.path.join(tile_path, folder)):
                if not os.path.exists(os.path.join(TILE_ROOT, "total_group_patches", folder)):
                    os.makedirs(os.path.join(TILE_ROOT, "total_group_patches", folder))
                shutil.copy(os.path.join(tile_path, folder, file), os.path.join(TILE_ROOT, "total_group_patches", folder, file))


if __name__ == "__main__":
    main()