import os

# N:0
# M:l
# D:2

def main():

    SOURCE_ROOT = "../dataset/graph_transformer_3_class"
    SUBSET = sorted(["train", "val", "test"])
    # CLASSES = sorted(["D","M","N"])

    for subset in SUBSET:
        with open(f"{subset}_set_3.txt","w+") as txt:
            source_path = f"{SOURCE_ROOT}/{subset}/D"
            wsi_files = os.listdir(source_path)
            for wsi in wsi_files:
                if wsi.endswith(".mrxs"):
                    wsi = wsi.split(".mrxs")[0]
                    txt.write(wsi + "\\t2")
                    txt.write("\n")


if __name__ == "__main__":
    main()



