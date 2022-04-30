# Graph VIT based SSL Documentation

1. Training Patch Feature Extractor

**How to run:**

- cd feature_extractor/
- python feature_extract.py

**Notes:**

- check config.yaml for configuration
- change either want to use SimCLR or MocoV3 in feature_extract.py (simply comment/uncomment it)
- if want to use Pytorch Lightning, run feature_extract_lightning.py instead (only MoCoV3 is ready)
- model path: runs/(simclr or mocov3)/runs/(name of model)

2. Build Graph

**How to run:**

- python build_graphs.py

**Notes:**

- check arg parser for configuration
- choose between loading SimCLR and MoCoV3 model in build_graphs.py (simply comment/uncomment it)
- graph path: ../build_graphs/(simclr or mocov3)/(name of model)/simclr_files/

3. Training Graph VIT

**How to run:**

- cd ..
- python main.py (check arg parser at option.py)

**Notes:**

- when testing: train=False, test=True, val_set=(test data), resume=(graph VIT model path)
- result path: results/(simclr or mocov3)/(name of result folder)/

4. Visualization (In Progress)
