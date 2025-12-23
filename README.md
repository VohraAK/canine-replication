# Group 4: CANINE QA

### GitHub Project Repo: https://github.com/VohraAK/canine-replication/

---

### Main Notebooks:
1. `Reproducing_CANINE_results_tydiqa.ipynb` (TyDiQA with CANINE-C model training + eval)
2. `Train_CANINE_S_LoRA_UQA_unfiltered.ipynb` (Unfiltered UQA with CANINE-S model training + eval)
3. `Train_CANINE_S_LoRA_UQA_filtered.ipynb` (Filtered UQA with CANINE-S model training + eval)

<br>

### Running the notebooks:
Training was done on Kaggle, with additional `pip` dependencies:
- `peft`
- `evaluate`
- `transformers`
- `Levenshtein`
- `ipywidgets`

<br>

There might be some cells commented out which load the dataset, split and preprocess them, and cache them in local / Kaggle storage, make sure to tweak them based on the environment and requirements.

A personal HuggingFace token was used to push model checkpoints to HF repositories under `VohraAK`. However, I have included downloaded model checkpoints for the three main training sessions under the `checkpoints` directory.

Evaluation assets are available under `assets` directory.

---
