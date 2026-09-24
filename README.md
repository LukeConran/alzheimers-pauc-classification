# Alzheimer's Classification via Partial AUC

Thesis project on Alzheimer's classification optimized for partial AUC.

Builds on prior work in [alzheimers-contrastive-learning](https://github.com/LukeConran/alzheimers-contrastive-learning).

## Structure

- `experiments/losses/` — loss function implementations
- `experiments/training/` — training scripts
- `experiments/models/` — model definitions
- `experiments/evaluation/` — evaluation / pAUC metrics
- `experiments/configs/` — run configs
- `experiments/jobs/` — job scripts (e.g. cluster/slurm)
- `experiments/data/` — dataset loading code (raw/processed data itself is gitignored)
- `figures/` — key result figures for the writeup

Thesis drafts, reading notes, meeting notes, and admin are kept separately (not in this repo).
