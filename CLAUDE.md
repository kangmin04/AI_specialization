# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Personal study repository for Andrew Ng's Coursera "Machine Learning Specialization" (README: "Finish ML - DeepLearning course within a week"). It is a set of standalone Python scripts transcribed from the course's Jupyter notebooks — there are no `.ipynb` files, no test suite, no linter/formatter config, and no dependency manifest (`requirements.txt`/`pyproject.toml`/etc). Treat every script as a study lab, not production code.

## Running scripts

There is no build or test command. Run any lab directly with Python 3 from the repo root:

```bash
python3 ML/<chapter>/<week>/<lab-file>.py
```

Dependencies (installed ad hoc in the environment, no manifest tracks them): `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`/`keras`, `xgboost`, `pandas`, `scipy`, `networkx`, `Pillow`. Install whichever a given script imports if missing.

Some scripts import Coursera-provided helper modules (e.g. `lab_utils_common`, `lab_coffee_utils`) that are **not present** in the folder they're used from — those scripts will raise `ImportError` if run as-is (see Architecture below). This is a known, pre-existing state of the repo, not something to silently "fix" by deleting the import unless asked.

## Architecture

The repo mirrors the structure of the Coursera specialization: `ML/<chapterN>/weekN/`, where each `chapterN` corresponds to one course in the specialization and `weekN` to that course's week.

- `ML/chapter1/` — Course 1, Supervised Machine Learning (linear regression → logistic regression, regularization). Has a shared `utils/` folder (`lab_utils_common.py`, `lab_utils_multi.py`, `lab_utils_uni.py`, `deeplearning.mplstyle`) plus per-week copies of the same helper files pasted directly into `week2/` and `week3/`. `data/houses.txt` is used by week1/week2 labs.
- `ML/chapter2-advanced/` — Course 2, Advanced Learning Algorithms (neural networks, multiclass/softmax, decision trees & tree ensembles). `week1/` = neurons/TensorFlow basics, `week2/` = activation functions (ReLU vs. sigmoid, why non-linearity is needed) & multiclass classification/softmax, `week4/` = decision trees / random forest / XGBoost using `week4/heart.csv` and a local `week4/utils.py`. Note: `week3/` does not exist (skipped in this repo). `tensorflow-dimension-definition.py` at the chapter root is a standalone concept note on numpy vs. TensorFlow array shapes, not a lab.
- `ML/chapter3_unsupervised/` — Course 3, Unsupervised Learning (clustering, and further course topics not yet started). `week1/k-means/` implements K-means from scratch (`find_closest_centroids`, `compute_centroids` in `lab01-kmeans.py`) and applies it to image color-quantization in `lab02-imagecompression.py`, backed by a local `utils.py`/`public_tests.py`.

Key conventions to follow when adding new labs:
- File naming: `lab0N-topic-slug.py`, numbered sequentially within its `weekN/` folder.
- Each `weekN/` folder is self-contained — Coursera helper modules (`lab_utils_*.py`, `utils.py`, `public_tests.py`, `.mplstyle`) are copied per-folder rather than imported from a shared location; don't refactor these into a shared module unless asked.
- A single lab file commonly walks through several sequential sections (e.g. a from-scratch numpy implementation followed by the scikit-learn/TensorFlow equivalent), separated by comment banners, rather than being split into multiple files.
- Comments are informal and mostly in Korean, written as the author's own study notes (explaining *why*, reasoning through results, conclusions). Match that tone/language in new labs rather than switching to formal English docstrings throughout. Typos in filenames/identifiers from the original course material are left as-is.
- plt.show() 작성 
