# Dataset

This app is built on the **"The Movies Dataset"** from Kaggle:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

The raw `movies_metadata.csv` is not committed to this repo (it's ~33MB and
redistributing third-party datasets via git isn't great practice). The
pre-built model artifacts in `app/model/` are committed instead, so the app
runs out of the box without needing this file.

If you want to retrain the model yourself:

1. Download `movies_metadata.csv` from the Kaggle link above.
2. Place it in this `data/` folder.
3. Run `python build_model.py` from the project root — this regenerates
   everything in `app/model/`.
