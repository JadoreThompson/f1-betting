import os
import ydf


BPATH = os.path.dirname(__file__)
DPATH = os.path.join(BPATH, "datasets")
MPATH = os.path.join(BPATH, "models")
MDPATH = os.path.join(DPATH, "model_datasets")


for p in (
    BPATH,
    DPATH,
    MPATH,
    MDPATH,
):
    if not os.path.exists(p):
        os.mkdir(p)

LEARNER_TYPE = ydf.RandomForestLearner
MODEL_TYPE = ydf.RandomForestModel
TARGET_LABEL = "target"

TRAINED_MODEL: MODEL_TYPE = None
TRAINED_MODEL_FEATURES: tuple[str] | None = None
TRAINED_MODEL_CLASSES: tuple[str] | None = None

mpath = os.path.join(MPATH, "model_x")

if os.path.exists(mpath):
    TRAINED_MODEL = ydf.load_model(mpath)
    TRAINED_MODEL_CLASSES = tuple(TRAINED_MODEL.label_classes())
    TRAINED_MODEL_FEATURES = tuple(TRAINED_MODEL.input_feature_names())
    # print("Trained model features:", TRAINED_MODEL_FEATURES)
