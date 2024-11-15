import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = os.path.join(ROOT_DIR, 'datasets')
PREDICTIONS_FOLDER = os.path.join(ROOT_DIR, 'predictions')
TMP_FOLDER = os.path.join(ROOT_DIR, 'tmp')
ALLOWED_EXTENSIONS = {'xarff'}