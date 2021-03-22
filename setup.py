import os
import zipfile

from urllib.request import urlopen, Request

FILE_URL = 'https://storage.googleapis.com/trending-sentiments-data/bert_model.zip'
MODEL_PATH = './bert_model'
DOWNLOAD_PATH = './bert_model.zip'

is_model_downloaded = os.path.exists(MODEL_PATH)

if not is_model_downloaded:
    print('Model Downloading...')
    req = Request(FILE_URL)
    with open(DOWNLOAD_PATH, 'wb') as target:
        target.write(urlopen(req).read())
    print('Download complete')
    with zipfile.ZipFile(DOWNLOAD_PATH, "r") as zf:
        zf.extractall()
    print('Cleaning Up...')
    os.remove(DOWNLOAD_PATH)
    print('Model downloaded to: {}'.format(MODEL_PATH))
else:
    print("Model exists at: {}".format(MODEL_PATH))
