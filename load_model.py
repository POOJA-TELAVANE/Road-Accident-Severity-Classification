import joblib
import requests


def get_model(model_path):
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")

        # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download"
        url = "Paste your shareable url here after uploading it to google drive"
        r = requests.get(url, allow_redirects=True)
        open(r"models/model.pkl", 'wb').write(r.content)
        del r
        with open(r"models/model.pkl", "rb") as m:
            rf = joblib.load(m)
    return rf