import joblib
import requests


def get_model(model_path):
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")

        # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download"
        url = "https://drive.google.com/u/0/uc?id=1eZQFdqwAMHLsjwD2C-PyvcZlVAwmAudD&export=download"
        r = requests.get(url, allow_redirects=True)
        open(r"Model/model.pkl", 'wb').write(r.content)
        del r
        with open(r"Model/model.pkl", "rb") as m:
            rf = joblib.load(m)
    return rf
