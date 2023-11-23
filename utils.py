import requests
import os
from pathlib import Path


def download_file(url: str, name: str):
    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    path = Path("dataset") / name
    with open(path, "wb") as f:
        response = requests.get(url)
        f.write(response.content)
