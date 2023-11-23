from dotenv import load_dotenv
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

from utils import download_file

load_dotenv()

SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "JPG", "PNG", "JPEG"]


df = pd.read_csv("ocr_requests-牛気-將軍澳店.csv")

df.drop_duplicates(subset=["file_hash"], inplace=True)

df["extension"] = df["url"].apply(lambda row: row.split(".")[-1])

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    if row["extension"] in SUPPORTED_EXTENSIONS:
        url = os.environ["BASE_URL"] + f"/{row['url']}"
        path: Path = download_file(url, name=f"{row['id']}.{row['extension']}")
