"""Download MovieLens-1M into data/raw/ml-1m/ if not present."""
import os, urllib.request, zipfile, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "ml-1m")
os.makedirs(DATA, exist_ok=True)

# (this script lives in src/ - keep imports isolated; here is just file I/O)
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ZIP = os.path.join(DATA, "ml-1m.zip")
rat_path = os.path.join(DATA, "ratings.dat")
if os.path.exists(rat_path) and os.path.getsize(rat_path) > 0:
    print("Already downloaded.")
else:
    print("Downloading MovieLens-1M...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        with open(ZIP, "wb") as f:
            shutil.copyfileobj(r, f)
    print("Extracting...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(DATA)  # creates ml-1m/ subdir inside our data/ml-1m
    # Move nested ml-1m/* up
    nested = os.path.join(DATA, "ml-1m")
    if os.path.isdir(nested):
        for n in os.listdir(nested):
            shutil.move(os.path.join(nested, n), os.path.join(DATA, n))
        os.rmdir(nested)
    os.remove(ZIP)

for n in os.listdir(DATA):
    p = os.path.join(DATA, n)
    print(n, os.path.getsize(p))
