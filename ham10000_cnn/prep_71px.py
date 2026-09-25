"""Replicates cells 0-28 of 'Skin_Cancer (1).ipynb': load metadata, decode and
resize every image to 71x71, build the lesion-grouped 80/10/10 split.

Only deviation: cv2 is not installed, so PIL does the decode/resize.
"""
import os, glob, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SP = os.path.dirname(os.path.abspath(__file__))
np.random.seed(1)

# Override with HAM10000_METADATA=/path/to/HAM10000_metadata.csv
METADATA = os.environ.get("HAM10000_METADATA",
                          os.path.join(SP, "HAM10000_metadata.csv"))
df = pd.read_csv(METADATA)
print("metadata:", df.shape)

df = df.dropna(subset=["age"]).reset_index(drop=True)
print("after dropping missing age:", df.shape)

lesion_ID = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
df['lesion_ID'] = df['dx'].map(lesion_ID.get)

images_path = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(SP, "images", "*.jpg"))}
df['path'] = df['image_id'].map(images_path.get)
assert df['path'].notna().all(), f"{df['path'].isna().sum()} images not found on disk"

cache = os.path.join(SP, "x71.npy")
if os.path.exists(cache):
    x = np.load(cache)
else:
    x = np.empty((len(df), 71, 71, 3), dtype=np.uint8)
    for i, p in enumerate(df['path'].values):
        x[i] = np.asarray(Image.open(p).convert("RGB").resize((71, 71), Image.BILINEAR))
        if i % 2000 == 0:
            print(f"  decoded {i}/{len(df)}", flush=True)
    np.save(cache, x)

y = df["lesion_ID"].values
print("x:", x.shape, x.dtype, "| y:", y.shape)

# ---- lesion-grouped split (notebook cell 27) ----
lesions = df.drop_duplicates("lesion_id")[["lesion_id", "lesion_ID"]]
train_les, rest_les = train_test_split(lesions, test_size=0.20, random_state=28,
                                       stratify=lesions["lesion_ID"])
val_les, test_les = train_test_split(rest_les, test_size=0.50, random_state=28,
                                     stratify=rest_les["lesion_ID"])

is_train = df["lesion_id"].isin(train_les["lesion_id"]).values
is_val = df["lesion_id"].isin(val_les["lesion_id"]).values
is_test = df["lesion_id"].isin(test_les["lesion_id"]).values

# no lesion may appear in two splits
assert not (set(train_les.lesion_id) & set(val_les.lesion_id) & set(test_les.lesion_id))
assert (is_train.sum() + is_val.sum() + is_test.sum()) == len(df)

np.savez_compressed(os.path.join(SP, "split.npz"),
                    y=y, is_train=is_train, is_val=is_val, is_test=is_test)
print(f"images  train {is_train.sum()} | val {is_val.sum()} | test {is_test.sum()}")
print(f"lesions train {len(train_les)} | val {len(val_les)} | test {len(test_les)}")
