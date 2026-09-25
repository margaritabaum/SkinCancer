"""Re-decode HAM10000 at 256x256 (the notebook used 71x71) and keep lesion ids
so predictions can be pooled per lesion later. Split logic is unchanged.
"""
import os, glob, numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

SP = os.path.dirname(os.path.abspath(__file__))
SIZE = 256
np.random.seed(1)

# Override with HAM10000_METADATA=/path/to/HAM10000_metadata.csv
METADATA = os.environ.get("HAM10000_METADATA",
                          os.path.join(SP, "HAM10000_metadata.csv"))
df = pd.read_csv(METADATA)
df = df.dropna(subset=["age"]).reset_index(drop=True)
lesion_ID = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
df['lesion_ID'] = df['dx'].map(lesion_ID.get)

images_path = {os.path.splitext(os.path.basename(p))[0]: p
               for p in glob.glob(os.path.join(SP, "images", "*.jpg"))}
df['path'] = df['image_id'].map(images_path.get)
assert df['path'].notna().all()

cache = os.path.join(SP, f"x{SIZE}.npy")
if not os.path.exists(cache):
    x = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8,
                                  shape=(len(df), SIZE, SIZE, 3))
    for i, p in enumerate(df['path'].values):
        x[i] = np.asarray(Image.open(p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        if i % 2000 == 0:
            print(f"  decoded {i}/{len(df)}", flush=True)
    x.flush()
    print("cached", cache)

lesions = df.drop_duplicates("lesion_id")[["lesion_id", "lesion_ID"]]
train_les, rest_les = train_test_split(lesions, test_size=0.20, random_state=28,
                                       stratify=lesions["lesion_ID"])
val_les, test_les = train_test_split(rest_les, test_size=0.50, random_state=28,
                                     stratify=rest_les["lesion_ID"])
is_train = df["lesion_id"].isin(train_les["lesion_id"]).values
is_val = df["lesion_id"].isin(val_les["lesion_id"]).values
is_test = df["lesion_id"].isin(test_les["lesion_id"]).values

# integer lesion index per image, for pooling several photos of one lesion
lesion_codes = pd.factorize(df["lesion_id"])[0]

np.savez_compressed(os.path.join(SP, "split256.npz"),
                    y=df["lesion_ID"].values, is_train=is_train, is_val=is_val,
                    is_test=is_test, lesion_codes=lesion_codes)
print(f"images  train {is_train.sum()} | val {is_val.sum()} | test {is_test.sum()}")
