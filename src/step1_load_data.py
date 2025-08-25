import pandas as pd

# Load with no header
df = pd.read_csv("/Users/ganavimc/Downloads/viral-nlp-ml-starter/data/fullset_train (1).csv", header=None)

# Renaming columns properly
df.columns = ["id", "sequence", "label"]

print("Shape of dataset:", df.shape)
print(df.head())

# Check class balance
print("\nLabel counts:\n", df["label"].value_counts())
