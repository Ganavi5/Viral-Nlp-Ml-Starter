import pandas as pd

# path to dataset
DATA_PATH = "data/fullset_train (1).csv"

# load dataset
df = pd.read_csv(DATA_PATH)

# show basic info
print("Shape of dataset:", df.shape)
print("First 5 rows:\n", df.head())

# check missing values
print("\nMissing values:\n", df.isnull().sum())

# check class balance
print("\nLabel counts:\n", df['label'].value_counts())
