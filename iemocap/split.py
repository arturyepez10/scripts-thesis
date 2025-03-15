from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def load_dataframe(path: str) -> pd.DataFrame:
  return pd.read_csv(path, sep='\t')

def save_dataframe(df: pd.DataFrame, filename: str):
  df.to_csv("out/" + filename, sep='\t', index=False)

def main(filename: str):
  df = load_dataframe("./out/" + filename)

  # We use a generator to create a random number between 0 and 1
  # and select 42 as the seed for reproducibility
  gen = np.random.default_rng(42)
  
  # We add a column to the dataframe with a random number for each row
  df['split'] = [gen.random() for _ in range(0, len(df))]

  # Use the random number to split the dataset based on the 70/30 rule
  msk = df["split"] <= 0.7

  train = df[msk]
  test = df[~msk]

  # Remove the split column
  train = train.drop(columns=["split"])
  test = test.drop(columns=["split"])

  # Save the resulting dataframes
  save_dataframe(train, filename.replace(".csv", "-train.csv"))
  save_dataframe(test, filename.replace(".csv", "-test.csv"))

if __name__ == "__main__":
  main("iemocap.csv")