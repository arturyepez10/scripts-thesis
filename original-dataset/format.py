import pandas as pd

CLASSIFIER_LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "love", "optimism", "pessimism", "sadness", "surprise", "trust"]
NEW_CLASSIFIER_LABELS = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

"""
----------------------------
RULES OF EMOTION ASSIGNMENT
----------------------------
Due to differences in the emotions portrayed in the different datasets used for the thesis project, we have come up with a new minimized
set of emotions (based on the principal dyad of the Plutchik's wheel of emotions) that we will use for the final classification of the text. 

The end result of available emotions will follow the ones depicted in the NEW_CLASSIFIER_LABELS list.

Emotions will be assigned as follows:

- Anger: anger
- Anticipation: anticipation
- Disgust: disgust
- Fear: fear
- Joy: joy
- Love: joy, trust
- Optimism: joy, anticipation
- Pessimism: sadness, anticipation
- Sadness: sadness
- Surprise: surprise
- Trust: trust
"""

def load_dataframe(path: str) -> pd.DataFrame:
  return pd.read_csv(path, sep='\t')

def save_dataframe(df: pd.DataFrame, filename: str):
  df.to_csv("out/" + filename, sep='\t', index=False)

def pre_process(df: pd.DataFrame) -> pd.DataFrame:
  # Obtain the text and the labels separately
  tweet_id = df.iloc[:, 0:1]
  text = df.iloc[:, 1:2]
  labels = df.iloc[:, 2:].to_numpy()

  # Join the text and the labels in a new dataframe
  text_and_labels = [
    [
      tweet_id.iloc[index, 0],
      text.iloc[index, 0],
      labels[index].tolist()
    ] for index in range(0,int(labels.shape[0]))
  ]

  return pd.DataFrame(text_and_labels, columns=["id", "text", "target"])

def convert_row(row: list[1 | 0]) -> list[1 | 0]:
  new_row = [0 for _ in range(0, len(NEW_CLASSIFIER_LABELS))]
  for index, element in enumerate(row):
    # If there is an emotion detected in the index of the row
    if element == 1:

      # We assign it according to the end result assignments we set up as end result
      assignment_index = []

      # Manually go through the original labels and parse them to the new emotions/labels
      if CLASSIFIER_LABELS[index] == "anger":
        assignment_index.append(5)
      elif CLASSIFIER_LABELS[index] == "anticipation":
        assignment_index.append(7)
      elif CLASSIFIER_LABELS[index] == "disgust":
        assignment_index.append(3)
      elif CLASSIFIER_LABELS[index] == "fear":
        assignment_index.append(4)
      elif CLASSIFIER_LABELS[index] == "joy":
        assignment_index.append(0)
      elif CLASSIFIER_LABELS[index] == "love":
        assignment_index.append(0)
        assignment_index.append(2)
      elif CLASSIFIER_LABELS[index] == "optimism":
        assignment_index.append(0)
        assignment_index.append(7)
      elif CLASSIFIER_LABELS[index] == "pessimism":
        assignment_index.append(1)
        assignment_index.append(7)
      elif CLASSIFIER_LABELS[index] == "sadness":
        assignment_index.append(1)
      elif CLASSIFIER_LABELS[index] == "surprise":
        assignment_index.append(6)
      elif CLASSIFIER_LABELS[index] == "trust":
        assignment_index.append(2)

      # We assign the new indexes to the new resulting row
      for new_index in assignment_index:
        new_row[new_index] = 1

  return new_row

def main(filename: str):
  # We process the dataset from the original format to the 2 columns (text, target)
  df = load_dataframe("./original-dataset/dataset/" + filename)
  df = pre_process(df)

  # Obtain individual columns
  tweet_id_column = df["id"].tolist()
  text_column = df["text"].tolist()
  target_column = df["target"].tolist()

  # Convert the target column to the new format
  new_target_column = [convert_row(row) for row in target_column]

  # Join the text and the labels in a new dataframe
  text_and_labels = [
    [
      tweet_id_column[index],
      text_column[index], 
      *new_target_column[index]
    ] for index in range(0,int(len(new_target_column)))
  ]
  new_df = pd.DataFrame(text_and_labels, columns=["ID", "Tweet", *NEW_CLASSIFIER_LABELS])

  # Save the new dataframe
  save_dataframe(new_df, filename.replace(".txt", ".v2.txt"))


if __name__ == "__main__":
  # We tranform both training and test datasets
  main("2018-E-c-En-train.txt")

  main("2018-E-c-En-test-gold.txt")