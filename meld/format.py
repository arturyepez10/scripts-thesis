import pandas as pd

MELD_LABELS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
CLASSIFIER_LABELS = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

"""
----------------------------
RULES OF EMOTION ASSIGNMENT
----------------------------
Due to differences in the emotions portrayed in the different datasets used for the thesis project, we have come up with a new minimized
set of emotions (based on the principal dyad of the Plutchik's wheel of emotions) that we will use for the final classification of the text. 

The end result of available emotions will follow the ones depicted in the CLASSIFIER_LABELS list.

Emotions will be assigned as follows:

- Neutral:
- Joy: joy
- Surprise: surprise
- Anger: anger
- Sadness: sadness
- Disgust: disgust
- Fear: fear
"""

def load_dataframe(path: str) -> pd.DataFrame:
  return pd.read_csv(path)

def save_dataframe(df: pd.DataFrame, filename: str):
  df.to_csv("out/" + filename, sep='\t', index=False)

def pre_process(df: pd.DataFrame) -> pd.DataFrame:
  # Obtain necessary columns separately
  scene_id = df["Sr No."].to_list()
  dialog = df["Utterance"].to_list()
  emotion = df["Emotion"].to_list() 

  # Join the text and the labels in a new dataframe
  text_and_labels = [
    [
      scene_id[index],
      dialog[index],
      emotion[index]
    ] for index in range(0,len(dialog))
  ]

  return pd.DataFrame(text_and_labels, columns=["id", "text", "target"])

def convert_row(element: str) -> list[1 | 0]:
  new_row = [0 for _ in range(0, len(CLASSIFIER_LABELS))]

  # We assign it according to the end result assignments we set up as end result
  assignment_index = []

  # Manually go through the labels and parse them to the new emotions/labels
  # while ignoring the 'neutral' emotions
  if element == "joy":
    assignment_index.append(0)
  elif element == "surprise":
    assignment_index.append(6)
  elif element == "anger":
    assignment_index.append(5)
  elif element == "sadness":
    assignment_index.append(1)
  elif element == "disgust":
    assignment_index.append(3)
  elif element == "fear":
    assignment_index.append(4)
    

  # We assign the new indexes to the new resulting row
  for new_index in assignment_index:
    new_row[new_index] = 1

  return new_row

def main(filename: str):
  # We process the dataset from the original format to the 3 columns (id, text, target)
  df = load_dataframe("./meld/dataset/" + filename)

  df = pre_process(df)

  # Obtain individual columns
  id_column = df["id"].tolist()
  text_column = df["text"].tolist()
  target_column = df["target"].tolist()

  # Convert the target column to the new format
  new_target_column = [convert_row(row) for row in target_column]

  # Join the text and the labels in a new dataframe
  text_and_labels = [
    [
      id_column[index],
      text_column[index], 
      *new_target_column[index]
    ] for index in range(0,int(len(new_target_column)))
  ]
  new_df = pd.DataFrame(text_and_labels, columns=["ID", "Dialog", *CLASSIFIER_LABELS])

  # Save the new dataframe
  save_dataframe(new_df, filename.replace(".csv", ".v2.csv"))

if __name__ == "__main__":
  main("train_sent_emo.csv")
  main("test_sent_emo.csv")