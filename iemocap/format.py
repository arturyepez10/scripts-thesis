import pandas as pd
import json

IEMOCAP_LABELS = ["happiness", "anger", "sadness", "frustration", "neutral state", "excitement", "fear", "surprise", "other"]
CLASSIFIER_LABELS = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

"""
----------------------------
RULES OF EMOTION ASSIGNMENT
----------------------------
Due to differences in the emotions portrayed in the different datasets used for the thesis project, we have come up with a new minimized
set of emotions (based on the principal dyad of the Plutchik's wheel of emotions) that we will use for the final classification of the text. 

The end result of available emotions will follow the ones depicted in the CLASSIFIER_LABELS list.

Emotions will be assigned as follows:

- Hapiness: joy, trust
- Anger: anger
- Sadness: sadness
- Frustration: anger, sadness
- Neutral state: 
- Excitement: joy, anticipation
- Fear: fear
- Surprise: surprise
- Other:
"""

def load_json(path: str):
  with open(path, "r") as file:
    return json.load(file)

def save_dataframe(df: pd.DataFrame, filename: str):
  df.to_csv("out/" + filename, sep='\t', index=False)

def pre_process(json_data: dict):
  data = []
  error = []

  for session in json_data:
    for id in json_data[session]:
      # We manually save the row with id, dialog and emotions present
      row = []

      row.append(id)
      row.append(json_data[session][id]["dialog"])

      # There may be some cases were the emotions are not present, we discard them
      if "emotions" not in json_data[session][id]:
        error.append(id)
        continue

      row.append(json_data[session][id]["emotions"])

      data.append(row)

  return data, error

def get_column(label: str, data: list):
  index = 0

  if label == "id":
    index = 0
  elif label == "dialog":
    index = 1
  elif label == "emotions":
    index = 2

  return [row[index] for row in data]

def convert_row(row: list[1 | 0]) -> list[1 | 0]:
  new_row = [0 for _ in range(0, len(CLASSIFIER_LABELS))]

  # We assign it according to the end result assignments we set up as end result
  assignment_index = []

  for element in row:
    # Manually go through the labels and parse them to the new emotions/labels
    # while ignoring the 'neutral state' and 'other' emotions
    if element == "Happiness":
      assignment_index.append(0)
      assignment_index.append(2)
    elif element == "Anger":
      assignment_index.append(5)
    elif element == "Sadness":
      assignment_index.append(1)
    elif element == "Frustration":
      assignment_index.append(5)
      assignment_index.append(1)
    elif element == "Excited":
      assignment_index.append(0)
      assignment_index.append(7)
    elif element == "Fear":
      assignment_index.append(4)
    elif element == "Surprise":
      assignment_index.append(6)

  # We assign the new indexes to the new resulting row
  for new_index in assignment_index:
    new_row[new_index] = 1

  return new_row

def main(filename: str):
  # We process the dataset from the original format to the 3 columns (id, text, target)
  df = load_json("./out/" + filename)
  data, error = pre_process(df)

  if len(error) > 0:
    print("Amount of rows malformed with errors: ", len(error))

  # Obtain individual columns
  dialog_id_column = get_column("id", data)
  dialog_column = get_column("dialog", data)
  target_column = get_column("emotions", data)

  # Convert the target column to the new format
  new_target_column = [convert_row(row) for row in target_column]

  # Join the text and the labels in a new dataframe
  text_and_labels = [
    [
      dialog_id_column[index],
      dialog_column[index], 
      *new_target_column[index]
    ] for index in range(0,int(len(new_target_column)))
  ]
  new_df = pd.DataFrame(text_and_labels, columns=["ID", "Tweet", *CLASSIFIER_LABELS])

  # Save the new dataframe
  save_dataframe(new_df, filename.replace(".json", ".csv"))
  

if __name__ == "__main__":
  main("iemocap.json")
  