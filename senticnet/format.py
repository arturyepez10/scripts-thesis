import pandas as pd

SENTICNET_LABELS = [
  "grief",
  "sadness",
  "melancholy",
  "contentment",
  "joy",
  "ecstasy",
  "rage",
  "anger",
  "annoyance",
  "serenity",
  "calmness",
  "bliss",
  "loathing",
  "disgust",
  "dislike",
  "acceptance",
  "pleasantnes",
  "delight",
  "terror",
  "fear",
  "anxiety",
  "responsiveness",
  "eagerness",
  "enthusiasm"
]
CLASSIFIER_LABELS = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

"""
----------------------------
RULES OF EMOTION ASSIGNMENT
----------------------------
Due to differences in the emotions portrayed in the different datasets used for the thesis project, we have come up with a new minimized
set of emotions (based on the principal dyad of the Plutchik's wheel of emotions) that we will use for the final classification of the text. 

The end result of available emotions will follow the ones depicted in the NEW_CLASSIFIER_LABELS list.

Emotions will be assigned as follows:

- grief: sadness, fear
- sadness: sadness
- melancholy: sadness, anticipation
- contentment: joy
- joy: joy
- ecstasy: joy

- rage: anger
- anger: anger
- annoyance: anger
- serenity: joy, trust
- calmness: joy, trust
- bliss: joy, trust

- loathing: disgust, anger
- disgust: disgust
- dislike: disgust
- acceptance: joy, trust
- pleasantnes: joy
- delight: joy, surprise

- terror: fear, surprise
- fear: fear
- anxiety: anticipation, fear
- responsiveness: anticipation, trust
- eagerness: anticipation, joy
- enthusiasm: anticipation, joy
"""

def load_dataframe(path: str) -> pd.DataFrame:
  return pd.read_csv(path, sep='\t')

def save_dataframe(df: pd.DataFrame, filename: str):
  df.to_csv("out/" + filename, sep='\t', index=False)

def convert_row(row: str) -> list[1 | 0]:
  new_row = [0 for _ in range(0, len(CLASSIFIER_LABELS))]

  if pd.isna(row):
    return new_row

  row_array = row.split(" ")

  # We assign it according to the end result assignments we set up as end result
  assignment_index = []

  for element in row_array:
    # Manually go through the labels and parse them to the new emotions/labels
    # while ignoring the 'neutral state' and 'other' emotions
    if element == "grief":
      assignment_index.append(1)
      assignment_index.append(4)
    elif element == "sadness":
      assignment_index.append(1)
    elif element == "melancholy":
      assignment_index.append(1)
      assignment_index.append(7)
    elif element == "contentment":
      assignment_index.append(0)
    elif element == "joy":
      assignment_index.append(0)
    elif element == "ecstasy":
      assignment_index.append(0)
    elif element == "rage":
      assignment_index.append(5)
    elif element == "anger":
      assignment_index.append(5)
    elif element == "annoyance":
      assignment_index.append(5)
    elif element == "serenity":
      assignment_index.append(0)
      assignment_index.append(2)
    elif element == "calmness":
      assignment_index.append(0)
      assignment_index.append(2)
    elif element == "bliss":
      assignment_index.append(0)
      assignment_index.append(2)
    elif element == "loathing":
      assignment_index.append(3)
      assignment_index.append(5)
    elif element == "disgust":
      assignment_index.append(3)
    elif element == "dislike":
      assignment_index.append(3)
    elif element == "acceptance":
      assignment_index.append(0)
      assignment_index.append(2)
    elif element == "pleasantness":
      assignment_index.append(0)
    elif element == "delight":
      assignment_index.append(0)
      assignment_index.append(6)
    elif element == "terror":
      assignment_index.append(4)
      assignment_index.append(6)
    elif element == "fear":
      assignment_index.append(4)
    elif element == "anxiety":
      assignment_index.append(4)
      assignment_index.append(7)
    elif element == "responsiveness":
      assignment_index.append(2)
      assignment_index.append(7)
    elif element == "eagerness":
      assignment_index.append(0)
      assignment_index.append(7)
    elif element == "enthusiasm":
      assignment_index.append(0)
      assignment_index.append(7)

  # We assign the new indexes to the new resulting row
  for new_index in assignment_index:
    new_row[new_index] = 1

  return new_row

def main(filename: str):
  # We process the dataset from the original format to the 3 columns (id, text, target)
  df = load_dataframe("./senticnet/dataset/" + filename)

  # Obtain individual columns
  text_column = df["id"].tolist()
  emotions_column = df["text"].tolist()

  # Convert the target column to the new format
  new_target_column = [convert_row(row) for row in emotions_column]

  # Join the text and the labels in a new dataframe
  text_and_labels = [
    [
      index,
      text_column[index], 
      *new_target_column[index]
    ] for index in range(0, int(len(new_target_column)))
  ]
  new_df = pd.DataFrame(text_and_labels, columns=["id", "dialog", *CLASSIFIER_LABELS])

  # Save the new dataframe
  save_dataframe(new_df, filename.replace(".txt", ".csv"))
  

if __name__ == "__main__":
  main("senticnet_iemocap_test.txt")
  main("senticnet_iemocap_train.txt")
  main("senticnet_meld_test.txt")
  main("senticnet_meld_train.txt")
  main("senticnet_sem_eval_test.txt")
  main("senticnet_sem_eval_train.txt")