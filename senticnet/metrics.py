import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix

CLASSIFIER_LABELS = ["joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation"]

def load_dataframe(path: str) -> pd.DataFrame:
  return pd.read_csv(path, sep='\t')

def pre_process(df: pd.DataFrame) -> pd.DataFrame:
  # Obtain the text and the labels separately
  text = df.iloc[:, 1:2]
  labels = df.iloc[:, 2:].to_numpy()

  # Join the text and the labels in a new dataframe
  text_and_labels = [[text.iloc[index, 0], labels[index].tolist()] for index in range(0,int(labels.shape[0]))]

  return pd.DataFrame(text_and_labels, columns=["text", "target"])

def show_confusion_matrix(
  confusion_matrix: np.ndarray,
  labels: np.ndarray
):
  tn = confusion_matrix[:, 0, 0]
  tp = confusion_matrix[:, 1, 1]
  fn = confusion_matrix[:, 1, 0]
  fp = confusion_matrix[:, 0, 1]

  print()
  print(f"{'Label:':<15}TN \tTP \tFN \tFP")
  print(f"{'':<15}\t----\t----\t----\t----")
  for i, label in enumerate(labels):
    print(f"{label:<15}{tn[i]:>4} \t{tp[i]:>4} \t{fn[i]:>4} \t{fp[i]:>4}")
    print()

def main(filename_og: str, filename_pred: str):
  # We load the original and the predicted dataframes
  dataset_og = load_dataframe(filename_og)
  dataset_pred = load_dataframe(filename_pred)

  df_og = pre_process(dataset_og)
  df_pred = pre_process(dataset_pred)

  # We check if the two dataframes have the same length
  if len(df_og) != len(df_pred):
    print(f"[ERROR] The two dataframes do not have the same length: {len(df_og)} vs {len(df_pred)}")
    return

  # We get the labels from the original dataframe
  y_true = df_og["target"].tolist()
  y_pred = df_pred["target"].tolist()

  # We print the classification report
  report = classification_report(
    y_true,
    y_pred,
    target_names=CLASSIFIER_LABELS,
    zero_division=0
  )
  print(report)

  # We show the confusion matrix in a readable format
  confusion_matrices = multilabel_confusion_matrix(
    y_true,
    y_pred
  )

  show_confusion_matrix(confusion_matrices, CLASSIFIER_LABELS)

if __name__ == "__main__":
  # main("out/2018-E-c-En-test-gold.v2.txt", "out/senticnet_sem_eval_test.csv")
  # main("out/iemocap-test.csv", "out/senticnet_iemocap_test.csv")
  main("out/test_sent_emo.v2.csv", "out/senticnet_meld_test.csv")