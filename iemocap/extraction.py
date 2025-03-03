from glob import glob
from typing import Dict
import re
import json

BASE_PATH = "/Users/ayepez/Desktop/IEMOCAP_full_release"
SESSIONS = ["Session1", "Session2", "Session3", "Session4", "Session5"]

def get_transcription_files():
  """Get a list of all of transcription files existing in the dataset
  per session recorded.
  """
  transcription_files = []
  for session in SESSIONS:
    transcription_files.append(glob(f"{BASE_PATH}/{session}/dialog/transcriptions/*.txt"))

  return transcription_files

def get_emotion_evaluation_files():
  """Get a list of all of emotion evaluation categories files existing
  in the dataset per session recorded.
  """
  emotion_evaluation_files = []
  for session in SESSIONS:
    emotion_evaluation_files.append(glob(f"{BASE_PATH}/{session}/dialog/EmoEvaluation/Categorical/*.txt"))

  return emotion_evaluation_files

def get_dialog(path: str):
  """Get the dialogs from a single transcription file.

  It saves then as a dictionary where the key is the dialog_id and the value itself.

  The dictionary is structured as follows:
    {
      dialog_id: dialog
    }
  """
  dialogs: Dict[str, str] = {}
  with open(path, "r") as file:
    for line in file:
      # We split the line into: [dialog_id, timestampt, dialog]
      splitted_line = line.strip().split(" ", 2)

      # There are comments made during the session irrelevant for the dialog processings
      if (len(splitted_line) < 3):
        continue

      dialogs[splitted_line[0]] = splitted_line[2]

  return dialogs

def get_dialog_emotions(path_list: list[str]):
  """Gets the emotions categorized in the dialog for a specific session.

  Since every session usually gets multiple evaluators, the return the 
  result as a dictionary where the key is the dialog_id and the value is
  a list of emotions containing the emotiones categorized by all of the evaluators.

  The dictionary is structured as follows:
    {
      dialog_id: set([emotion_1, ..., emotion_n])
    }
  """
  emotions_dict: Dict[str, set] = { }

  for path in path_list:
    with open(path, "r") as file:
      for line in file:
        # We split the line into: [dialog_id, emotion_1, ..., emotion_n, comments]
        dialog_id, emotions = line.strip().split(" ", 1)

        # List of emotions (omitting the comments and 'Neutral state')
        emo = re.findall(":([\w]+);", emotions)

        # If the key exists or is being added for the first time then we create the set
        if dialog_id not in emotions_dict.keys():
          emotions_dict[dialog_id] = set()
          
        emotions_dict[dialog_id].update(emo)

  return emotions_dict

def extract_dialog_emotions():
  """For each one of the sessions, extract the dialog and the emotions
  categorized by all of the evaluators involved in the respectived session.

  The result is a dictionary where the key is the dialog_id and the value is another
  dictionary with the dialog and the emotions categorized by all of the evaluators.

  The dictionary is structured as follows:
    {
      session_nr: {
        dialog_id: {
          dialog: { dialog_id: dialog },
          emotions: { dialog_id: set([emotion_1, ..., emotion_n]) }
        }
      }
    }
  """
  all_information: Dict[str, Dict[str, str | set]] = { }

  # Create the dict instances for each session
  for session in SESSIONS:
    all_information[session] = { }

  # Get the transcription files
  transcription_files = get_transcription_files()

  # For each one of the sessions we fill the dictionary with the dialog
  for index, session in enumerate(SESSIONS):
    # Files for current session
    files = transcription_files[index]

    for file in files:
      # Get the dialog
      dialogs = get_dialog(file)

      # Save the dialog based on the proposed format
      for dialog_id, dialog in dialogs.items():
        # Create instance if it does not exist
        if dialog_id not in all_information[session].keys():
          all_information[session][dialog_id] = { }

        all_information[session][dialog_id]["dialog"] = dialog

  # Get the emotions files
  emotion_evaluation_files = get_emotion_evaluation_files()

  # For each one of the sessions we fill the dictionary with the emotions evaluation
  for index, session in enumerate(SESSIONS):
    # Files for current session
    files = emotion_evaluation_files[index]

    emotions = get_dialog_emotions(files)

    for dialog_id, emotions in emotions.items():
      # # Create instance if it does not exist
      # if dialog_id not in all_information[session].keys():
      #   print("Dialog ID not found: ", dialog_id)

      all_information[session][dialog_id]["emotions"] = list(emotions)

  return all_information


def save_dialog_emotions_json():
  """Saves the dialog and the emotions categorized by all of the evaluators
  into a JSON file.

  The JSON file constains the structure follows the one defined in the function
  `extract_dialog_emotions`.
  """
  all_information = extract_dialog_emotions()

  # Save the result as a JSON file with proper formatting to ensure readability
  with open("./out/iemocap.json", "w", encoding='utf-8') as file:
    json.dump(all_information, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
  save_dialog_emotions_json()