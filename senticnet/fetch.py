from senticnet.adapter import SenticNetAdapter
import argparse
import pandas as pd

def write_to_file(file_path: str, data: list):

  with open(file_path, 'a') as file:
    for item in data:
      file.write(f"{item}\t")

    file.write("\n")

def main(parser: argparse.ArgumentParser):
  args = parser.parse_args()

  # Get the values for the arguments needed
  file_path: str = args.file
  column_name: str = args.name
  batch_size: int = args.batch_size
  offset: int = args.offset

  print(f"File path: {file_path}")
  print(f"Batch size: {batch_size}")
  print(f"Offset: {offset}")

  # We initiate the SenticNet adapter
  adapter = SenticNetAdapter()

  # Read the file
  dataframe = pd.read_csv(file_path, sep='\t')

  # Check if the offset and the batch size are valid
  if offset > len(dataframe):
    print(f"[ERROR] Offset {offset} is greater than the number of rows in the file.")
    return
  
  if offset + batch_size > len(dataframe):
    print(len(dataframe), offset + batch_size)
    print(f"[ERROR] Offset {offset} + batch size {batch_size} is greater than the number of rows in the file.")
    return
  
  text_column = dataframe[column_name].tolist()

  # Get the current batch
  batch = text_column[offset : offset + batch_size]

  # Process the batch and then save the result into the selected file
  for index, sentence in enumerate(batch):
    # SenticNet API call and parse the response
    emotions_list = adapter.get_emotions(sentence)
    emotions = " ".join(emotions_list)

    write_to_file("out/senticnet.txt", [f"{index + offset}", f"{sentence}", f"{emotions}"])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fetcher for SenticNet emotions | SENTI-Lib")

  parser.add_argument(
    "-f",
    "--file",
    help="File path to the text file containing the sentences to be processed",
    action='store',
    required=True
  )

  parser.add_argument(
    "-n",
    "--name",
    help="Name of the text column with text to be queried ",
    action='store',
    default="Dialog",
    required=True
  )

  parser.add_argument(
    "-b",
    "--batch-size",
    help="Batch size used for querying the SenticNet API",
    action='store',
    type=int,
    default=100
  )

  parser.add_argument(
    "-o",
    "--offset",
    help="Offset used for starting point for the queries the SenticNet API",
    action='store',
    type=int,
    default=0
  )

  main(parser)