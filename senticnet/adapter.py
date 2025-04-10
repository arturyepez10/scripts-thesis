from senticnet.client import SenticNetClient
import re

class SenticNetAdapter:
  def __init__(self):
    self.client = SenticNetClient()

    self.forbidden_char = ['&', '#', ';', '{', '}']

    # Obtain the words/emotions from SenticNet response
    self.regex_expression = r"(\b\w* (\w*))"

  def _preprocess_text(self, text: str):
    value = text
    for char in self.forbidden_char:
      value = value.replace(char, ':')

    return value

  def _parse_emotions_response(self, response: str):

    if "No emotions detected" in response:
      return []

    x = re.findall(self.regex_expression, response)

    # extract the words and strip whitespaces
    words = []
    for i in x:
      words.append(i[0].strip())

    return words

  def get_emotions(self, text: str) -> list[str]:
    preprocessed_text = self._preprocess_text(text)

    emotions_res = self.client.get_emotions(preprocessed_text)

    return self._parse_emotions_response(emotions_res)