import requests
from config import config

class SenticNetClient:  
  def __init__(self):
    self.path = config.SENTIC_NET_URL + "/" + config.SENTIC_NET_LANGUAGE + "/" + config.SENTIC_NET_API_KEY + ".py"

  def get_emotions(self, text: str):
    response = requests.get(self.path, params={"text": text})
    return response.text
