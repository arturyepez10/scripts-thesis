from dotenv import dotenv_values

class ConfigEnvars:

  def __init__(self, sentic_net_url: str, sentic_net_api_key: str, sentic_net_language):
    self.SENTIC_NET_URL = sentic_net_url
    self.SENTIC_NET_API_KEY = sentic_net_api_key
    self.SENTIC_NET_LANGUAGE = sentic_net_language

loaded_config = dotenv_values(".env")



config = ConfigEnvars(
  loaded_config['SENTIC_NET_URL'],
  loaded_config['SENTIC_NET_API_KEY'],
  loaded_config['SENTIC_NET_LANGUAGE']
)