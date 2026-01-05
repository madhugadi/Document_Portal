import yaml

def load_config(config_path: str = "config/config.yaml") -> dict:

    """
    Loads a YAML configuration file and returns its contents as a dictionary.

    :param config_path: Path to the YAML configuration file
    :return: Dictionary containing the configuration
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config

load_config(config_path="config/config.yaml")
