import json

import torch.optim

def save_config(config, path="config.json"):
    """
    Saves configuration to a JSON file.
    
    Args:
        config (dict): Dictionary containing configuration info.
        path (str, optional): Path to save JSON file. Defaults to "dataset.json".
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=4)

def load_config(path: str):
    """
    Loads configuration dict from a JSON file.
    
    Args:
        path (str, optional): Path to JSON file configuration.

    Returns:
        dict: Dictionary containing configuration info.
    """
    with open(path, "r") as f:
        return json.load(f)

def create_optimizer(model, optimizer_config):
    opt_type = optimizer_config["type"]
    params = optimizer_config["params"]
    if opt_type == "AdamW":
        return torch.optim.AdamW(model.parameters(), **params)
    if opt_type == "SGD":
        return torch.optim.SGD(model.parameters(), **params)
    