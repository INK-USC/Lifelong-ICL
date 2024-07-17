import os

def load_api_key(file_path):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError("Cannot find API key at {}.".format(file_path))
    
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key