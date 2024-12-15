import yaml

def load_parameters(file_path):
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters
