import json

def map_labels(name):
    with open(name, 'r') as f:
        class_names = json.load(f)
    return class_names