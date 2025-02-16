import json

def read_data(data_path):
    list_data_dict = None
    with open(data_path) as f:
        list_data_dict = [json.loads(line) for line in f]
    return list_data_dict
