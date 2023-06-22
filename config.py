# Read from config.json file

import json

def load_kohonen_config():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['kohonen']

def load_oja_config():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['oja']
    
def load_hopfield_config():
    with open('config.json') as json_file:
        config = json.load(json_file)
        return config['hopfield']