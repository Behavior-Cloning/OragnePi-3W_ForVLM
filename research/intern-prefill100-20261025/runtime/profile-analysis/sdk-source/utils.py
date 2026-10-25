import json
import numpy as np

def load_json(file):
    with open(file, 'r') as f:
        dict = json.load(f)
    return dict

def load_bin(file):
    with open(file, 'rb') as f:
        data = f.read()
    return data

def parse_lid(lid):
    name, uid = lid.rsplit('_', 1)
    return name, int(uid)

def list_add(list1, list2):
    if len(list1) == 0 and len(list2) != 0:
        return list2
    if len(list2) == 0 and len(list1) != 0:
        return list1
    if len(list1) == 0 and len(list2) == 0:
        return []
    ret = np.add(list1, list2)
    return list(ret)
