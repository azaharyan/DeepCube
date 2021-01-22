import json
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_entry(line):
    entry = {}
    splitted = line.split(',')

    entry['exploration_const'] = re.findall(r'[-+]?\d*\.\d+|\d+', splitted[1])[0]
    entry['loss_const'] = re.findall(r'[-+]?\d*\.\d+|\d+', splitted[2])[0]
    entry['avg'] = re.findall(r'[-+]?\d*\.\d+|\d+', splitted[3])[0]

    return entry

with open('mcts_data_from_python.txt') as f:
    lines = f.readlines()
    list_entries = []
    for line in lines:
        if 'avg' not in line:
            continue

        entry = parse_entry(line)
        list_entries.append(entry)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = []
    y = []
    z = []
    for entry in list_entries:
        x.append(int(entry['exploration_const']))
        y.append(int(entry['loss_const']))
        z.append(float(entry['avg']))

    ax.scatter(x,y,z)

    plt.show()