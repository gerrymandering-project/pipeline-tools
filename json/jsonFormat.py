import json
import sys

filename = sys.argv[1]

with open(filename, 'r') as f:
    data = json.load(f)

with open(filename, 'w') as f:
    json.dump(data, f, indent=2)