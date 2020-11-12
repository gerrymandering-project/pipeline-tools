import json
import sys

# Loads JSON config file
if len(sys.argv) != 2:
    print("Provide single filename")
    sys.exit()
else:
    configFileName = sys.argv[1]

with open(configFileName, 'r') as json_file:
    try:
        global file
        file = json.load(json_file)
    except:
        print("Unable to load JSON file")
        sys.exit()

print(len(file['nodes']))