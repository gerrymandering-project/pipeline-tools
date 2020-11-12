import json
from collections import defaultdict

def getTuple(arr):
    n = len(arr)
    arr.sort()
    return (arr[n // 1000], arr[n // 100], arr[n // 20], sum(arr) / n, arr[~(n // 20)], arr[~(n // 100)], arr[~(n // 1000)])

def openFile(filename):

    dict_ = defaultdict(list)
    for i in range(25):
        with open(filename + str(i) + '.json') as f:
            data = json.load(f)
        for key in data:
            dict_[key].append(getTuple(data[key]))
    print(dict_)


def main():
    openFile('./generated_data/run_statistics_NC_GERRY_RUN_STATS')

main()