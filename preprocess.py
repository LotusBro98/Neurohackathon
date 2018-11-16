import json
import pickle
import numpy as np

datafile = open("datasets/238.json", "rt")
datastr = datafile.read()
datafile.close()

datastr = datastr.replace("}\n  {\"sample\"", "},\n  {\"sample\"")

dataset = json.loads(datastr)
dataset = dataset['DATABLOCK']

dataset = dataset[50000:]

features = []
labels = []
Nhistory = 1
for i in range(Nhistory, len(dataset)):
    inData = dataset[i - Nhistory: i]
    target = True
    for entry in inData:
        if entry['data'][0] != entry['data'][1]:
            target = False

    #if target == False:
    #    continue

    features.append(
        np.vstack([
            np.asarray(entry['data'][2:], dtype=np.float)
            for entry in inData
        ])
    )

    print(i, "/", len(dataset))

    labels.append(inData[-1]['data'][0])

features = np.asarray(features)
labels = np.asarray(labels)

dataset = {
    'features': features,
    'labels': labels,
    'Nhistory': Nhistory
}

pickle.dump(dataset, open("datasets/data.pickle", "wb+"))
