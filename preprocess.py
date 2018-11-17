import json
import pickle
import numpy as np

datafile = open("datasets/alexpan1.json", "rt")
datastr = datafile.read()
datafile.close()

#datastr = datastr.replace("}\n  {\"sample\"", "},\n  {\"sample\"")

dataset = json.loads(datastr)
dataset = dataset['DATABLOCK']

#dataset = dataset[10000:50000]

features = []
labels = []
Nhistory = 100

for i in range(Nhistory, len(dataset)):
    inData = dataset[i - Nhistory: i]

    zeros = False
    for line in inData:
        if (line['data'][0] == 0):
            zeros = True

    target = False
    for j in range(1, min(Nhistory, 10)):
        line1 = inData[j - 1]
        line2 = inData[j]
        if (line2['data'][0] == line2['data'][1] and line2['data'][1] != line1['data'][1]):
            target = True

    if (zeros or not target):
        continue

    #
    # if (inData[-1]['data'][0] == inData[-1]['data'][1]):
    #     label = 1
    # else:
    #     label = 0

    label = inData[-1]['data'][0]

    features.append(
        np.vstack([
            np.asarray(entry['data'][2:], dtype=np.float)
            for entry in inData
        ])
    )

    labels.append(label)

    print(i, "/", len(dataset))

features = np.asarray(features)
labels = np.asarray(labels)

dataset = {
    'features': features,
    'labels': labels,
    'Nhistory': Nhistory
}

pickle.dump(dataset, open("datasets/data.pickle", "wb+"))
