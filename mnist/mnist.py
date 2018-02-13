# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import struct

def loadMNIST(dataset, callback):
    # Mnist Files
    files = {
        'train_images': 'train-images-idx3-ubyte',
        'train_labels': 'train-labels-idx1-ubyte',
        'test_images': 't10k-images-idx3-ubyte',
        'test_labels': 't10k-labels-idx1-ubyte',
    }

    # Datasets to load
    if dataset == 'training':
        fname_img = files.get('train_images')
        fname_lbl = files.get('train_labels')
    elif dataset == 'testing':
        fname_img = files.get('test_images')
        fname_lbl = files.get('test_labels')
    else:
        raise ValueError("Dataset must be 'training' or 'testing'")

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack('>II', flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    for i in range(len(lbl)):
        yield get_img(i)

    callback()


# training_data = list(loadMNIST('training', lambda: print("Done")))
# print(len(training_data))
# label, pixles = training_data[0]
# print(label)
# print(pixles.shape)
# show(pixles)
