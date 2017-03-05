import os
import struct

import numpy as np


def load_mnist(dataset="training", path="../data"):
    if dataset == "training":
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("Dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.array(struct.unpack("{}B".format(size), flbl.read()))
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.array(struct.unpack("{}B".format(size*rows*cols), fimg.read()))
    img = img.reshape((size, rows, cols))
    fimg.close()

    return img, lbl
