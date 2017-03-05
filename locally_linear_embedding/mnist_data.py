'''
This code is taken from PyBrain's MNIST tutorial with minor 
modifications, and it can be found at 
https://martin-thoma.com/classify-mnist-with-pybrain/
All rights to Martin Thomas.
'''

from struct import unpack
import gzip
from numpy import zeros, uint8, float32
import numpy as np

def get_labeled_data(imagefile, labelfile, nNumber=2000):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = np.zeros((nNumber, rows*cols), dtype=float32)
    x_temp = np.zeros((nNumber, rows, cols), dtype=float32)  # Initialize numpy array
    y = np.zeros((nNumber, 1), dtype=uint8)  # Initialize numpy array
    for i in range(nNumber):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x_temp[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    for i in range(nNumber):
        x[i,:] = x_temp[i,:].flatten()
    return (x,y)