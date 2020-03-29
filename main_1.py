import argparse, struct

import numpy as np
import matplotlib.pyplot as plt

from math import log

train_image_path = './train-images-idx3-ubyte'
train_label_path = './train-labels-idx1-ubyte'
test_image_path = './t10k-images-idx3-ubyte'
test_label_path = './t10k-labels-idx1-ubyte'

def showBytesImage(byte, row=28, col=28):
    im = [[byte[i * 28 + j] for j in range(col)] for i in range(row)]
    plt.imshow(im, cmap='gray')
    plt.show()

def read_file(path, type):
    if type == 'image':
        with open(path, 'rb') as image:
            magic_num, data_num, rows, cols = struct.unpack('>IIII', image.read(16))
            images = np.fromfile(image, dtype=np.uint8).reshape(data_num, 784)
        return images
    elif type == 'label':
        with open(path, 'rb') as label:
            magic, data_num = struct.unpack('>II', label.read(8))
            labels = np.fromfile(label, dtype=np.uint8)
        return labels

def show_result(Posterior, Label):
    error = int(0)
    for index in range(len(Label)):
        '''print("Postirior (in log scale):")
        for label, prob in enumerate(Posterior[index]):
            print(f'{label}: {prob / np.sum(Posterior[index])}')'''
        predict = np.argmax(Posterior[index])
        if predict != Label[index]:
            error +=1
        #print(f'Prediction: {predict}, Ans: {Label[index]}\n')
    print(f'Error rate: {float(error / len(Label))}')

def Naive_Bayes_Discrete(train_images, train_labels, test_images, test_labels):
    PseudoCount = int(1)
    Label_num = int(10)
    Bin_num = int(32)
    Pixel_num = int(len(train_images[0]))

    '''Bins = [label, pixel, bin]'''
    Bins = np.array([[[PseudoCount for _ in range(Bin_num)] for _ in range(Pixel_num)] for _ in range(Label_num)])
    Prior = np.array([float(0) for _ in range(Label_num)])

    for index in range(len(train_labels)):
        Prior[train_labels[index]] += 1
        for pixel in range(Pixel_num):
            Bins[train_labels[index]][pixel][train_images[index][pixel] // 8] += 1

    Prior = Prior / len(train_labels)
    Posterior = np.array([[float(0) for _ in range(Label_num)] for _ in range(len(test_images))])
    for index in range(len(test_images)):
        for label in range(Label_num):
            Posterior[index][label] += log(Prior[label])
        for label in range(Label_num):
            for pixel in range(Pixel_num):
                bin = Bins[label][pixel][test_images[index][pixel] // 8]
                Posterior[index][label] += log(float(bin / np.sum(Bins[label][pixel])))

    show_result(Posterior, test_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="toggle", type=int, help="Toggle Option")
    args = parser.parse_args()

    train_images = read_file(train_image_path, 'image')
    train_labels = read_file(train_label_path, 'label')
    test_images = read_file(test_image_path, 'image')
    test_labels = read_file(test_label_path, 'label')

    #showBytesImage(test_images[0])
    Naive_Bayes_Discrete(train_images, train_labels, test_images, test_labels)
