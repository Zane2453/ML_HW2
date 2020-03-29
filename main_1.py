import argparse, struct

import numpy as np
#import matplotlib.pyplot as plt

from math import log, exp, sqrt, pi

train_image_path = './train-images-idx3-ubyte'
train_label_path = './train-labels-idx1-ubyte'
test_image_path = './t10k-images-idx3-ubyte'
test_label_path = './t10k-labels-idx1-ubyte'

'''def showBytesImage(byte, row=28, col=28):
    im = [[byte[i * 28 + j] for j in range(col)] for i in range(row)]
    plt.imshow(im, cmap='gray')
    plt.show()'''

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

def show_result(Posterior, Label, Result, Mode):
    error = int(0)
    for index in range(len(Label)):
        print("Postirior (in log scale):")
        for label, prob in enumerate(Posterior[index]):
            print(f'{label}: {prob / np.sum(Posterior[index])}')
        predict = np.argmax(Posterior[index])
        if predict != Label[index]:
            error +=1
        print(f'Prediction: {predict}, Ans: {Label[index]}\n')

    print("Imagination of numbers in Bayesian classifier:\n")

    if Mode == 'Discrete':
        for label in range(len(Result)):
            print(f'{label}:')
            row = ''
            for pixel in range(784):
                if pixel % 28 == 0:
                    row = ''
                grey = "0 " if np.argmax(Result[label][pixel]) <= 15 else "1 "
                row += grey
                if pixel % 28 == 27:
                    print(row)
            print()
    elif Mode == 'Continuous':
        for label in range(len(Result)):
            print(f'{label}:')
            row = ''
            for pixel in range(784):
                if pixel % 28 == 0:
                    row = ''
                grey = "0 " if Result[label]['mean'][pixel] <= 127 else "1 "
                row += grey
                if pixel % 28 == 27:
                    print(row)
            print()

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

    show_result(Posterior, test_labels, Bins, 'Discrete')

def Naive_Bayes_Continuous(train_images, train_labels, test_images, test_labels):
    Label_num = int(10)
    Pixel_num = int(len(train_images[0]))

    '''Gaussian = [label, {'mean', 'variance'}, pixel]'''
    Gaussian = np.array([dict() for _ in range(Label_num)])
    Prior = np.array([float(0) for _ in range(Label_num)])

    for index in range(len(train_labels)):
        Prior[train_labels[index]] += 1

    for label in range(Label_num):
        images = train_images[train_labels == label]
        Gaussian[label]['mean'] = np.mean(images, axis=0)
        Gaussian[label]['variance'] = np.var(images, axis=0)

    Prior = Prior / len(train_labels)
    Posterior = np.array([[float(0) for _ in range(Label_num)] for _ in range(len(test_images))])

    log_zero = 10 ** -10

    for index in range(10000):
        for label in range(Label_num):
            Posterior[index][label] += log(Prior[label])
        for label in range(Label_num):
            for pixel in range(Pixel_num):
                if Gaussian[label]['variance'][pixel] <= log_zero:
                    #Posterior[index][label] += log(float(log_zero))
                    continue
                else:
                    Posterior[index][label] += (((test_images[index][pixel] - Gaussian[label]['mean'][pixel]) ** 2) / (2 * Gaussian[label]['variance'][pixel]) * -1) - log(sqrt(2*pi*Gaussian[label]['variance'][pixel]))

    show_result(Posterior, test_labels, Gaussian, 'Continuous')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", dest="toggle", type=int, help="Toggle Option")
    args = parser.parse_args()

    train_images = read_file(train_image_path, 'image')
    train_labels = read_file(train_label_path, 'label')
    test_images = read_file(test_image_path, 'image')
    test_labels = read_file(test_label_path, 'label')

    #showBytesImage(test_images[0])
    if args.toggle == 0:
        Naive_Bayes_Discrete(train_images, train_labels, test_images, test_labels)
    elif args.toggle == 1:
        Naive_Bayes_Continuous(train_images, train_labels, test_images, test_labels)