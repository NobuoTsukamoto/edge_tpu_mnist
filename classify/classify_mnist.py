#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Edge TPU MNIST classify.

    Copyright (c) 2019 Nobuo Tsukamoto

    This software is released under the MIT License.
    See the LICENSE file in the project root for more information.
"""

import sys
import argparse
import io
import time
import random

from mnist_data import load_mnist

import numpy as np
from edgetpu.basic.basic_engine import BasicEngine
from PIL import Image

def display(img):
    print('+----------------------------+')
    for y in range(0, 28):
        sys.stdout.write('|')
        for x in range(0, 28):
            if img[y*28+x] < 128:
                sys.stdout.write(" ")
            elif img[y*28+x] < 200:
                sys.stdout.write("*")
            else:
                sys.stdout.write("+")
        sys.stdout.write('|\n')
    print('+----------------------------+')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
            '--data_set', help='Whether to use \"train\" or \"test\" dataset.',
            type=str, default='test')
    parser.add_argument(
            '--display', help='Whether to display the image on the consol3.',
            default=True)
    args = parser.parse_args()

    # load mnist data.
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

    # Initialize engine.
    engine = BasicEngine(args.model)

    # get mnist data.
    if args.data_set == 'train':
        index = random.randint(0, len(x_train) - 1)
        img = x_train[index]
        label = t_train[index]
    else:
        index = random.randint(0, len(x_test) - 1)
        img = x_test[index]
        label = t_test[index]

    # display image.
    if args.display:
        display(img)

    # Run inference.
    array_size = engine.required_input_array_size()

    print('\n------------------------------')
    print('Run infrerence.')
    print('  required_input_array_size: ', array_size)
    print('  input shape: ', img.shape)

    result = engine.RunInference(img)
    print('------------------------------')
    print('Result.')
    print('Inference latency: ', result[0], ' milliseconds')
    print('Output tensor: ', result[1])
    print('Inference: ', np.argmax(result[1]))
    print('Label    : ', label)

if __name__ == '__main__':
  main()

