import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def main(args):
    array = np.load(args.path)['arr_0']
    print(array.shape)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(array[0])
    axes[0, 1].imshow(array[1])
    axes[1, 0].imshow(array[2])
    axes[1, 1].imshow(array[3])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path", type=str)
    
    args = parser.parse_args()

    main(args)