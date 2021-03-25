import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import
def fourier(x):
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def inverse_fourier(x):
    x = np.asarray(x, dtype=float)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return 1 / N * np.dot(M, x)

def main(args):
    data = pd.read_csv(args.input_file).values.reshape(1023)
    res = fourier(data)
    plt.plot(np.abs(res))
    plt.show()
    plt.plot(data)
    plt.show()
    res1 = np.copy(res)
    # res1[abs(res) < 60] = 0
    plt.plot(np.abs(res1))
    plt.show()
    plt.plot(inverse_fourier(res1))
    plt.show()
    #
    # plt.plot(fourier(res1))
    # plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="This is a fourier transform")
    parser.add_argument('-i', '--input_file', required=True, type=str, help='Path to csv file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
