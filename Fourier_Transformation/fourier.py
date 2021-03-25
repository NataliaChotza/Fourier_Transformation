import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fourier(data):
    data = np.asarray(data, dtype=float)
    amount_of_data = len(data)
    tab = np.arange(amount_of_data)
    table_of_data = tab.reshape((amount_of_data, 1))  # tab.reshape(liczba wierszy,liczba kolumn)
    E = np.exp(-2j * np.pi * table_of_data * tab / amount_of_data)  # wykładnik/ inaczej  e^x
    return np.dot(E, data)  # funkcja suma mnożenia E*data


def inverse_fourier(data):
    data = np.asarray(data, dtype=float)
    amount_of_data = len(data)
    n = np.arange(amount_of_data)
    k = n.reshape((amount_of_data, 1))
    M = np.exp(2j * np.pi * k * n / amount_of_data)
    return 1 / amount_of_data * np.dot(M, data)


def build_show(labelx, labely, title, data_to_show):
    plt.plot(data_to_show)
    plt.title(title)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.grid()
    plt.show()


def main(args):
    data = pd.read_csv(args.input_file)
    build_show('Time[s]', 'Amplitude', "Sygnały wejściowe przed Transformatą Fouriera", data)

    fourier_data = fourier(data)
    build_show('Frequency[kHz]', 'Amplitude', 'Sygnały wejściowe po Transformacie Fouriera', np.abs(fourier_data))
    # abs to jest wartosc bezwzględna plot rysuje linię od x=[0,n-1] do y = np.abs(fourier_data)

    fourier_data_without_noises = np.copy(fourier_data)
    fourier_data_without_noises[abs(fourier_data) < 200] = 0  # wartości <200 =0
    build_show('Frequency[kHz]', 'Amplitude', 'Sygnały po oddzieleniu szumów', np.abs(fourier_data_without_noises))

    inverse = inverse_fourier(fourier_data_without_noises)
    build_show('Time[s]', 'Amplitude', 'Sinusoida sygnałów', inverse)

    noises = np.subtract(data, inverse)
    build_show('Time[s]', 'Amplituda', 'Szumy', abs(noises))


def parse_arguments():  # przetwarzanie argumentów z linii poleceń
    parser = argparse.ArgumentParser(description='Implementation fourier transformation')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='csv file with time series')
    parser.add_argument('-m', '--max_frequency', type=float, default=1000,
                        help='maksimum value of frequency that  will be tested')
    parser.add_argument('-fs', '-- frequency_step', type=float, default=1.0,
                        help='step betweeen two analized frequencies')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())

# co ile herców sprawdzamy czestoliwość
