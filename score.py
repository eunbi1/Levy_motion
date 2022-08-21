import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from functools import partial


class score_fn():

    def __init__(self, alpha=1.9, sigma1=0, sigma2=1, t0=5, Fs=10):
        self.alpha = alpha
        self.Fs = Fs
        self.sigma_1 = sigma1
        self.sigma_2 = sigma2
        self.t0 = t0
        self.Fs = Fs
        self.t = t = np.arange(-t0, t0, 1. / Fs)
        self.f = f = np.linspace(-Fs / 2, Fs / 2, len(t))
        self.scores = self.score_function(self.g1, self.g2, t, f)

    def score_function(self, g1, g2, t, f):
        approx1 = np.fft.fftshift(np.fft.fft(g1(t)) * np.exp(2j * np.pi * f * self.t0) * 1 / self.Fs)
        approx2 = np.fft.fftshift(np.fft.fft(g2(t)) * np.exp(2j * np.pi * f * self.t0) * 1 / self.Fs)
        approx_score = np.divide(approx2, approx1)
        return np.divide(approx2, approx1)

    def g1(self, x):
        return np.exp(-1 / 2 * (2 * np.pi * x * self.sigma_1) ** 2) * np.exp(
            -np.power(2 * np.pi * np.abs(x * self.sigma_2), self.alpha))

    def g2(self, x):
        return (-2j * np.pi * x) * np.exp(-1 / 2 * (2 * np.pi * x * self.sigma_1) ** 2) * np.exp(
            -np.power(2 * np.pi * np.abs(x * self.sigma_2), self.alpha))

    def point_evaluation(self, x):
        # x : number
        # output: number
        if x <= self.Fs / 2:

            k = np.argmin(np.abs(self.f - x), axis=0)
            return np.real(self.scores[k])
        else:

            Fs = Fs = 2 * x + 2
            f = np.linspace(-Fs / 2, Fs / 2, len(self.t))
            scores = self.score_function(self.g1, self.g2, self.t, f)

            k = np.argmin(np.abs(f - x))

            return np.real(scores[k])

    def evaluation(self, x):

        result = np.zeros(x.shape, dtype=complex)
        xxx = np.nditer(x, flags=['multi_index'])

        for element in xxx:
            i = xxx.multi_index
            result[i] = self.point_evaluation(element)

        return result