import numpy as np
import librosa, os
from scipy.signal import correlate
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft

import matplotlib.pyplot as plt

class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt='phat', epsilon=0.001, beta=None):
        super().__init__()

        ''' GCC implementation based on Knapp and Carter,
        "The Generalized Correlation Method for Estimation of Time Delay",
        IEEE Trans. Acoust., Speech, Signal Processing, August, 1976 '''

        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y):

        n = x.shape[-1] + y.shape[-1]

        # Generalized Cross Correlation Phase Transform
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        if self.filt == 'phat':
            phi = 1 / (torch.abs(Gxy) + self.epsilon)

        elif self.filt == 'roth':
            phi = 1 / (X * torch.conj(X) + self.epsilon)

        elif self.filt == 'scot':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            phi = 1 / (torch.sqrt(X * Y) + self.epsilon)

        elif self.filt == 'ht':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            gamma = Gxy / torch.sqrt(Gxx * Gxy)
            phi = torch.abs(gamma)**2 / (torch.abs(Gxy)
                                         * (1 - gamma)**2 + self.epsilon)

        elif self.filt == 'cc':
            phi = 1.0

        else:
            raise ValueError('Unsupported filter function')

        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(
                    Gxy * torch.pow(phi, self.beta[i]), n))

            cc = torch.cat(cc, dim=1)

        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, :max_shift+1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat(
                (cc[:, :, -max_shift:], cc[:, :, :max_shift+1]), dim=-1)

        return cc

# case3-2
mic_locs = np.array([[0, 0], [9, 0], [3, 10], [12, 10]]).T
source_loc = np.array([2.5,2.5]) 
room_dim = [12,10]
c = 343 # m/s
fs = 22050  # 샘플링 레이트

folder = "C:/GCC-PHAT/위치추정_데이터_및_마이크배치/data/case2-3/uses"
audio = os.listdir(folder)
sig1, _ = librosa.load(folder + '/' + audio[0], sr = fs)  # 1초 신호
sig2, _ = librosa.load(folder + '/' + audio[1], sr = fs)
sig3, _ = librosa.load(folder + '/' + audio[2], sr = fs)
sig4, _ = librosa.load(folder + '/' + audio[3], sr = fs)

x = torch.cat((torch.tensor(sig1)[None, :], 
               torch.tensor(sig2)[None, :], 
               torch.tensor(sig3)[None, :], 
               torch.tensor(sig4)[None, :]), 
              dim=0)
print(x.shape)

# Calculate the true TDOA 
delays = []
for pairs in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
    # Calculate the distance from the source to each microphone in the pair
    d1 = np.linalg.norm(mic_locs[:, pairs[0]] - source_loc)
    d2 = np.linalg.norm(mic_locs[:, pairs[1]] - source_loc)
    # Calculate the difference in distances
    d = d1 - d2
    # Convert distance difference to time difference
    tdoa = d / c
    # Convert time difference to sample difference
    delays.append(tdoa * fs)

print("The true TDOAs are " + str(delays))
#max_tau = max_distance * sampling_rate / sound_speed

max_tau = 964
gcc = GCC(max_tau)
# ngcc = NGCCPHAT(max_tau, 'classifier', True, sig_len, 128, fs)

# # Load the model weights
# ngcc.load_state_dict(torch.load(
#         "experiments/ngccphat/model.pth", map_location=torch.device('cpu')))
# ngcc.eval()

gcc_delays = []
for i , pairs in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    x1 = x[pairs[0]].unsqueeze(0)
    x2 = x[pairs[1]].unsqueeze(0)
    cc = gcc(x1, x2).squeeze()
    cc = cc / torch.max(cc)

    inds = range(-max_tau, max_tau+1)
    plt.figure()
    plt.plot(inds, cc, label='gcc-phat')
    plt.legend()

    shift_gcc = float(torch.argmax(cc, dim=-1)) - max_tau
    
    gcc_delays.append(shift_gcc)

    plt.scatter(shift_gcc, 1.0, marker='*')
    plt.title("Correlation between mic " + str(pairs[0]) + " and " + str(pairs[1]))
    plt.show()

    print("True TDOA (in samples): " + str(delays[i]))
    print("GCC-PHAT estimate: " + str(shift_gcc))  

"""
find the source position using multilateration
"""

def loss(x, mic_locs, tdoas):
    return sum([(np.linalg.norm(x - mic_locs[:, pairs[0]]) - \
                    np.linalg.norm(x - mic_locs[:, pairs[1]]) - \
                    tdoas[i] / fs * c) ** 2 for i, pairs in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])])

guess = [0, 0]
bounds = ((0, room_dim[0]), (0, room_dim[1]))
xhat_gcc = minimize(loss, guess, args=(mic_locs[:2], gcc_delays), bounds=bounds).x

print("Grount truth position: " + str(source_loc[:2]))
print("GCC estimate: " + str(xhat_gcc))

"""
visualizing solution as intersection between hyperbolas
"""

xx = np.linspace(0, room_dim[0], 100)
yy = np.linspace(0, room_dim[1], 100)
xx, yy = np.meshgrid(xx, yy)

def plot_hyperbolas(tdoas, name, estimate=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, pairs in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):

        plt.contour(xx, yy, (np.sqrt((xx-mic_locs[0, pairs[0]])**2 + (yy-mic_locs[1, pairs[0]])**2) 
             - np.sqrt((xx-mic_locs[0, pairs[1]])**2 + (yy-mic_locs[1, pairs[1]])**2) 
             - tdoas[i] / fs * c), [0])

    ax.scatter(mic_locs[0], mic_locs[1], c='b', label='microphones')
    ax.scatter(source_loc[0], source_loc[1], c='r', label='sound source')
    if estimate is not None:
        ax.scatter(estimate[0], estimate[1], c='g', label='estimate', marker='*', s=200)
    ax.set_xlim([0, room_dim[0]])
    ax.set_ylim([0, room_dim[1]])
    plt.title('From above')
    plt.legend()
    plt.title(name)
    plt.show()
    
plot_hyperbolas(delays, 'Ground truth')
plot_hyperbolas(gcc_delays, 'GCC-PHAT', xhat_gcc)