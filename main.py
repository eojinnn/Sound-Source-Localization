import numpy as np
import librosa, os
from scipy.signal import correlate
from scipy.optimize import minimize

def gcc_phat(sig1, sig2, fs, max_tau = None, interp = 16):
    """
    generalized cross correlation Phase Transform
    """
    n = sig1.size + sig2.size
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)

    cc = np.fft.irfft(R / np.abs(R), n = (interp * n)) # interp는 보간(interpolation) 계수로 inverse fourier시에 해상도 높이기 위해 사용
    max_shift = int(interp * n / 2)

    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau),max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1])) # cc의 중앙을 기준으로 뒤에게 앞으로 앞에게 뒤로

    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp *fs)

    return tau

def ssl(mic_positions, taus, fs):
    """
    sound source localization
    """
    def error_func(x, mic_positions, taus, fs):
        errors = []
        # 마이크 쌍 인덱스에 따라 수정된 로직
        sound_speed = 343 # 공기중 소리의 속도 m/s
        mic_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for pair_idx, (i, j) in enumerate(mic_pairs):
            d_ij = np.linalg.norm(mic_positions[i] - mic_positions[j])
            predicted_tau = d_ij / sound_speed
            errors.append((predicted_tau - taus[pair_idx])**2)
        return np.sum(errors)
    
    
    # 초기 위치 추정
    x0 = np.mean(mic_positions, axis=0)
    
    # 최적화
    res = minimize(error_func, x0, method = "BFGS", args=(mic_positions, taus, fs))
    return res.x

# case3-2
mic_positions = np.array([[0, 0], [9, 0], [3, 10], [12, 10]]) 
fs = 22050  # 샘플링 레이트

folder = "C:/GCC-PHAT/위치추정_데이터_및_마이크배치/data/case2-2/uses"
audio = os.listdir(folder)
sig1, _ = librosa.load(folder + '/' + audio[0], sr = fs)  # 1초 신호
sig2, _ = librosa.load(folder + '/' + audio[1], sr = fs)
sig3, _ = librosa.load(folder + '/' + audio[2], sr = fs)
sig4, _ = librosa.load(folder + '/' + audio[3], sr = fs)

# 각 마이크 쌍 간의 시간 지연 추정
tau12 = gcc_phat(sig1, sig2, fs)
tau13 = gcc_phat(sig1, sig3, fs)
tau14 = gcc_phat(sig1, sig4, fs)
tau23 = gcc_phat(sig2, sig3, fs)
tau24 = gcc_phat(sig2, sig4, fs)
tau34 = gcc_phat(sig3, sig4, fs)

# 스피커 위치 추정
source_position = ssl(mic_positions, np.array([tau12, tau13, tau14, tau23, tau24, tau34]), fs)
print(f"Estimated source position: {source_position}")

