import cv2
import numpy as np

def homomorphic_filter_yuv(img, sigma=10, gamma1=0.3, gamma2=1.5):
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y = img_YUV[:,:,0]
    rows, cols = y.shape
    imgLog = np.log1p(np.array(y, dtype='float') / 255)
    M, N = 2*rows + 1, 2*cols + 1
    (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M))
    Xc, Yc = np.ceil(N/2), np.ceil(M/2)
    gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2
    LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
    HPF = 1 - LPF
    LPF_shift, HPF_shift = np.fft.ifftshift(LPF.copy()), np.fft.ifftshift(HPF.copy())
    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))
    img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]
    img_exp = np.expm1(img_adjusting)
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_out = np.array(255*img_exp, dtype = 'uint8')
    img_YUV[:,:,0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return result
