# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:19:45 2023

@author: User
"""


import numpy as np
import matplotlib.pyplot as plt

npts=1000

x = np.linspace(0, 10, npts, endpoint=True) 


mean = 5
sigma = 1
# sigma = np.sqrt(sigma2)

y1 = 1* np.exp(-(x-mean)**2 / (2*sigma**2))
# y1 = 1 / (1*np.pi*sigma) * np.exp(-(x)**2 / (2*sigma2))
n = 0.1*np.random.randn(npts)
y1n = y1 + n

plt.figure()
plt.plot(x,y1)
plt.title('Gaussian Distribution')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time[Sec]')
plt.grid('on')

plt.figure()
plt.plot(x,y1)
plt.plot(x,y1n)
plt.title('Gaussian Distribution with noise')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time[Sec]')
plt.legend(('Signal', 'Signal+Noise'))
plt.grid('on')

# Perform FFT
def make_fft_DS(x,t):
    
    ts = (t[-1]-t[0]) / (len(t)-1)
    Fs=1/ts

    L = len(x)
    
    spectr = np.fft.fft(x)/L; 
    spectr= np.fft.fftshift(spectr)
    dF = Fs/L; 
    
    if  L%2==0:
        f=np.arange(-1*(Fs/2), Fs/2, dF)
        # f=f[0:-1];
    else:
        f=np.arange( -1*(Fs/2 -dF/2), (Fs/2 -dF/2),dF)

    return f, spectr

# Performs non-coherent integration
def make_icoh(t, y, icoh):
    
    nptsn=int(np.floor(len(y)/icoh))
    Yi=np.empty(nptsn)
    tn=np.empty(nptsn)
    speci=np.zeros([nptsn,icoh])+1j*np.zeros([nptsn,icoh])

    # Time series splitting
    for i in range(0,icoh):
        yn=y[i*nptsn:i*nptsn+nptsn]
        tn=t[i*nptsn:(i+1)*nptsn]
        
        fi,Yi=make_fft_DS(yn,tn)
        
        # Individual spectra
        speci[:,i]=Yi
        
        # Integrated spectrum
        spectri=np.mean(abs(speci), axis=1)/np.sqrt(icoh)
        
        if len(fi)>len(spectri):
            fi=fi[:-1]
        
    return fi,spectri

# FFT of signal and signal with noise
f0,spectr0= make_fft_DS(y1,x)

f,spectr= make_fft_DS(y1n,x)

plt.figure()
plt.plot(f0,abs(spectr0))
plt.plot(f,abs(spectr))
plt.title('Gaussian Spectra')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Freq[Hz]')
plt.legend(('Signal', 'Signal+Noise'))


# Perform Non Coherent integration with different averaging
power = []
icoh = [1, 5,10,25,50,100]
for i in icoh:
    fi,spectri = make_icoh(x, y1n, i)
    
    
    power.append(np.var(abs(spectri)))
    
    plt.figure()
    plt.plot(f,abs(spectr))
    plt.plot(fi,abs(spectri),'*-')
    plt.title(str(i)+ ' incoherent integration ')
    plt.grid('on')
    plt.ylabel('Amplitude')
    plt.xlabel('Freq[Hz]')
    
# Plot effect on SNR
y1_power = np.var(abs(y1n))

snr = []
for index in range(len(power)):
    snr_t = 10*np.log10(y1_power/power[index])
    snr.append(snr_t)

plt.figure()
plt.plot(icoh,snr,'*-')
plt.grid('on')
plt.title('Number of integrations Vs SNR')
plt.xlabel('Number of integrations')
plt.ylabel('SNR')