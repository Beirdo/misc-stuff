#! /usr/bin/python
# vim:ts=4:sw=4:ai:et:si:sts=4

import sys
from scipy.io.wavfile import read, write
from numpy.fft import rfft, irfft
import numpy as np
from pylab import *
import scipy.signal as signal

def fbin(freq,binsize):
    return int(freq/binsize)

def fpwr(value):
    return np.real(value * np.conj(value))

def fthresh(y,yout,freq,binsize,threshold,factor):
    binnum = fbin(freq,binsize)
    value = y[binnum] * factor
    power = fpwr(value)
    if power < threshold:
        yout[binnum] = 0
    else:
        yout[binnum] = np.sqrt(power)

def mfreqz(b,a=1):
    w,h = signal.freqz(b,a,1048576)
    h_dB = 20 * log10 (abs(h))
    subplot(211)
    plot(w/max(w),h_dB)
    ylim(-150, 5)
    ylabel('Magnitude (db)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Frequency response')
    subplot(212)
    h_Phase = unwrap(arctan2(imag(h),real(h)))
    plot(w/max(w),h_Phase)
    ylabel('Phase (radians)')
    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    title(r'Phase response')
    subplots_adjust(hspace=0.5)

def impz(b,a=1):
    impulse = repeat(0.,50); impulse[0] =1.
    x = arange(0,50)
    response = signal.lfilter(b,a,impulse)
    subplot(211)
    stem(x, response)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Impulse response')
    subplot(212)
    step = cumsum(response)
    stem(x, step)
    ylabel('Amplitude')
    xlabel(r'n (samples)')
    title(r'Step response')
    subplots_adjust(hspace=0.5)

def normsig(x):
    print "Normalizing signal from",x.dtype,"to float64"
    iinfo = np.iinfo(x.dtype)
    midval = (iinfo.max + iinfo.min) / 2.0
    numerator = (iinfo.max - midval) * 1.0
    y = np.zeros(len(x), dtype=np.float64)
    for i in range(0, len(x)-1):
        y[i] = (x[i] + midval) / numerator
    return y

def unnormsig(x):
    print "Normalizing signal from float64 to int16"
    iinfo = np.iinfo(np.int16)
    y = np.zeros(len(x), dtype=np.int16)
    for i in range(0, len(x)-1):
        y[i] = int(x[i] * iinfo.max)
    return y

def gcd(a,b):
    mn = min(a,b)
    mx = max(a,b)
    while mn > 0:
        md = mx % mn
        mx = mn
        mn = md
    return mx

def resample(oldrate,newrate,x,n,dtype,factor):
    print "Resampling from",oldrate,"Hz to",newrate,"Hz, amplification factor",factor
    rategcd = gcd(oldrate,newrate)
    uprate = newrate / rategcd
    dnrate = oldrate / rategcd

    oldcount = len(x)
    midcount = oldcount * uprate
    newcount = midcount / dnrate

    print "Upsampling by",uprate
    if uprate == 1:
        yout = np.asarray(x, dtype=dtype)
    else:
        yout = np.zeros(midcount, dtype=dtype)
        for i in range(0, oldcount-1):
            yout[i * uprate] = x[i] * uprate

    wl = min(1.0/uprate,1.0/dnrate)
    print "Antialias filtering at",wl
    
    midrate = oldrate * uprate
    y = firfilter(0, (midrate * wl) / 2.0, midrate, yout, n)

    print "Downsampling by",dnrate
    if dnrate == 1:
        yout = np.asarray(y, dtype=dtype)
    else:
        yout = np.zeros(newcount, dtype=dtype)
        for i in range(0, newcount-1):
            yout[i] = y[i * dnrate] * factor

    return yout

def firfilter(flow, fhigh, rate, x, n):
    wpl = (flow * 2.0) / rate
    wph = (fhigh * 2.0) / rate

    print "Filtering Band",flow,"Hz to",fhigh,"Hz at",rate,"Hz sampling"
    print "Freq:",wpl,"to",wph

    if wpl > 0:
        a =   signal.firwin(n, cutoff=wpl, window='blackmanharris')
        b = - signal.firwin(n, cutoff=wph, window='blackmanharris')
        b[n/2] = b[n/2] + 1
        d = -(a+b)
        d[n/2] = d[n/2] + 1
    else:
        d = signal.firwin(n, cutoff=wph, window='blackmanharris')

#    figure(firfilter.fig)
#    mfreqz(d)
    firfilter.fig += 1

    print "Running Filter"
    y = signal.lfilter(d, 1, x)
    return y
    
def mix(flo, flow, fhigh, rate, x, n):
    freqgcd = gcd(rate,flo)
    samples = rate / freqgcd
    stride  = flo  / freqgcd
    samplecount = len(x)

    print "Generating sine wave using",samples,"samples and",stride,"stride"
    lo = np.sin(np.linspace(np.pi/2.0, (3.0*np.pi)/2.0, samples, False))
    fif = np.zeros(samplecount, dtype=np.float64)

    print "Mixing with LO of",flo,"Hz, Lo:",flow,"Hz, High:",fhigh,"Hz"
    j = 0
    for i in range(0, samplecount-1):
        fif[i] = x[i] * lo[j]
        j = (j + stride) % samples

    print "Filtering at IF base of",flo,"Hz"
    y = firfilter(flow, fhigh, rate, fif, n)
    return y

if len(sys.argv) < 3:
    print "Usage: ", sys.argv[0], "inputfile.wav outputfile.wav"
    sys.exit(2)

infile = sys.argv[1]
outfile = sys.argv[2]

rate, data = read(infile)

# Setup all the parameters
samplecount = len(data)
factor = 1

print "Input File:", infile
print "Input Samples:", samplecount, "Input Rate:", rate
print "Input Type:", data.dtype, "Factor:", factor

flow  = 1200
fhigh = 3800

n = 91
firfilter.fig = 1
midrate = rate

# Normalize to -1.0..1.0 as float64
x = normsig(data)

# Bandpass filter (lowpass if starting at 0)
y = firfilter(flow, fhigh, rate, x, n)
fbw = fhigh - flow

if flow > 0 and flow + flow <= fhigh:
    if midrate < 40000:
        midrate = 40000
    flo = 20000 - fbw
    print "Translating to",flo,"Hz to avoid aliasing"
    if rate != midrate:
        yout = resample(rate,midrate,y,n,np.float64,factor)
        y = yout

    flo  -= flow
    yout = mix(flo, flo + flow, flo + fhigh, midrate, y, n)
    y = yout

    flow  += flo
    fhigh += flo

if flow > 0:
    print "Translating to baseband"
    yout = mix(flow, 0, fbw, midrate, y, n)
    y = yout

newrate = 8000
print "Decimating to",newrate,"Hz"
yout = resample(midrate,newrate,y,n,np.float64,1.0)

# Renormalize to int16
y = unnormsig(yout)

print "Writing"
write(outfile,newrate,np.asarray(y, dtype=np.int16))

#show()

sys.exit(0)

