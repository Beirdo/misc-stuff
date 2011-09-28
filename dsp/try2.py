#! /usr/bin/python
# vim:ts=4:sw=4:ai:et:si:sts=4

import sys
import time
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

def normsample(x,dtype):
    iinfo = np.iinfo(dtype)
    midval = (iinfo.max + iinfo.min) / 2.0
    numerator = (iinfo.max - midval) * 1.0
    y = ((x * 1.0) + midval) / numerator
    return y

def unnormsample(x):
    iinfo = np.iinfo(np.int16)
    y = int(x * iinfo.max)
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
    filt = firfilter(0, (midrate * wl) / 2.0, midrate, n)
    y = signal.lfilter(filt, 1, yout)

    print "Downsampling by",dnrate
    if dnrate == 1:
        yout = np.asarray(y, dtype=dtype)
    else:
        yout = np.zeros(newcount, dtype=dtype)
        for i in range(0, newcount-1):
            yout[i] = y[i * dnrate] * factor

    return yout

def firfilter(flow, fhigh, rate, n):
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

    return d
    
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

def mksine(flo, rate):
    freqgcd = gcd(rate,flo)
    samples = rate / freqgcd
    stride  = flo  / freqgcd

    print "Generating sine wave using",samples,"samples and",stride,"stride at sampling rate of",rate
    lo = np.sin(np.linspace(np.pi/2.0, (3.0*np.pi)/2.0, samples, False))
    return (lo, stride)

def dofir(filt, lfilt, x, xin):
    for i in range(1, lfilt):
        x[i] = x[i-1]
    x[0] = xin

    y = 0
    for i in range(0, lfilt):
        y += x[i] * filt[lfilt - i]
    return (y, x)

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

flow  = 0
fhigh = 3800
fbw = fhigh - flow

n = 91
firfilter.fig = 1
newrate = 8000
intype = data.dtype

if rate < 40000:
    yout = resample(rate,40000,data,n,np.float64,factor)
    data = yout
    rate = 40000

dolo1 = (flow > 0 and flow + flow <= fhigh)
if dolo1:
    flo1 = 20000 - fbw - flow
    print "Translating to",flo1,"Hz to avoid aliasing"
    # Build the sine wave for LO1
    (lo1, stride1) = mksine(flo1, rate)
    sample1 = len(lo1)
    index1  = 0
    # Build the IF filter
    if1 = firfilter(flo1 + flow, flo1 + fhigh, rate, n)
    lif1 = size(if1) - 1
    xif1 = np.zeros(lif1+1, np.float64)

dolo2 = (flow > 0)
if dolo2:
    flo2 = 20000 - fbw
    # Build the sine wave for LO2
    (lo2, stride2) = mksine(flo2, rate)
    sample2 = len(lo2)
    index2  = 0
    # Build the IF filter
    if2 = firfilter(0, fbw, rate, n)
    lif2 = size(if2) - 1
    xif2 = np.zeros(lif2+1, np.float64)

# Initial BPF/LPF for selecting the band
bpf = firfilter(flow, fhigh, rate, n)
lbpf = size(bpf) - 1
xbpf = np.zeros(lbpf+1, np.float64)

samplecount = len(data)
decimation = rate / newrate
outsamplecount = samplecount / decimation

# Antialiasing filter for decimation
aaf = firfilter(0, (rate * 1.0) / (decimation * 2.0), rate, n)
laaf = size(aaf) - 1
xaaf = np.zeros(laaf+1, np.float64)

# Final output array
y = zeros(outsamplecount, np.int16)

#show()

start = time.time()
oldelapsed = 0
for i in range(0, samplecount-1):
    xnorm = normsample(data[i], intype)
    (ybpf, xbpf) = dofir(bpf, lbpf, xbpf, xnorm)

    if dolo1:
        y1 = ybpf * lo1[index1]
        index1 = (index1 + stride1) % sample1
        (yif1, xif1) = dofir(if1, lif1, xif1, y1)
    else:
        yif1 = ybpf

    if dolo2:
        y2 = yif1 * lo2[index2]
        index2 = (index2 + stride2) % sample2
        (yif2, xif2) = dofir(if2, lif2, xif2, y2)
    else:
        yif2 = yif1

    (yout, xaaf) = dofir(aaf, laaf, xaaf, yif2)

    if i % decimation == 0:
        y[i / decimation] = unnormsample(yout)

    elapsed = int(time.time() - start)
    if elapsed != oldelapsed and elapsed % 10 == 1:
        oldelapsed = elapsed
        sps = i / elapsed
        print i,"samples in",elapsed,"s =",sps,"samples/s"


print "Writing"
write(outfile,newrate,y)

sys.exit(0)

