import csv
import matplotlib.pyplot as plt
import math
import matlab.engine

from activitydetection import VoiceActivityDetector

oldwindow = []
firsttime = True
eng = matlab.engine.start_matlab()
movrms = None

def sliding_rms(x, N):
    global firsttime, movrms
    if firsttime:
        movrms = eng.initrms(N)
        firsttime = False
    input = matlab.double(x)
    ret = eng.movrms(input,movrms)
    retarray = [element for element in ret[0]._data]
    return retarray


def find_signals(input):
    data = input
    n = len(data)

    # fid.close()

    FrameLength = 7
    Fs = 100
    # movrmsWin = dsp.MovingRMS(FrameLength);

    ''' Create the input signal.Thesignal is a noisy staircase with a frame % length of 20.
    The threshold value is 200. % Compute the energy of the signal by squaring the RMS value and
    multiplying the result with the window length.Compare the signal energy with the threshold value.
    Detect the event, and when the signal energy crosses the threshold, mark it as 1.'''
    count = 1
    index = 1
    threshold = 0
    start = 0
    endnumber = FrameLength
    frames = (n / FrameLength)
    testter = False
    for i in range(0, frames):
        x = data[start:endnumber]
        y1 = sliding_rms(x, FrameLength)
        y1ener = (y1[-1] ** 2) * FrameLength
        threshold += y1ener
        start += FrameLength
        endnumber += FrameLength

    threshold /= frames
    threshold *= 1.05
    start = 0
    endnumber = FrameLength
    startaddresses = []
    energies = []
    events = []
    for i in range(0, frames):
        x = data[start:endnumber]
        y1 = sliding_rms(x, FrameLength)
        y1ener = (y1[-1] ** 2) * FrameLength
        startaddresses.append(start)
        energies.append(y1ener)
        event = (y1ener > threshold)
        events.append(event)
        start += FrameLength
        endnumber += FrameLength

    founddata = []
    foundsignalsize = 6000
    i = 0
    while True:
        if i >= frames:
            break

        if events[i]:
            if i == 1:
                foundsignal = []
                for j in range(0, foundsignalsize):
                    foundsignal.append(data[j])
                founddata += foundsignal
                i += foundsignalsize / FrameLength
            else:
                foundsignal = []
                start = 0
                end = 0
                if startaddresses[i] - 2000 < 0:
                    test = (startaddresses[i] + foundsignalsize - 3000)+(abs(startaddresses[i] - 2000))
                    for j in range(0, (startaddresses[i] + foundsignalsize - 3000)+(abs(startaddresses[i] - 2000))):
                        foundsignal.append(data[j])
                elif startaddresses[i] + foundsignalsize - 3000 >= len(data):
                    test = (startaddresses[i] - 2000 - (startaddresses[i] + foundsignalsize - 3000 -len(data)))
                    for j in range((startaddresses[i] - 2000 - (startaddresses[i] + foundsignalsize - 3000 -len(data))), len(data)):
                        foundsignal.append(data[j])
                else:
                    for j in range((startaddresses[i] - 2000), (startaddresses[i] + foundsignalsize - 3000)):
                            foundsignal.append(data[j])

                founddata.append(foundsignal)
                i += foundsignalsize / FrameLength

        i += 1
        i = int(i)

    foundsigs = len(founddata)
    return founddata

def find_signals_improved(file):
    fid = open(file)
    data = []
    for line in fid.readlines():
        try:
            tline = float(line)
            data.append(tline)
        except:
            pass
    activity_detector = VoiceActivityDetector(data,)
    res = activity_detector.detect_speech()
    # activity_detector.plot_detected_speech_regions()
    conv = activity_detector.convert_windows_to_labels(res)
    # activity_detector.plot_detected_speech_regions()
    started = False
    signals = []
    signalsize = 6000
    for period in conv:
        speech_len = period["speech_end"] - period["speech_begin"]
        amount_of_mixed_speech = math.floor(speech_len / (signalsize * 1.0))
        amount_of_mixed_speech = int(amount_of_mixed_speech)
        if amount_of_mixed_speech > 0:
            startindex = period["speech_begin"]
            startindex = int(startindex)
            for speechsignal in range(0, amount_of_mixed_speech):
                if startindex + signalsize > len(data):
                    signals.append(data[startindex - ( (startindex + signalsize)- len(data)):len(data)])
                    break
                else:
                    signals.append(data[startindex:startindex + signalsize])
                    startindex += signalsize
        else:
            startindex = period["speech_begin"]
            startindex = int(startindex)
            if startindex + signalsize > len(data):
                signals.append(data[startindex - ( (startindex + signalsize)- len(data)):len(data)])
                break
            else:
                signals.append(data[startindex:startindex + signalsize])
        if len(signals[-1]) != signalsize:
            pass
    blanks = []
    if len(signals) > 2:
        for index, speechperiod in enumerate(conv):
            if index == len(conv)-2:
                #last block
                break
            else:
                speech_len = conv[index+1]["speech_begin"] - speechperiod["speech_end"]
                if speech_len>signalsize:
                    amount_of_mixed_speech = math.floor(speech_len / (signalsize * 1.0))
                    amount_of_mixed_speech = int(amount_of_mixed_speech)
                    if amount_of_mixed_speech > 0:
                        startindex = speechperiod["speech_end"]
                        startindex = int(startindex)
                        for speechsignal in range(0, amount_of_mixed_speech):
                            if startindex + signalsize > len(data):
                                blanks.append(data[startindex - (len(data) - startindex + signalsize):len(data) - 1])
                                break
                            else:
                                blanks.append(data[startindex:startindex + signalsize])
                                startindex += signalsize
                    else:
                        startindex = speechperiod["speech_end"]
                        startindex = int(startindex)
                        if startindex + signalsize > len(data):
                            blanks.append(data[startindex - (len(data) - startindex + signalsize):len(data) - 1])
                            break
                        else:
                            blanks.append(data[startindex:startindex + signalsize])
    for signal in signals:
        if len(signal) != signalsize:
            pass
        # plt.figure()
        # plt.plot(signal)
        # plt.show()
    return signals, blanks
if __name__ == '__main__':
    print find_signals_improved('testdata/newvoweldata/aoutput.txt')



