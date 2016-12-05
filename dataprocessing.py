import csv

import matlab.engine

oldwindow = []
firsttime = True
eng = matlab.engine.start_matlab()
movrms =None

def sliding_rms(x, N):
    global  firsttime, movrms
    if firsttime:
        movrms = eng.initrms(N)
        firsttime = False
    input = matlab.double(x)
    ret = eng.movrms(input,movrms)
    retarray = [element for element in ret[0]._data]
    return retarray

if __name__ == '__main__':
    fid = open('rawdata/output.txt')
    data = []
    for line in fid.readlines():
        try:
            tline = float(line)
            data.append(tline)
        except:
            pass

    n = len(data)

    fid.close()

    FrameLength = 7
    Fs = 100
    #movrmsWin = dsp.MovingRMS(FrameLength);

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
    for i in range(0,frames):
        x = data[start:endnumber]
        y1 = sliding_rms(x,FrameLength)
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
    for i in range(0,frames):
        x = data[start:endnumber]
        y1 = sliding_rms(x,FrameLength)
        y1ener = (y1[-1] ** 2) * FrameLength
        startaddresses .append(start)
        energies.append(y1ener)
        event = (y1ener > threshold)
        events .append(event)
        start += FrameLength
        endnumber += FrameLength


    founddata = []
    foundsignalsize = 6000
    i = 0
    while True:
        if i >=frames:
            break

        if events[i]:
            if i == 1:
                foundsignal = []
                for j in range(0,foundsignalsize):
                    foundsignal.append(data[j])
                founddata += foundsignal
                i += foundsignalsize / FrameLength
            else:
                foundsignal = []
                for j in range((startaddresses[i]-2000),(startaddresses[i] + foundsignalsize - 3000)):
                    foundsignal.append(data[j])

                founddata .append( foundsignal)
                i += foundsignalsize / FrameLength

        i += 1
        i = int(i)

    foundsigs = len(founddata)
    for i in range(0,foundsigs):
        # some = 1:(m);
        # thing = founddata(:, i);
        # a = plot(some, thing);
        b = 'data'+ str(i)+'.png'
        c = 'outputs/data'+ str(i)+'.csv'
        # saveas(a, b);
        with open(c, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(founddata[i])







