import serial
import icatest
import dataprocessing
from scipy.interpolate import interp1d
import time

serialport = serial.Serial("/dev/ttyACM0", 9600, timeout=0.5)
readbuffer = ""
datacollectionarray = []
maxdatasize = 15000
sampletake_secoonds = 20
icatest.initialize_classifier()
m = interp1d([0,1023],[0,720])
while True:
    start = time.time()
    while True:
        command = serialport.read()
        if command == " ":
            continue
        if command == "\n" or command == "\r":
            # if maxdatasize == len(datacollectionarray):
            #     print "Flushing collected data", datacollectionarray
            #     # preprocessed = icatest.preprocess_data_array(datacollectionarray)
            #     extracted = dataprocessing.find_signals(datacollectionarray)
            #     result = icatest.classify(extracted)
            #     print result
            #     datacollectionarray = []
            stripped = readbuffer.strip()
            if stripped != '':
                mappedvalue = int(stripped)/4
                try:
                    mappedvalue = float(m(mappedvalue))
                except:
                    continue
                datacollectionarray.append(mappedvalue)
                print mappedvalue

            readbuffer = ""
            continue
        readbuffer+=command

        if time.time() - start > sampletake_secoonds:
            print "Flushing collected data", len(datacollectionarray)
            extracted = dataprocessing.find_signals(datacollectionarray)
            result = icatest.classify(extracted)
            print result
            datacollectionarray = []
            break


