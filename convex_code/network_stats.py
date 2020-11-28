#!/usr/bin/env python3

from subprocess import Popen, DEVNULL
import os
import pandas
import time
import matplotlib.pyplot as plt

DSTAT_FNAME = "dstat.csv"

if os.path.exists(DSTAT_FNAME):
    os.remove(DSTAT_FNAME) #dstat appends by default

dstat = Popen(["dstat", "--output="+DSTAT_FNAME], stdout=DEVNULL)
print("Dstat initialized.")

time.sleep(20) # run for 20 seconds

dstat.kill()

dstat_file = pandas.read_csv(DSTAT_FNAME, header=5)
print(len(dstat_file))
plt.plot(range(0,len(dstat_file)), dstat_file['recv'] / (1024*1024), label="Network receive")
plt.xlabel("Time in s")
plt.ylabel("Data recvd over network in MB")
# plt.show()

plt.plot(range(0,len(dstat_file)), dstat_file['send'] / (1024*1024), label="Network send")
plt.xlabel("Time in s")
plt.ylabel("Data recvd over network in MB")
plt.show()