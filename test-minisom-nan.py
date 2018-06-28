#!/usr/bin/python3

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from time import (clock_gettime, CLOCK_MONOTONIC_RAW)
import shutil

columns = shutil.get_terminal_size().columns
np.set_printoptions(precision = 2, linewidth = columns)

sz=3
vsz=10
c=np.zeros(vsz);  c[0]=0.5
data = np.random.rand(sz, vsz) / 5.0 + c
c=np.zeros(vsz);  c[1]=0.5
data = np.concatenate((data, np.random.rand(sz, vsz) / 5.0 + c))
c=np.zeros(vsz);  c[2]=0.5
data = np.concatenate((data, np.random.rand(sz, vsz) / 5.0 + c))

som = MiniSom(16, 16, vsz, sigma=4, learning_rate=1.0, neigh_threshold=0.2)
t1 = clock_gettime(CLOCK_MONOTONIC_RAW)
som.train_random(data, 20)
t2 = clock_gettime(CLOCK_MONOTONIC_RAW)
print("SOM train_random() time: " + str(t2 - t1) + " seconds")
print("SOM distance map:")
print(str(som.distance_map()))

for i in range(len(data)):
    v = data[i].copy()
    print("Winner for original vector  " + str(v) + ": " + str(som.winner(v)))
    v[int(np.random.rand()*len(v))] = None
    print("Winner for vector with 1nan " + str(v) + ": " + str(som.winner(v)))
    v[int(np.random.rand()*len(v))] = None
    print("Winner for vector with 2nan " + str(v) + ": " + str(som.winner(v)))
    print()

v = data[0].copy()
v[:] = None
print("Winner for all-nan vector " + str(v) + ": " + str(som.winner(np.nan)))
