#!/usr/bin/python3

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from time import (clock_gettime, CLOCK_MONOTONIC_RAW)

sz=3
vsz=10
c=np.zeros(vsz);  c[0]=0.5
data = np.random.rand(sz, vsz) / 5.0 + c
c=np.zeros(vsz);  c[1]=0.5
data = np.concatenate((data, np.random.rand(sz, vsz) / 5.0 + c))
c=np.zeros(vsz);  c[2]=0.5
data = np.concatenate((data, np.random.rand(sz, vsz) / 5.0 + c))

som = MiniSom(16, 16, vsz, sigma=6, learning_rate=1.0, neigh_threshold=0.2)
t1 = clock_gettime(CLOCK_MONOTONIC_RAW)
som.train_random(data, 20)
t2 = clock_gettime(CLOCK_MONOTONIC_RAW)
print("SOM train_random() time: " + str(t2 - t1) + " seconds")

for i in range(len(data)):
    v = data[i].copy()
#    print("i=" + str(i) + ", v=" + str(v))
    print(som.winner(v))
    v[int(np.random.rand()*len(v))] = None
#    print("i=" + str(i) + ", v=" + str(v))
    print(som.winner(v))
    print()
