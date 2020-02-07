import pickle

filename = 'F_bridge_results.pickle'

with open(filename, 'rb') as f:
    x = pickle.load(f)

import matplotlib.pyplot as plt

for rep in range(5):
    plt.plot(x["Loss"][0][rep])
    plt.plot(x["Loss"][1][rep])
    plt.plot(x["Loss"][2][rep])
    plt.show()