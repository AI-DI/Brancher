import pickle
import numpy as np

filename = 'Full_brownian_results.pickle'

with open(filename, 'rb') as f:
    x = pickle.load(f)

MSE = {"ASVI": (np.mean([np.sqrt(error) for error in x["PE"]["MSE"]]), np.std([np.sqrt(error) for error in x["PE"]["MSE"]])/np.sqrt(len(x["PE"]["MSE"]))),
"ADVI (MF)": (np.mean([np.sqrt(error) for error in x["ADVI (MF)"]["MSE"]]), np.std([np.sqrt(error) for error in x["ADVI (MF)"]["MSE"]])/np.sqrt(len(x["ADVI (MF)"]["MSE"]))),
"ADVI (MN)": (np.mean([np.sqrt(error) for error in x["ADVI (MN)"]["MSE"]]), np.std([np.sqrt(error) for error in x["ADVI (MN)"]["MSE"]])/np.sqrt(len(x["ADVI (MN)"]["MSE"]))),
"NN": (np.mean([np.sqrt(error) for error in x["NN"]["MSE"]]), np.std([np.sqrt(error) for error in x["NN"]["MSE"]])/np.sqrt(len(x["NN"]["MSE"])))}

for key, val in MSE.items():
    print(key + ": {} +- {}".format(val[0], val[1]))