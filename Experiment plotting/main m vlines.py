import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import io
import os 

output_file = "PlotsExp"
input_file = "Data"
SampleTime = 0.00667

#User input
filenames = ["Qscale20",
             "Qscale20doubletest",
             "Standard"]

#filenames = ["Qscale20doubletest"]


def plotSave(data:np.ndarray,
             t:np.ndarray, 
             filename:str,
             xlabel:str, 
             ylabel:str, 
             title:str,):

    plt.plot(t,data)
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_file, filename, filename + "_" + title))
    plt.clf()


def main(filename:str) -> None:
    csv_path = os.path.join(input_file, filename + ".csv")
    #DO NOT TOUCH THE WIZARDRY
    with open(csv_path, 'r', encoding='utf-8') as file:
        multi = ''.join(ln for ln in file if ',' in ln)
    data = pd.read_csv(io.StringIO(multi), header=None) 

    # coerce every cell to numeric; non-convertible -> NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    #Extracting data as floats
    x = data.iloc[:,0].to_numpy(dtype=float)
    x = x - 0.385
    theta = data.iloc[:,1].to_numpy(dtype=float)
    dx = data.iloc[:,2].to_numpy(dtype=float)
    dtheta = data.iloc[:,3].to_numpy(dtype=float)
    current = data.iloc[:,4].to_numpy(dtype=float)
    true_theta = data.iloc[:,5].to_numpy(dtype=float) + np.pi

    t = np.array([SampleTime * i for i in range(len(x))])
    
    if filename != "Standard":
        #Finder tid, hvor der stabiliseres
        thetadiff = theta[0:-35]-theta[35:]
        stabilizeIndex = len(theta[0:-35][abs(thetadiff)>=0.00001])
        tStabil = t[stabilizeIndex]
    
        print(tStabil)
    



    #plotting
    os.makedirs(os.path.join(output_file, filename), exist_ok=True)
    plt.hlines([-0.385,0.385],t[0],t[-1],colors='k',linestyles='--')
    if filename != "Standard":
        plt.axvline(tStabil,color='m',linestyle='--')
    plotSave(x, t, filename, "Time [s]","Cart displacement [m]", "Cart displacement")
    if filename != "Standard":
        plt.axvline(tStabil,color='m',linestyle='--')
    plotSave(true_theta, t, filename, "Time [s]", "Pendulum angle [rad]", "Pendulum angle")
    plotSave(dx, t, filename, "Time [s]", "Cart velocity [m/s]", "Cart velocity")
    plotSave(dtheta, t, filename, "Time [s]", "Pendulum angular velocity [rad/s]", "Pendulum angular velocity")
    plt.hlines([78.9],t[0],t[-1],colors='k',linestyles='--')
    plotSave(current, t, filename, "Time [s]", "Current [A]", "motor Current")

if __name__ == "__main__":
    for filename in filenames:
        main(filename)
