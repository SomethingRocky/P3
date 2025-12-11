import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import os

#Parameters for the model phase portrait
mc = 6.28  #massen af carten
mp = 0.25 # massen af pendulet
l = 0.281 # længden af pendulet
g = 9.82 # tyngdeacceleration
alpha = 0.5e-3 #Viskosfriktion koefficient for pendulet
F_c = 3.2 #Coulomb friktion cart
F_p = 4.1e-3 #Coulomb friktion pendul
r = 0.028 #Pulley radius
torque = 93.4e-3 #Motor konstant

#x and y plot limits
xa = -5
xb = 5

ya = -5
yb = 5

#number of points both up and across
NumPoints = 7

xPoints = np.linspace(xa,xb,NumPoints)
yPoints = np.linspace(ya,yb,NumPoints)

#To solve specific
resolution = 1000
xInPoints = np.linspace(xa,xb,resolution)

portraitName = "Model phase portrait"

""" l = 1
g = 9.82

def model(x: np.ndarray,t)->np.ndarray:
    return np.array([x[1],-(g/l) * np.sin(x[0])])

def backWardsModel(x:np.ndarray,t) -> np.ndarray:
    return -1*model(x,t) """

n = 1
J_A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (mp*g)/mc, -(n*F_c)/mc, -(n*F_p + alpha)/(l*mc)],
    [0, (g/l)*(1+(mp/mc)), -(n*F_c)/(l*mc), -((F_p + alpha)*(mc + mp))/(mc*mp*l**2)]
])

dxdt = J_A


def model(x:np.ndarray, t) -> np.ndarray:
    return dxdt @ x

def backWardsModel(x:np.ndarray,t) -> np.ndarray:
    return (-1 * dxdt) @ x

plt.xlim(xa,xb)
plt.ylim(ya,yb)


def plotSolution(initialState:np.ndarray):
    solution = odeint(model, initialState, xInPoints)
    backSolution = odeint(backWardsModel, initialState, xInPoints)
    fullSolution = np.concatenate((backSolution,solution), axis=0)
    plt.plot(solution[:, 1], solution[:, 3], linewidth=1, color='black', alpha=0.6)
    plt.plot(backSolution[:, 1], backSolution[:, 3], linewidth=1, color='black', alpha=0.6)

    Derivative = model(initialState,0)
    plt.quiver(initialState[1], initialState[3], Derivative[1], Derivative[3])
    """
    #add arrow at midpoint
    SolutionWithinIndices = (fullSolution[:,0] >= xa) & (fullSolution[:,0] <= xb) & (fullSolution[:,1] >= ya) & (fullSolution[:,1] <= yb)
    
    # Fix: Get indices of True values, then find middle
    true_indices = np.where(SolutionWithinIndices)[0]
    if len(true_indices) > 0:
        middle_index = true_indices[len(true_indices) // 2]
        middlePoint = fullSolution[middle_index]
        Der = dxdt @ middlePoint
        plt.quiver(middlePoint[0], middlePoint[1], Der[0], Der[1], alpha=0.6)
    """
    
    

# Plot phase portrait (direction field)
for i in range(NumPoints):
    for j in range(NumPoints):
        plotSolution(np.array([0, xPoints[i], 0, yPoints[j]]))        




#plt.xlabel('Position (x)')
#plt.ylabel('Velocity (dx/dt)')
plt.title(portraitName)
plt.legend()
plt.grid(True, alpha=0.5)

# Create folder and save plot
output_folder = 'phase_portraits'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plt.savefig(os.path.join(output_folder, portraitName + '.jpg'), 
            dpi=150, bbox_inches='tight', 
            pil_kwargs={'quality': 70, 'optimize': True}) 

plt.show()