import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.integrate import odeint
import numpy as np
import os



#parameters
g = 9.82
l = 0.281

#x and y plot limits
xa = -2*np.pi
xb = 2*np.pi

ya = -2*np.pi
yb = 2*np.pi

#number of points both up and across
NumPoints = 9

xPoints = np.linspace(xa,xb,NumPoints)
yPoints = np.linspace(ya,yb,NumPoints)

#To solve specific
resolution = 1000
xInPoints = np.linspace(xa,xb,resolution)

portraitName = "Inverted Pendulum"

""" l = 1
g = 9.82

def model(x: np.ndarray,t)->np.ndarray:
    return np.array([x[1],-(g/l) * np.sin(x[0])])

def backWardsModel(x:np.ndarray,t) -> np.ndarray:
    return -1*model(x,t) """

dxdt = np.array([
    [0,1],
    [g/l,0]
])


""" A = np.array([[-1,0],[0,-2]])
T = np.array([
    [np.cos(np.pi/4),-np.sin(np.pi/4)],
    [np.sin(np.pi/4),np.cos(np.pi/4)]
])
B = T@A@T.transpose()

dxdt =A """

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
    plt.plot(solution[:, 0], solution[:, 1], linewidth=1, color='black', alpha=0.6)
    plt.plot(backSolution[:, 0], backSolution[:, 1], linewidth=1, color='black', alpha=0.6)

    Derivative = model(initialState,0)
    plt.quiver(initialState[0], initialState[1], Derivative[0], Derivative[1])
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
        plotSolution(np.array([xPoints[i],yPoints[j]]))        




#plt.xlabel('Position (x)')
#plt.ylabel('Velocity (dx/dt)')
plt.title(portraitName)
plt.xlabel("θ")
plt.ylabel("dθ/dt")
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
print(np.linalg.eigvals(dxdt))