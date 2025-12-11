import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

#x and y plot limits
xa = -10
xb = 10

ya = -10
yb = 10

#number of points both up and across
NumPoints = 21

xPoints = np.linspace(xa,xb,NumPoints)
yPoints = np.linspace(ya,yb,NumPoints)

#To solve specific
initialState = [3.2,-3]
xInPoints = np.linspace(xa,xb,1000)

dxdt = np.array([[1,3],[1,-1]])

def model(x:np.ndarray, t) ->np.ndarray:
    return dxdt @ x

Derivatives = np.zeros((NumPoints, NumPoints, len(dxdt)))

# Solve the ODE system
solution = odeint(model, initialState, xInPoints)

plt.xlim(xa,xb)
plt.ylim(ya,yb)

# Plot phase portrait (direction field)
for i in range(NumPoints):
    for j in range(NumPoints):
        Derivatives[i][j] = dxdt @ [xPoints[i], yPoints[j]]
        xDer = Derivatives[i][j][0]
        yDer = Derivatives[i][j][1]
        plt.quiver(xPoints[i], yPoints[j], xDer, yDer, alpha=0.6)

# Plot the solution trajectory correctly
plt.plot(solution[:, 0], solution[:, 1], 'r-', linewidth=2, label='Trajectory')


plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()