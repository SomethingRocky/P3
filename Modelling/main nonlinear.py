import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are

 #variable der bruges 
mc = 6.28  #massen af carten
mp = 0.25 # massen af pendulet
l = 0.281 # længden af pendulet
g = 9.82 # tyngdeacceleration
alpha = 0.5e-3 #Viskosfriktion koefficient for pendulet
F_c = 3.2 #Coulomb friktion cart
F_p = 4.1e-3 #Coulomb friktion pendul
r = 0.028 #Pulley radius
torque = 93.4e-3 #Motor konstant

timeInterval = [0,2]
t = np.linspace(timeInterval[0], timeInterval[1], 1001)



x_hat = 0.385 # Limit på distance
u_hat = 263.19 # Max strøm



Q = np.zeros((4,4))
Q[0,0] = 1/x_hat**2

R = np.array([[(1/(u_hat**2))]])
Rinv = np.array([[(u_hat**2)]])


n = 10
J_A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, (mp*g)/mc, -(n*F_c)/mc, -(n*F_p + alpha)/(l*mc)],
    [0, (g/l)*(1+(mp/mc)), -(n*F_c)/(l*mc), -((F_p + alpha)*(mc + mp))/(mc*mp*l**2)]
])


J_B = np.array([
    [0], 
    [0], 
    [1/mc], 
    [1/(l*mc)]
    ])



def stateToCurrent(solution:np.ndarray, K:np.ndarray)->np.ndarray:
    return np.array([(r/torque)* np.linalg.norm(-K@i) for i in solution[:]])

def plotStyling(x:str,y:str,title:str, folder) -> None:
    # ensure output folder exists
    os.makedirs(folder, exist_ok=True)

    plt.grid()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.title(title)
    filename = os.path.join(folder, title + ".png")
    plt.savefig(filename)
    plt.clf()

def plotSave(t:np.ndarray ,solutions:list, K:np.ndarray, folder:str="NonlinearPlots")-> None:
    for solution in solutions:
        plt.plot(t,solution[:,0], label=f"{round(solution[0,1],2)} rad")
    plt.hlines([-x_hat,x_hat],timeInterval[0],timeInterval[1],colors='k',linestyles='--')
    plotStyling("t (s)", "x (m)", "Cart displacement",folder)
    
    for solution in solutions:
        plt.plot(t,solution[:,2], label=f"{round(solution[0,1],2)} rad")
    plotStyling("t (s)", "dx/dt (m/s)", "Cart velocity",folder)
    
    for solution in solutions:
        plt.plot(t,solution[:,1], label=f"{round(solution[0,1],2)} rad")
    plotStyling("t (s)", "θ (rad)", "Pendulum angle",folder)
    
    for solution in solutions:
        plt.plot(t,solution[:,3], label=f"{round(solution[0,1],2)} rad")
    plotStyling("t (s)", "dθ/dt (rad/s)", "Pendulum angular velocity",folder)
    
    for solution in solutions:
        plt.plot(t, stateToCurrent(solution,K), label=f"{round(solution[0,1],2)} rad")
    plt.hlines([78.9],timeInterval[0],timeInterval[1],colors='k',linestyles='--')
    plotStyling("t (s)", "Current (A)", "Motor Current",folder)
    
    
    

def model(s:np.ndarray, t, K:np.ndarray) -> np.ndarray:
    u = float(-K@s)
    gammaC = - (np.tanh(n*s[2])*F_c)
    gammaP = - (np.tanh(n*s[3])*F_p)
    xi = mc + mp * (1-(np.cos(s[1]))**2)
    
    s1 = s[2]
    s2 = s[3]
    
    s3num1 = gammaC+((gammaP+alpha * s[3])/l)*np.cos(s[1]) 
    s3num2 = mp*g*np.cos(s[1])*np.sin(s[1])-mp*l*(s[3]**2)*np.sin(s[1])
    s3num = s3num1 + s3num2
    s3 = (s3num/xi) + (1/xi)*u
    
    s4part1 =(gammaP-alpha*s[3])/(mp*(l**2))
    s4part2 = (s3num/(l*xi))*np.cos(s[1])
    s4part3 = (g/l)*np.sin(s[1])
    s4part4 = (np.cos(s[1])/(l*xi))*u
    s4 = s4part1 + s4part2 + s4part3 + s4part4
    

    
    return np.array([s1,s2,s3,s4])
    
    
     

def main(initialStates:list)->None:
    solutions = []
    for initialState in initialStates:
        # Beregner k-matricen
        P = solve_continuous_are(J_A, J_B, Q, R)

        K = Rinv @ J_B.transpose() @ P
        solutions.append(odeint(model, initialState, t, (K,)))
        plotSave(t,solutions, K)

        print(f"With initialstate: \n{initialState}\n")
        print(f"Max current draw: {max(stateToCurrent(solutions[-1],K))}")
        print(f"Eigenværdierne for A:\n{np.linalg.eigvals(J_A)}")

        C = np.hstack([np.linalg.matrix_power(J_A, i) @ J_B for i in range(4)])

        print(f"Det C = {np.linalg.det(C)} ")
        print(f"C matrix:\n{C}")

        print(f"K matrix:\n {round(K[0,0],2)} , {round(K[0,1],2)} , {round(K[0,2],2)} , {round(K[0,3],2)} \n\n")
        
if __name__ == "__main__":
    initialStates = [np.array([0,0.1 * i, 0,0]) for i in range(1,5)]
    main(initialStates)
