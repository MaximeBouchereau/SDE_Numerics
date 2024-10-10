from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

q=0.3
eps=1e-6
t0=1
t=10

# Etude de la décroissance de la norme infinie de la solution quand eps tend vers 0

EPS = [2**(-i) for i in range(1,22)] # Ensemble des valeurs de epsilon qui seront testées

NOR = [] # Collection des normes infinies des solutions en fonction de eps

for eps in EPS:
    
    def model_eps(t,gt):
        dg=-eps**(2*q-1)*gt/(t**(-2*q))-q*gt/t+q/t # Tu avais oublié le "q" devant le -gt/t
        #dg = -gt*(t**(-2*q)+q*t^(-1))+q*t**(-1)
        return [dg]
    
    def model(t,gt):
        dg=-gt/(t**(-2*q))-gt/t+q/t
    
        return [dg]

    sol = solve_ivp(model_eps, [t0, t], [0], method='RK45', max_step=0.01)
    
    NOR.append(np.linalg.norm(sol.y[0],np.infty))

    #plt.plot(sol.t,sol.y[0])
    #plt.show()

plt.figure()
plt.scatter(EPS,NOR,color="green",label="Erreur (norme infinie)")
plt.xlabel("$\epsilon$")
plt.ylabel("$||y_{\epsilon}||_{L^{\infty}}$")
plt.legend()
plt.title("Evolution du maximum de la solution en fonction de epsilon")
plt.grid()
plt.loglog()