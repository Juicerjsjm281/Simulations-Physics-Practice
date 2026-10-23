import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#Still learning


f = lambda x: np.sin(np.pi * x) **2 * np.cos(np.pi*x)
n = 100                                    
x0, xf = -4, 4
x = np.linspace(x0,xf,2*n+1)
h = x[2::2] - x[:-2:2]                  #In case the difference between two nodes is not uniform.

#Mass-Matrix Global
M = sp.lil_matrix((2*n+1,2*n+1))        #lil_matrix is better for creating large matrices with almost zeros values.
b = np.zeros(2*n+1)

#Gaussian Quadrature for two points (Calculated by hand)
gp, gw = np.polynomial.legendre.leggauss(3)

for e in range(n):
    Me = (h[e]/30) * np.array([[4,2,-1],[2,16,2],[-1,2,4]])                       #Local matrix of the e element
    idx = [2*e,2*e+1,2*e+2]                                           #The element we are solving is between the nodes e and e+1
    M[np.ix_(idx,idx)] += Me
    for xi, w in zip(gp, gw):
        xq = 0.5*(x[2*e] + x[2*e+2] + h[e]*xi)
        phi_xi = np.array([xi*(xi-1)*0.5, 1- xi**2 , xi*(xi+1)*0.5])
        b[idx] += 0.5*h[e]*w*f(xq)*phi_xi


U = spla.spsolve(M.tocsr(),b)                               


error_L2 = 0.0
for e in range(n):
    idx = [2*e, 2*e+1, 2*e+2]
    for xi, w in zip(gp, gw):
        xq = 0.5*(x[2*e] + x[2*e + 2] + h[e]*xi)   # extremos del elemento
        uh = U[idx] @ np.array([xi*(xi-1)*0.5, 1- xi**2 , xi*(xi+1)*0.5])                  # u_h en xq
        error_L2 += 0.5*h[e]*w*(f(xq) - uh)**2

error_L2 = np.sqrt(error_L2)
print("Error L2 =", error_L2)

#Projection Quality
fig, ax = plt.subplots(facecolor = "#0B0F19")


ax.plot(x, f(x), label="$f(x)$", color = '#39C5BB')
ax.plot(x, U, "-", label=r"$u_h$", color = '#E60033')
ax.set_title(rf"$L^2: P2-1D$", color= '#FFFFFF')
ax.grid(color = '#374151')
ax.set_facecolor('#111827')
ax.tick_params(colors='#9CA3AF', labelsize=10)
ax.legend(loc='upper right')
for spine in ax.spines.values():
    spine.set_edgecolor('#374151')
plt.show()