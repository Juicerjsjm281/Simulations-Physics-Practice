import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#Still learning


f = lambda x: np.sin(np.pi * x)
n = 100
x0, xf = -4, 4
x = np.linspace(x0,xf,n+1)
h = np.diff(x)                      #In case the difference between two nodes is not uniform.

#Mass-Matrix Global
M = sp.lil_matrix((n+1,n+1))        #lil_matrix is better for creating large matrices with almost zeros values.
b = np.zeros(n+1)

#Gaussian Quadrature for two points (Calculated by hand)
gp = np.array([1,-1]) / np.sqrt(3)
gw = np.array([1,1])

for e in range(n):
    Me = (h[e]/6) * np.array([[2,1],[1,2]])                       #Local matrix of the e element
    idx = [e,e+1]                                           #The element we are solving is between the nodes e and e+1
    M[np.ix_(idx,idx)] += Me
    for xi, w in zip(gp, gw):
        xq = 0.5*(x[e] + x[e+1] + h[e]*xi)
        phi = np.array([(1-xi)*0.5, (1 + xi)*0.5])
        b[idx] += 0.5*h[e]*w*f(xq)*phi


U = spla.spsolve(M.tocsr(),b)                               #Finding our unkown values, the vector U.    M.tocsr() is better for solving systems

error_L2 = 0.0
for e in range(n):
    for xi, w in zip(gp, gw):
        xq = 0.5*(x[e] + x[e+1] + h[e]*xi)
        uh = U[e]*(1 - xi)*0.5 + U[e+1]*(1 + xi)*0.5
        error_L2 += 0.5*h[e]*w*(f(xq) - uh)**2


#Projection Quality
fig, ax = plt.subplots(facecolor = "#0B0F19")

ax.plot(x, f(x), label="$f(x)$", color = '#39C5BB')
ax.plot(x, U, "o-", label=r"$u_h$", color = '#E60033')
ax.set_title(rf"$||f-u_h||_{{L^2}}$: {np.sqrt(error_L2)}", color= '#FFFFFF')
ax.grid(color = '#374151')
ax.set_facecolor('#111827')
ax.tick_params(colors='#9CA3AF', labelsize=10)
ax.legend()
for spine in ax.spines.values():
    spine.set_edgecolor('#374151')
plt.show()
