import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#### This practice uses the value of delta_x = delta_y


#Mesh Configuration
n_side = 70                              #Squares in the mesh n_side**2
epsilon_s = 1e-5                         #Tolerance in %
alpha = 1.88                             #Value between 1 and 2
l = 5                                    #Size of one of the sides
delta_x = l / (n_side-1)
k_t = 1

u = np.zeros([n_side,n_side])

#Fixed Boundary Conditions (Dirichlet)
#u[:,0] = 0
#u[:,-1] = 0
#u[0,:] = 0
u[-1,:] = 1


#f(x,y)
q = 1
f = np.zeros([n_side, n_side])
f[n_side//2,n_side//2] = -q / (delta_x**2)


for n in range(1000):
    epsilon_a = 0.0
    for i in range(1,n_side-1):
        for j in range(1,n_side-1):
            old = u[i,j]
            new  = 0.25*(u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - (delta_x**2) * f[i,j])
            u[i,j]  = alpha * new + (1 - alpha) * old
            if u[i,j] != 0:
                epsilon_a = np.max([epsilon_a, np.abs((u[i,j] - old)/u[i,j])*100])
    if n % 50 == 0:
        print(f"Iteración {n}, error = {epsilon_a:.2e} %")

    if epsilon_a < epsilon_s:
        print(f"They converged at the {n+1} iteration.")
        break
else:
    print("They do not converge within the maximum number of iterations.")


#Gradient
Gx,Gy = np.gradient(u, delta_x)

x = np.linspace(0,l, n_side)
y = np.linspace(0,l, n_side)
X,Y = np.meshgrid(x,y, indexing='ij')

skip = 3



#Color map
cm_tm = LinearSegmentedColormap.from_list("t_m",['#39C5BB','#E60033'])

fig, ax = plt.subplots(facecolor = "#0B0F19")

im = ax.imshow(u.T, cmap = cm_tm, origin = 'lower', extent = [0,l,0,l])

ax.quiver(X[::skip,::skip],Y[::skip,::skip], - k_t * Gx[::skip,::skip], -k_t * Gy[::skip,::skip])

ax.grid(color = '#374151', alpha = 0.3)
ax.set_title('Poisson-Equation-Practice', color= '#FFFFFF')

ax.set_xlabel("x [m]", color='#E5E7EB')
ax.set_ylabel("y [m]", color='#E5E7EB')
ax.tick_params(colors='#9CA3AF', labelsize=10)

for spine in ax.spines.values():
    spine.set_edgecolor('#374151')

cbar = fig.colorbar(im, ax = ax)
cbar.set_label("T [°C]", color='#E5E7EB')
cbar.ax.yaxis.set_tick_params(color='#E5E7EB', labelcolor='#E5E7EB')

plt.show()