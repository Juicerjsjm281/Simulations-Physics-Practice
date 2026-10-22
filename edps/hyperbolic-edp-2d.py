import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation
from matplotlib.colors import LinearSegmentedColormap



#Parameters
L_x,L_y, alpha  = 1.0, 1.0, 5
n_x = 101
n_y = 101                               
dx = L_x/ (n_x - 1)
dy = L_y/(n_y-1)
dt = (1/alpha) * (1/np.sqrt((1/(dx**2) + 1/(dy**2))))  * (0.9)     #Stability as a consequence of bx**2 + by**2 = 1, 0.9 to avoid limit
nt = 300
bx = alpha *dt / dx
by = alpha * dt / dy
x = np.linspace(0,L_x,n_x)
y = np.linspace(L_y, 0 ,n_y)
X,Y = np.meshgrid(x,y)

#Initial Condition
u0 = np.exp(-500 * ((X-0.5)**2+(Y-0.5)**2))                                
g = np.zeros((n_y,n_x))                         #u0_dt


u_old, u = u0.copy(), u0.copy()

delta_ux_old = u_old[1:-1,2:] - 2*u_old[1:-1,1:-1] + u_old[1:-1,:-2]
delta_uy_old = u_old[2:,1:-1] - 2*u_old[1:-1,1:-1] + u_old[:-2,1:-1]

u[1:-1,1:-1] = u_old[1:-1,1:-1] + dt * g[1:-1,1:-1] + 0.5*((bx**2)*delta_ux_old + (by**2)*delta_uy_old)


# Dirichlet Boundary Condition
u[0,:] = 0
u[-1,:] = 0
u[:,0] = 0
u[:,-1] = 0



fig, ax = plt.subplots(facecolor = "#0B0F19")

cm_tm = LinearSegmentedColormap.from_list("t_m",['#39C5BB', '#FFD700','#E60033'])

wave = ax.imshow(u, cmap = cm_tm, origin="upper",extent=[0, L_x, 0, L_y], vmin=-0.2, vmax=0.2)
cbar = fig.colorbar(wave, ax=ax)
cbar.set_label("u", color = '#E5E7EB')
cbar.ax.yaxis.set_tick_params(color='#E5E7EB', labelcolor='#E5E7EB')
cbar.outline.set_edgecolor('#374151')
ax.grid(color = '#374151')
ax.set_title('Hyperbolic-EDP-Practice', color= '#FFFFFF')

ax.set_xlabel("x [m]", color='#E5E7EB')
ax.set_ylabel("y [m]", color='#E5E7EB')
ax.set_facecolor('#111827')
ax.tick_params(colors='#9CA3AF', labelsize=10)

for spine in ax.spines.values():
    spine.set_edgecolor('#374151')


delta_ux = u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]
delta_uy = u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]

def animate(i):
    global u, u_old
    u_new = np.zeros((n_y,n_x))
    delta_ux = u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]
    delta_uy = u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]
    u_new[1:-1,1:-1] = 2* u[1:-1,1:-1] - u_old[1:-1,1:-1] + (bx**2) * (delta_ux) + (by**2)*(delta_uy)
    u_old, u = u, u_new
    wave.set_data(u)
    return wave,

anim = FuncAnimation(fig, animate, frames = nt, interval = 20, blit = False)

plt.show()