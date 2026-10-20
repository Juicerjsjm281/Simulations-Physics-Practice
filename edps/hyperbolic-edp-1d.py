import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation



#Parameters
L, alpha  = 2.0, 100
beta = 0.5                              #Courant's Number
n_x = 101                               
dx = L/ (n_x - 1)
dt = (beta * dx) / alpha
nt = 300
x = np.linspace(0,L,n_x)

#Initial Conditions
u0 = np.exp(-200 * (x-1)**2)                                #Gaussian Pulse
u0_dt = np.zeros(n_x)

u_old, u = u0.copy(), u0.copy()
u[1:-1] = u_old[1:-1] + dt * u0_dt[1:-1] + 0.5*(beta**2)*(u_old[2:] - 2*u_old[1:-1] + u_old[:-2])

# Dirichlet Boundary Condition
u[0] = 0
u[-1] = 0



fig, ax = plt.subplots(facecolor = "#0B0F19")

wave, = ax.plot(x,u, color = '#E60033')

ax.grid(color = '#374151')
ax.set_title('Hyperbolic-EDP-Practice', color= '#FFFFFF')

ax.set_xlim(0, L)
ax.set_ylim(-1.1, 1.1)
ax.set_xlabel("x [m]", color='#E5E7EB')
ax.set_ylabel("u", color='#E5E7EB')
ax.set_facecolor('#111827')
ax.tick_params(colors='#9CA3AF', labelsize=10)

for spine in ax.spines.values():
    spine.set_edgecolor('#374151')

def animate(i):
    global u, u_old
    u_new = np.zeros(n_x)
    u_new[1:-1] = 2* u[1:-1] - u_old[1:-1] + (beta**2) * (u[2:] - 2*u[1:-1] + u[:-2])
    u_old, u = u, u_new
    wave.set_ydata(u)
    return wave,

anim = FuncAnimation(fig, animate, frames = nt, interval = 20, blit = False)

plt.show()