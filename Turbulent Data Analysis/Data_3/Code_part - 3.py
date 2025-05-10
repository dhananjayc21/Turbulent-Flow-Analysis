# %%
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import random
from scipy.stats import norm


data = np.load("isotropic1024_stack3.npz") # Remember that data is not periodic in the z - direction
u = data["u"]
v = data["v"]
w = data["w"]

# part-1 ---> calculating the kolmogorv stats:
Lx = 2 * np.pi # size of the domain
Ly = 2 * np.pi
Nx = 1024 # no. of grid points in one direction
Ny = 1024
dx = Lx/Nx; # size of grid cell
dy = Ly/Ny;
dz = dx
l = 1.364 # integral length scale 
nu = 0.000185
rms_u = 0;
rms_v = 0;
rms_w = 0;
pts = Nx * Ny * 3 # represents total grid pts

# Remember ---> i == x, j == y, k == z

for i in range(1024):
    for j in range(1024):
        for k in range(3):
            rms_u = rms_u + (u[i][j][k])*(u[i][j][k])
            rms_v = rms_v + (v[i][j][k])*(v[i][j][k])
            rms_w = rms_w + (w[i][j][k])*(w[i][j][k])

rms_u = np.sqrt(rms_u/pts)
rms_v = np.sqrt(rms_v/pts)
rms_w = np.sqrt(rms_w/pts)

U = rms_u;

Re = (U * l)/nu; # Reynolds number

epsilon = (U ** 3)/l;
eta = ((nu ** 3)/epsilon) ** 0.25; # kolmogorov length scale
tau_eta = (nu / epsilon) ** 0.5; # kolmogorov time scale
u_eta = (nu * epsilon) ** 0.25; # kolmogorov velocity scale
Re_eta = (u_eta * eta)/nu # this should be one
dt = (tau_eta / 20)   # Timestep


# %%
# Problem - 2 : Calculating the velocity gradient tensor for the middle plane:

A = np.zeros((Nx, Ny, 3, 3))
for i in range(1024):
    for j in range(1024):
        A[i][j][0][0] = (u[(i+1) % 1024][j][1] - u[((i-1) + 1024) % 1024][j][1])/(2 * dx)   # a_11 = du/dx
        A[i][j][1][0] = (v[(i+1) % 1024][j][1] - v[((i-1) + 1024) % 1024][j][1])/(2 * dx)   # a_21 = dv/dx
        A[i][j][2][0] = (w[(i+1) % 1024][j][1] - w[((i-1) + 1024) % 1024][j][1])/(2 * dx)   # a_31 = dw/dx
        

        A[i][j][0][1] = (u[i][(j+1) % 1024][1] - u[i][((j-1) + 1024) % 1024][1])/(2 * dy)   # a_12 = du/dy
        A[i][j][1][1] = (v[i][(j+1) % 1024][1] - v[i][((j-1) + 1024) % 1024][1])/(2 * dy)   # a_22 = dv/dy
        A[i][j][2][1] = (w[i][(j+1) % 1024][1] - w[i][((j-1) + 1024) % 1024][1])/(2 * dy)   # a_32 = dw/dy


        A[i][j][0][2] = (u[i][j][2] - u[i][j][0])/(2 * dz)   # a_13 = du/dz
        A[i][j][1][2] = (v[i][j][2] - v[i][j][0])/(2 * dz)   # a_23 = dv/dz
        A[i][j][2][2] = (w[i][j][2] - w[i][j][0])/(2 * dz)   # a_33 = dw/dz


# Finding the eigenvalues for each of the above 1024 * 1024 matrices:



eigen_vals_cmplx = np.empty((Nx, Ny, 3) ,  dtype=object)

for i in range (1024):
    for j in range(1024):
        eigen_vals_cmplx[i][j] = np.linalg.eigvals(A[i][j]) 


# Drawing the PDF's of the three lambda values where y-x is log-linear plot


counts, bin_edges = np.histogram(np.abs(eigen_vals_cmplx.ravel()), bins=200, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.plot(bin_centers, counts, label='PDF Estimate')
plt.xlabel('Value')
plt.ylabel('Probability Density Function')
plt.yscale('log')
plt.title('Custom Histogram Plot')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()


# %%
# Plotting PDF's for eigenvalues separately

eigen_vals_cmplx_copy = np.abs(eigen_vals_cmplx)
for i in range(1024):
    for j in range(1024):
        eigen_vals_cmplx_copy[i][j] = np.sort(eigen_vals_cmplx_copy[i][j])

lambda_1 = eigen_vals_cmplx_copy[:][:][2]  # largest eigenvalue
lambda_2 = eigen_vals_cmplx_copy[:][:][1]  # middle eigenvalue
lambda_3 = eigen_vals_cmplx_copy[:][:][0]  # smallest eigenvalue


# Drawing the PDF's of the three lambda values where y-x is log-linear plot


counts, bin_edges = np.histogram(lambda_3, bins=40, density=True)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
plt.plot(bin_centers, counts,)
plt.xlabel(r'$| \lambda_3 |$')  # change i in lambda_i(i = 1,2,3) accordingly
plt.ylabel('Probability Density')
plt.yscale('log')
# plt.xscale('log')
plt.title('Probability Density Function of the magnitudes of the smallest eigenvalue')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
# Verify whether ⟨|P|⟩ ≈ 0 over the 2D domain

P = 0

for i in range(1024):
    for j in range(1024):
        P = P + abs(-1 * (eigen_vals_cmplx[i][j][0] + eigen_vals_cmplx[i][j][1] + eigen_vals_cmplx[i][j][2]))
P = (P/(1024 * 1024))

print(P)  #  ---->  value is coming to be 0.33707765285807206


# %%
# Calculation of Q/Q_rms ----> Q_normalized


Q = np.zeros((1024,1024))
Q_rms = 0
for i in range(1024):
    for j in range(1024):
        Q[i][j] = np.real(eigen_vals_cmplx[i][j][0] * eigen_vals_cmplx[i][j][1] + eigen_vals_cmplx[i][j][1] * eigen_vals_cmplx[i][j][2] + eigen_vals_cmplx[i][j][0] * eigen_vals_cmplx[i][j][2])

for i in range(1024):
    for j in range(1024):
        Q_rms = Q_rms + (Q[i][j]) ** 2

Q_rms = (Q_rms/(1024 * 1024)) ** 0.5

Q_normalized = (Q / Q_rms)


# Calculation of R/R_rms -----> R_normalized

R = np.zeros((1024,1024))
R_rms = 0
for i in range(1024):
    for j in range(1024):
        R[i][j] = np.real(-1 * eigen_vals_cmplx[i][j][0] * eigen_vals_cmplx[i][j][1] * eigen_vals_cmplx[i][j][2])

for i in range(1024):
    for j in range(1024):
        R_rms = R_rms + (R[i][j]) ** 2

R_rms = (R_rms/(1024 * 1024)) ** 0.5

R_normalized = (R / R_rms)

# Plotting the fields of Q_normalized and R_normalized

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

N = 1024


im1 = axes[0].pcolor(Q_normalized[:N,:N], cmap='viridis', vmin=-1, vmax=1)
axes[0].set_title(r'Field of $\frac{Q}{Q_{\mathrm{rms}}}$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
fig.colorbar(im1, ax=axes[0])


im2 = axes[1].pcolor(R_normalized[:N,:N], cmap='viridis', vmin=-1, vmax=1)
axes[1].set_title(r'Field of $\frac{R}{R_{\mathrm{rms}}}$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# %%
# Entrosphy calculation for normalising Q and R and plotting Q_norm vs R_norm


# Calculating the required derivatives: ----> all calculated for the mid z - plane

vorti_x = np.zeros((1024,1024))
vorti_y = np.zeros((1024,1024))
vorti_z = np.zeros((1024,1024))

entrosphy = np.zeros((1024,1024))
avg_entrosphy = 0

for i in range(1024):
    for j in range(1024):
        vorti_x[i][j] = A[i][j][1][2] - A[i][j][2][1]  # dv/dz - dw/dy
        vorti_y[i][j] = A[i][j][0][2] - A[i][j][2][0]  # du/dz - dw/dx
        vorti_z[i][j] = A[i][j][1][0] - A[i][j][0][1]  # dv/dx - du/dy
        entrosphy[i][j] = (vorti_x[i][j] ** 2 + vorti_y[i][j] ** 2 + vorti_z[i][j] ** 2)
        avg_entrosphy = avg_entrosphy + entrosphy[i][j]

avg_entrosphy = (avg_entrosphy/(1024 * 1024))

Q_w = (avg_entrosphy/4)

Q_normalised = (Q / Q_w)

R_normalised = (R / (Q_w ** 1.5))




Q_arr = np.linspace(-40, 40, 1000)
R_squared = -(4 * Q_arr**3) / 27
mask = R_squared >= 0
R_pos = np.sqrt(R_squared[mask])
R_neg = -R_pos
plt.plot(R_pos, Q_arr[mask], 'r', label=r'$27R^2 + 4 Q^3 = 0$')
plt.plot(R_neg, Q_arr[mask], 'r')
plt.scatter(R_normalized, Q_normalised, marker='.', alpha=0.1)   # Scatter-plot/joint-distribution between Q_normalised and R_normalised


plt.xlabel('$R$')
plt.ylabel('$Q$')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True)
plt.legend(loc="upper right")
plt.show()


        

# %%
# Finding the prominent flow topologies

ct_9a = 0
ct_10b = 0
ct_6a = 0
ct_6b = 0
ct_12c = 0
ct_8c = 0
ct_7a = 0
ct_7b = 0

for i in range(1024):
    for j in range(1024):
        if (27 * R_normalised[i][j] * R_normalised[i][j] + 4 * Q_normalised[i][j]* Q_normalised[i][j]* Q_normalised[i][j]) > 0:
            
            if R_normalised[i][j] > 0:
                ct_10b = ct_10b + 1
            elif R_normalised[i][j] < 0:
                ct_9a = ct_9a + 1
            else:
                ct_12c = ct_12c + 1
                
        elif (27 * R_normalised[i][j] * R_normalised[i][j] + 4 * Q_normalised[i][j]* Q_normalised[i][j]* Q_normalised[i][j]) < 0:
            
            if R_normalised[i][j] > 0:
                ct_6b = ct_6b + 1
            elif R_normalised[i][j] < 0:
                ct_6a = ct_6a + 1
            else:
                ct_8c = ct_8c + 1
        else:
            if R_normalised < 0:
                ct_7a = ct_7a + 1
            else:
                ct_7b = ct_7b + 1



print(ct_9a) # ------> 383792 -----> stable focus/stretching -----> Highest
print(ct_10b) # ------> 256578 ------> unstable focus/compressing
print(ct_6a) # ------> 94821 ------> stable node/saddle/saddle
print(ct_6b) # -----> 313385 ------> unstable node/saddle/saddle -----> Second Highest
print(ct_12c) # ----> 0
print(ct_8c) # -----> 0
print(ct_7a) # -----> 0
print(ct_7b) # ------> 0

# %%
# The joint probability distribution


x = R_normalised.flatten()
y = Q_normalised.flatten()

fig, ax = plt.subplots(figsize=(8, 6))
hist, xedges, yedges = np.histogram2d(x, y, bins = 4000, density = True)
hist_masked = np.ma.masked_where(hist < 10 ** -4, hist)
hb = ax.pcolormesh(xedges, yedges, np.log10(hist_masked.T), cmap='plasma', shading='auto')

Q_arr = np.linspace(-40, 40, 1000)
R_squared = -(4 * Q_arr**3) / 27
mask = R_squared >= 0
R_pos = np.sqrt(R_squared[mask])
R_neg = -R_pos
plt.plot(R_pos, Q_arr[mask], 'r', label=r'$27R^2 + 4 Q^3 = 0$')
plt.plot(R_neg, Q_arr[mask], 'r')


cb = plt.colorbar(hb, ax=ax)
cb.set_label('Density')
plt.xlabel('$R$')
plt.ylabel('$Q$')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True)
plt.legend(loc="upper right")
plt.show()

# %%
# Lagrangian Aspects of Turbulence

# Problem - 1 & 2


T = 10   # Total Time
time_steps = int(np.round(T / dt))   # No. of timesteps

print(time_steps)

N_p = 1000  # No. of particles 

# Now we want to place 20 particles at random cooordinates in the grid:

all_coords = [(x, y) for x in range(Nx) for y in range(Ny)]
particles = random.sample(all_coords, N_p)

# print(particles)

trajectory = np.zeros((N_p,time_steps + 1,2)) # storing the trajectory of the particles in a 3d-array
i = 0  # Just a variable i have made for iterating

for particle in particles:
    x_par = particle[0]
    y_par = particle[1]
    trajectory[i][0][0] = x_par * dx
    trajectory[i][0][1] = y_par * dx
    i = i + 1

for step in range(0,time_steps):
    for i in range(N_p):

        x_old = trajectory[i][step][0]  # old coord in x
        y_old = trajectory[i][step][1]  # old coord in y
        
        x_old_coord = (((int(np.round(x_old/dx))) % 1024) + 1024) % 1024  # wrapping coordinates to get the velocity
        y_old_coord = (((int(np.round(y_old/dy))) % 1024) + 1024) % 1024

        
        x_new = x_old + dt * u[x_old_coord][y_old_coord][0]  # new coord in x
        y_new = y_old + dt * v[x_old_coord][y_old_coord][0]  # new coord in y


        trajectory[i][step + 1][0] = x_new   # putting the new coordintes in the trajectory array
        trajectory[i][step + 1][1] = y_new


# %%
# Plotting the above trajectories

plt.figure(figsize=(8, 6))
for p in range(N_p):
    x = trajectory[p, 0:594, 0]
    y = trajectory[p, 0:594, 1]
    plt.plot(x, y)

plt.title(f"2D Trajectories of {N_p} Particles for T = {T}")
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()

# %%
# Problem - 3 ---->   Plot the mean-square-displacement as a function of (pseudo)time

MSD = np.zeros((time_steps + 1, 1))
for step in range(time_steps + 1):
    for i in range(N_p):
        MSD[step] = MSD[step] + (trajectory[i][step][0] - trajectory[i][0][0]) ** 2 + (trajectory[i][step][1] - trajectory[i][0][1]) ** 2
    MSD[step] = (MSD[step] / N_p)
time_step_array = np.zeros((time_steps + 1, 1))
for i in range(time_steps + 1):
    time_step_array[i] = i


# %%
# MSD plot to check the diffusive and ballistic regime

t = time_step_array * dt
MSD_part1 = MSD[1:1500]
t_part1 = t[1:1500]
MSD_part2 = MSD[3750:10000]
t_part2 = t[3750:10000]


slope1, intercept1 = np.polyfit(np.log10(t_part1.ravel()), np.log10(MSD_part1.ravel()), 1)
slope2, intercept2 = np.polyfit(np.log10(t_part2.ravel()), np.log10(MSD_part2.ravel()), 1)
reference_line1 = (t_part1 ** slope1) * (10 ** intercept1)
reference_line2 = (t_part2 ** slope2) * (10 ** intercept2)

 
plt.figure()
plt.plot(t[1:10000], MSD[1:10000], color = 'black', label="Mean Square Displacement", linewidth=2)
plt.plot(t_part1, reference_line1, color='red', linestyle='--', linewidth=2.5, label="Slope = 2 (reference)", alpha=0.7)
plt.plot(t_part2, reference_line2, color='blue', linestyle='--', linewidth=2.5, label="Slope = 1 (reference)", alpha=0.7)

plt.xscale("log")
plt.yscale("log")

# Labels and legend
plt.xlabel("Time")
plt.ylabel("Mean Square Displacement")
plt.title("MSD(t) vs t")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()


# %%
# Turbulent Diffusivity Calculation

MSD_part = MSD[3750:10000]
t_part = t[3750:10000]


slope, intercept = np.polyfit(t_part.ravel(), MSD_part.ravel(), 1)
slope = 1.12 # found using hit and try with the graph as I did not found the python fitting satisfactory
fit_line = (t_part ** slope) + intercept


plt.figure()
plt.plot(t[:10000], MSD[:10000], label="Mean Square Displacement", linewidth=2)
plt.plot(t_part, fit_line, '--', label=f"Curve fit with Slope ≈ {slope:.2f}", linewidth=2)
# Labels and legend
plt.xlabel("Time")
plt.ylabel("Mean Square Displacement")
plt.title("Diffusivity Calculation using the linear zone of MSD(t) vs Time")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()

Turbulent_diffusivity_coeff = slope  # Found using the slope of the above linear curve fit
Dye_diffusivity_coeff = 10 ** -10   # This I found from google. Also it is an order of magnitude and not an exact value

Ratio = (Turbulent_diffusivity_coeff / Dye_diffusivity_coeff)

print(Ratio)  # Ratio of turbulent diffusivity and dye_diffusivity coefficients


# %%
# Richardson pair dispersion


N_p = 1000   # No. of particle pairs

# Now we want to place 20 particles at random cooordinates in the grid:

all_coords = [(x, y) for x in range(Nx) for y in range(Ny)]
particles = random.sample(all_coords, N_p)

# %%
# Plotting the pair trajectories


T = 15  # Total Time
time_steps = int(np.round(T / dt))   # No. of timesteps
epsilon = 10.0 * dx   # Can be changed

print(time_steps)

# N_p = 2000   # No. of particle pairs

# # Now we want to place 20 particles at random cooordinates in the grid:

# all_coords = [(x, y) for x in range(Nx) for y in range(Ny)]
# particles = random.sample(all_coords, N_p)

# print(particles)

trajectory_A = np.zeros((N_p,time_steps + 1,2)) # storing the trajectory of the particles of set-A in a 3d-array
trajectory_B = np.zeros((N_p,time_steps + 1,2)) # storing the trajectory of the particles of set-B in a 3d-array

i = 0  # Just a variable i have made for iterating

for particle in particles:
    x_par = particle[0]
    y_par = particle[1]
    trajectory_A[i][0][0] = x_par * dx
    trajectory_A[i][0][1] = y_par * dx

    if i < (N_p // 4):
        trajectory_B[i][0][0] = x_par * dx - epsilon
        trajectory_B[i][0][1] = y_par * dx
    elif i < (N_p // 2):
        trajectory_B[i][0][0] = x_par * dx + epsilon
        trajectory_B[i][0][1] = y_par * dx
    elif i < ((3 * N_p) // 4):
        trajectory_B[i][0][0] = x_par * dx
        trajectory_B[i][0][1] = y_par * dx - epsilon
    else:
        trajectory_B[i][0][0] = x_par * dx
        trajectory_B[i][0][1] = y_par * dx + epsilon

    i = i + 1

for step in range(0,time_steps):
    for i in range(N_p):

        # Set - A
        

        x_old_A = trajectory_A[i][step][0]  # old coord in x
        y_old_A = trajectory_A[i][step][1]  # old coord in y
        
        x_old_coord_A = (((int(np.round(x_old_A/dx))) % 1024) + 1024) % 1024  # wrapping coordinates to get the velocity
        y_old_coord_A = (((int(np.round(y_old_A/dy))) % 1024) + 1024) % 1024

        
        x_new_A = x_old_A + dt * u[x_old_coord_A][y_old_coord_A][0]  # new coord in x
        y_new_A = y_old_A + dt * v[x_old_coord_A][y_old_coord_A][0]  # new coord in y


        trajectory_A[i][step + 1][0] = x_new_A  # putting the new coordintes in the trajectory array
        trajectory_A[i][step + 1][1] = y_new_A


        # Set - B


        x_old_B = trajectory_B[i][step][0]  # old coord in x
        y_old_B = trajectory_B[i][step][1]  # old coord in y
        
        x_old_coord_B = (((int(np.round(x_old_B/dx))) % 1024) + 1024) % 1024  # wrapping coordinates to get the velocity
        y_old_coord_B = (((int(np.round(y_old_B/dy))) % 1024) + 1024) % 1024
        
        x_new_B = x_old_B + dt * u[x_old_coord_B][y_old_coord_B][0]  # new coord in x
        y_new_B = y_old_B + dt * v[x_old_coord_B][y_old_coord_B][0]  # new coord in y
        
        trajectory_B[i][step + 1][0] = x_new_B  # putting the new coordinates in the trajectory array
        trajectory_B[i][step + 1][1] = y_new_B



# plt.figure(figsize=(8, 6))

# for p in range(N_p):
#     x = trajectory_A[p, ::70, 0]
#     y = trajectory_A[p, ::70, 1]
#     plt.plot(x, y, '.', color='red', alpha=0.5,linestyle = 'None', label='Trajectory A' if p == 0 else "")

# for p in range(N_p):
#     x = trajectory_B[p, ::70, 0]
#     y = trajectory_B[p, ::70, 1]
#     plt.plot(x, y, '.', color='blue', alpha=0.5,linestyle = 'None', label='Trajectory B' if p == 0 else "")

# plt.title('2D Trajectories of 20 Particles (Set - A : Red, Set - B : Blue)')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.axis('equal')
# plt.legend()
# plt.tight_layout()
# plt.show()

# %%
# Pair separation Calculation ----> delta_r2


delta_r2 = np.zeros((time_steps + 1, 1))
for step in range(time_steps + 1):
    for i in range(N_p):
        delta_r2[step] = delta_r2[step] + (trajectory_A[i][step][0] - trajectory_B[i][step][0]) ** 2 + (trajectory_A[i][step][1] - trajectory_B[i][step][1]) ** 2
    delta_r2[step] = (delta_r2[step] / N_p)
time_step_array = np.zeros((time_steps + 1, 1))
for i in range(time_steps + 1):
    time_step_array[i] = i

t = time_step_array * dt

delta_r2 = (delta_r2 / (epsilon ** 2))
t = (t / tau_eta)


# %%
# Checking the slope of pair - separation of trajectories in the inertial range

t_part = t[300:5000]
delta_r2_part = delta_r2[300:5000]
log_t = np.log10(t_part).ravel()
log_delta_r2 = np.log10(delta_r2_part).ravel()


slope, intercept = np.polyfit(log_t, log_delta_r2, 1)
fit_line = 10**(intercept) * (t_part ** slope)


plt.figure()
plt.plot(t[1:5000], delta_r2[1:5000], label=r'Simulation data', linewidth=2)
plt.plot(t_part[:], fit_line[:], '--', label=f"Curve fit with Slope ≈ {slope:.2f}", linewidth=2)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r'$\left( \frac{t}{\tau_\eta} \right)$',fontsize = 14)
plt.ylabel(r'$\left( \frac{\Delta r^2(t)}{\epsilon^2} \right)$',fontsize = 14,rotation = 0,labelpad = 30)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title(r'$\epsilon = %.1f \Delta x$' % (np.round((epsilon/dx), 2)))
plt.show()


# %%
# Lyapunov Exponent Calculation


t_part = t[1:1000].ravel()
delta_r2_part = delta_r2[1:1000]
log_delta_r2 = np.log10(delta_r2_part).ravel()


slope, intercept = np.polyfit(t_part, log_delta_r2, 1)
fit_line = 10 ** (intercept + t_part * slope)


plt.figure()
plt.semilogy(t[1:], delta_r2[1:], label=r'Simulation data', linewidth=2)
plt.semilogy(t_part[:], fit_line[:], '--', label=f"Curve fit with Slope ≈ {slope:.2f}", linewidth=2)
plt.xlabel(r'$\left( \frac{t}{\tau_\eta} \right)$',fontsize = 14)
plt.ylabel(r'$\left( \frac{\Delta r^2(t)}{\epsilon^2} \right)$',fontsize = 14,rotation = 0,labelpad = 30)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.title(r'$\epsilon = %.1f \Delta x$' % (np.round((epsilon/dx), 2)))
plt.show()


# %%



