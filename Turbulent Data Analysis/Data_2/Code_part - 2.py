# %%
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

data = np.load("isotropic1024_slice.npz" )
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
l = 1.364 # integral length scale 
nu = 0.000185
rms_u = 0;
rms_v = 0;
rms_w = 0;
pts = Nx * Ny # represents total grid pts


for i in range(1024):
    for j in range(1024):
        rms_u = rms_u + (u[i][j])*(u[i][j])
        rms_v = rms_v + (v[i][j])*(v[i][j])
        rms_w = rms_w + (w[i][j])*(w[i][j])

rms_u = np.sqrt(rms_u/pts)
rms_v = np.sqrt(rms_v/pts)
rms_w = np.sqrt(rms_w/pts)

U = rms_u;
epsilon = (U ** 3)/l;
eta = ((nu ** 3)/epsilon) ** 0.25; # kolmogorov length scale
tau_eta = (nu / epsilon) ** 0.5; # kolmogorov time scale


# Problem - 1 : Verifying the parseval's theorem :

u_data = u[0] # physical space
u_hat = np.fft.fft(u_data,n = None,axis = 0) # Fourier space

val1 = 0 # integration wrt x
val2 = 0 # integration wrt k

for i in range(Nx):
    val1 = val1 + u_data[i] ** 2;
    val2 = val2 + (np.abs(u_hat[i]) ** 2)/Nx

print(val1)
print(val2)







# %%
# Problem - 2 : we need to find the fourier transform of the KE along the x - direction for every row.

E_avg = np.zeros(1024)
for i in range(Ny):
    u_hat = np.fft.fft(u[i],n = None,axis = 0)
    v_hat = np.fft.fft(v[i],n = None,axis = 0)
    E = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
    E_avg = E_avg + E;
E_avg = (E_avg/Ny)

k = np.fft.fftfreq(Nx, dx) * 2 * np.pi
k = np.fft.fftshift(k)
E_avg = np.fft.fftshift(E_avg)

# Inertial range data

idx2 = (2 * np.pi) / (l/6);
idx1 = (2 * np.pi) / (60 * eta);

idx1 = int(np.round(idx1))
idx2 = int(np.round(idx2))

# taking only the positive data of k and the data associated with it

k = k[513:]
E_avg = E_avg[513:]


k_inertial = k[idx2 : idx1]
E_inertial = E_avg[idx2 : idx1]
coefficients = np.polyfit(np.log(k_inertial),np.log(E_inertial),1);
slope, intercept = coefficients
print(f"Slope: {slope}")
E_fit = (np.e ** intercept) * (k_inertial ** slope)
# print(E_fit)
# print(E_inertial)

plt.figure()
plt.plot(k, E_avg, label="One - Dimensional Energy Spectrum",linewidth = 2)
# plt.plot(k_inertial, E_fit, label = "Fitted curve for Inertial Range", linewidth = 2)
plt.plot()
plt.yscale("log")
plt.xscale("log")
plt.title("Energy Spectrum")
plt.legend()
plt.show()

    

# %%
# Problem - 3 : Two dimensional fourier transform

u_hat = np.fft.fft2(u)
v_hat = np.fft.fft2(v)
E = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
E = np.fft.fftshift(E)
E = np.abs(E);
kx = np.fft.fftfreq(Nx, d=Lx/Nx) * Lx
ky = np.fft.fftfreq(Ny, d=Ly/Ny) * Ly
kx = np.fft.fftshift(kx)
ky = np.fft.fftshift(ky)
k = np.zeros((1024,1024))
for i in range(Nx):
    for j in range(Ny):
        k[i][j] = np.sqrt(kx[i] ** 2 + ky[j] ** 2);

k_max = np.max(k);
k_min = np.min(k);

KX, KY = np.meshgrid(kx, ky)
plt.figure()
plt.pcolormesh(KX, KY, np.log(E), shading='auto', cmap='inferno')
plt.xlabel("kx (Frequency in x)")
plt.ylabel("ky (Frequency in y)")
plt.title("2-Dimensional Energy Spectrum")
plt.colorbar()
plt.show()

# k_min = 0  and  k_max = 724.07

k_vals = np.linspace(0,724,725)
E_vals = np.zeros(725)

for i in range(1024):
    for j in range(1024):
        val = int(k[i][j])
        E_vals[val] = (E_vals[val] + E[i][j])



k_inertial = k_vals[idx2 : idx1]
E_inertial = E_vals[idx2 : idx1]
coefficients = np.polyfit(np.log(k_inertial),np.log(E_inertial),1);
slope, intercept = coefficients
print(f"Slope: {slope}")
E_fit = (np.e ** intercept) * (k_inertial ** slope)

plt.figure()
plt.plot(k_vals, E_vals, label = "Two - Dimensional Energy Spectrum", linewidth = 2)
plt.plot(k_inertial, E_fit, label = "Fitted curve for Inertial Range", linewidth = 2)
plt.plot()
plt.yscale("log")
plt.xscale("log")
plt.title("Energy Spectrum")
plt.legend()
plt.show()





# %%
# Problem - 4 : longitudinal and transverse correlation functions

# Boundary conditions are periodic ----> u(l,y) = u(l+2*pi,y) ----> only true for x but not for the y direction

u_mean = np.mean(u)
v_mean = np.mean(v)

u_prime = u - u_mean  # subtract mean from u component
v_prime = v - v_mean  # subtract mean from v component


u_prime_rms = np.sqrt(np.mean(u_prime**2))
v_prime_rms = np.sqrt(np.mean(v_prime**2))

    
correlation_ux = np.zeros(Nx)
correlation_uy = np.zeros(Nx)
correlation_vx = np.zeros(Nx)
correlation_vy = np.zeros(Nx)



r = np.zeros(Nx)
for i in range(Nx):
    r[i] = i * dx


for k in range (Nx):
    if (k % 100) == 0:
        print(k)
    correlation_r1 = 0
    correlation_r2 = 0
    correlation_r3 = 0
    correlation_r4 = 0
    for i in range(Ny):
        for j in range(Nx):
            correlation_r1 = correlation_r1 + (u_prime[i][j] * u_prime[i][((j + k) % 1024)])
            correlation_r2 = correlation_r2 + (v_prime[i][j] * v_prime[((i + k) % 1024)][j])
            correlation_r3 = correlation_r3 + (v_prime[i][j] * v_prime[i][((j + k) % 1024)])
            correlation_r4 = correlation_r4 + (u_prime[i][j] * u_prime[((i + k) % 1024)][j])
            
    correlation_ux[k] = ((correlation_r1/pts)/(u_prime_rms ** 2))
    correlation_vy[k] = ((correlation_r2/pts)/(v_prime_rms ** 2))
    correlation_vx[k] = ((correlation_r3/pts)/(v_prime_rms ** 2))
    correlation_uy[k] = ((correlation_r4/pts)/(u_prime_rms ** 2))



plt.figure()
plt.plot(r, correlation_ux, label = "Correlation of the fluctuating u-component in x", linewidth = 2)
plt.plot(r, correlation_vx, label = "Correlation of the fluctuating v-component in x", linewidth = 2)
plt.plot(r, correlation_uy, label = "Correlation of the fluctuating u-component in y", linewidth = 2)
plt.plot(r, correlation_vy, label = "Correlation of the fluctuating v-component in y", linewidth = 2)
plt.plot()
plt.title("Correlation functions of different fluctuating components of velocities")
plt.legend()
plt.show()




# %%
plt.figure()
plt.plot(r, correlation_ux, label = "Correlation of u-component in x", linewidth = 2)
plt.plot(r, correlation_vx, label = "Correlation of v-component in x", linewidth = 2)
plt.plot(r, correlation_uy, label = "Correlation of u-component in y", linewidth = 2)
plt.plot(r, correlation_vy, label = "Correlation of v-component in y", linewidth = 2)
plt.plot()
plt.xlabel("r") 
plt.title("Correlation functions of different components of velocities")
plt.legend()
plt.show()

# %%
# Problem - 5 : Longitudinal Structure Functions


p_val = np.array([1,2,3,4,5,6,7])
S_p = np.zeros((7,1024))
theta = np.pi/6
th_val = np.zeros(round((2 * np.pi)/theta))
for i in range(np.size(th_val)):
    th_val[i] = i * theta
th_cos = np.cos(th_val)
th_sin = np.sin(th_val)
sz = np.size(th_val) # size of the theta array
for p in range(1,8):
    print(p)
    for r_val in range(0,512,20):
        s_p = 0; # a dummy variable for calculations
        print(r_val)
        d = r_val * dx
        for i in range(Ny):
            for j in range(Nx):
                # we want to rotate the d in different directions and average ----> <(u(x+d) - u(x)).r_hat>
                for th in range(sz):
                    # we want to find u at x+d in a particular direction
                    x_old = j * dx
                    y_old = i * dy

                    x_new = x_old + d * th_cos[th]
                    y_new = y_old + d * th_sin[th]

                    if x_new < 0:
                        x_new = x_new + Lx
                    if y_new < 0:
                        y_new = y_new + Ly
                    if x_new >= Lx:
                        x_new = x_new - Lx
                    if y_new >= Ly:
                        y_new = y_new - Ly

                    # new indices ----> not the exact pt but near the exact pt.
                    i_new = (round(y_new/dy)) % 1024
                    j_new = (round(x_new/dx)) % 1024

                    delta_u = u[i_new][j_new] - u[i][j];
                    delta_v = v[i_new][j_new] - v[i][j];

                    s_p = s_p + (np.abs(delta_u * th_cos[th] + delta_v * th_sin[th]))**p

        S_p[p-1][r_val] = (s_p/(pts * np.size(th_val)))


# r = np.zeros(Nx/2)
# for i in range(Nx/2):
#     r[i] = i * dx

# by the above code our S[p][r_val] array will be filled

# plt.figure()
# plt.plot(S_p[2], r * dx, label = "S_3", linewidth = 2)
# plt.plot()
## plt.yscale("log")
## plt.xscale("log")
# plt.title("4/5 lay of kolmogorov")
# plt.legend()
# plt.show()


                    
     


# %%
# Problem - 5(a)


S_p = np.load('matrix.npy') # This matrix took 4-5 hrs to compute. That is my I saved it as matrix.npy.


r = np.zeros(26)  # I am taking only 26 values of r as the code runs very slow. 
for i in range(26):
    r[i] = i * dx * 20

# by the above code our S[p][r_val] array will be filled
S_1 = S_p[0][0:512:20]
S_2 = S_p[1][0:512:20]
S_3 = S_p[2][0:512:20]
S_4 = S_p[3][0:512:20]
S_5 = S_p[4][0:512:20]
S_6 = S_p[5][0:512:20]
S_7 = S_p[6][0:512:20]

plt.figure()
plt.plot(r,S_1, label = "S_1", linewidth = 2)
plt.plot(r,S_2, label = "S_2", linewidth = 2)
plt.plot(r,S_3, label = "S_3", linewidth = 2)
plt.plot(r,S_4, label = "S_4", linewidth = 2)
plt.plot(r,S_5, label = "S_5", linewidth = 2)
plt.plot(r,S_6, label = "S_6", linewidth = 2)
plt.plot(r,S_7, label = "S_7", linewidth = 2)
plt.xlabel("r")  
plt.ylabel("Structure functions")  
plt.plot()
plt.title("Structure functions vs r")
plt.legend()
plt.show()



# %%
# Problem - 5(b)

coefficients = np.polyfit(r,S_3,1);
slope, intercept = coefficients
S_3_fit = slope * r + intercept
# r = r[0:18]
# S_3 = S_3[0:18]
coefficients = np.polyfit(r,S_3,1);
slope, intercept = coefficients
S_3_fit = slope * r + intercept
plt.figure()
plt.plot(r, S_3, label = "S_3", linewidth = 2)
plt.plot(r, S_3_fit, label = "S_3_fit", linewidth = 2)
plt.plot()
plt.title("Verifying the 4/5-th law of kolmogorov")
plt.legend()
plt.show()

# Slope calculated from calculation:
epsilon = ((rms_u**2 + rms_v**2) ** 1.5)/l
calc_slope = 0.8 * epsilon
print(f"Slope: {slope}")
print(f"Calculated Slope: {calc_slope}")




# %%
# Problem - 5(c)

plt.figure()
plt.plot(S_3,S_1, label = "S_1", linewidth = 2)
plt.plot(S_3,S_2, label = "S_2", linewidth = 2)
plt.plot(S_3,S_3, label = "S_3", linewidth = 2)
plt.plot(S_3,S_4, label = "S_4", linewidth = 2)
plt.plot(S_3,S_5, label = "S_5", linewidth = 2)
plt.plot(S_3,S_6, label = "S_6", linewidth = 2)
plt.plot(S_3,S_7, label = "S_7", linewidth = 2)

plt.plot()
plt.xlabel("S_3")  
plt.ylabel("Structure functions")  
plt.yscale("log")
plt.xscale("log")
plt.title("ESS Plots or log - log plot of Structure Functions vs S_3")
plt.legend()
plt.show()

# %%
# Problem - 5(d)

coefficients1 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_1[S_1 > 0]),1);
coefficients2 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_2[S_2 > 0]),1);
coefficients3 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_3[S_3 > 0]),1);
coefficients4 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_4[S_4 > 0]),1);
coefficients5 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_5[S_5 > 0]),1);
coefficients6 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_6[S_6 > 0]),1);
coefficients7 = np.polyfit(np.log(S_3[S_3 > 0]),np.log(S_7[S_7 > 0]),1);
slope1, intercept1 = coefficients1
slope2, intercept2 = coefficients2
slope3, intercept3 = coefficients3
slope4, intercept4 = coefficients4
slope5, intercept5 = coefficients5
slope6, intercept6 = coefficients6
slope7, intercept7 = coefficients7
slopes = np.array([slope1, slope2, slope3, slope4, slope5, slope6, slope7])
arr = np.arange(1, 8)
errors = np.abs(slopes - arr * 0.3333333)
plt.figure()
plt.plot(arr,slopes, label = "Slopes from ESS Plots", linewidth = 2)
plt.errorbar(arr, 0.333333 * arr, yerr=errors, fmt='o', capsize=5, label=r'Absolute deviation')
plt.plot(arr,0.333333 * arr, label = "Kolmogorov's Prediction", linewidth = 2)
plt.plot()
# plt.xlabel("p/3")  
plt.ylabel("p")
plt.title("Deviation from the Kolmogorov's Prediction with error bars")
plt.legend()
plt.show()

print(slopes)
print(0.333333 * arr)

# %%


# %%



