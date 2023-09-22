import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as integrate
import imageio
from math import isnan


filters = np.loadtxt('CIE_xyz_1964_10deg.csv', delimiter=',')

# Constants
h = 6.62607015e-34 # J/Hz
k_b = 1.380649e-23 # J/K
c = 299792458.0 # m/s

def Blackbody(wavelength, T):
    wl = wavelength*1e-9
    return 2*h*c**2 / wl**5 / (np.exp(h*c / (wl*k_b*T)) - 1)


resolution = 64

wl = np.linspace(filters[0, 0], filters[-1, 0], 1000)
T = np.logspace(2.6, 6, resolution**2)

# calculating XYZ(T)
X = np.array([integrate(Blackbody(wl, T[i]) * np.interp(wl, filters[:, 0], filters[:,1]), wl)[-1] for i in range(len(T))])
Y = np.array([integrate(Blackbody(wl, T[i]) * np.interp(wl, filters[:, 0], filters[:,2]), wl)[-1] for i in range(len(T))])
Z = np.array([integrate(Blackbody(wl, T[i]) * np.interp(wl, filters[:, 0], filters[:,3]), wl)[-1] for i in range(len(T))])

# calculating RGB(T)
red = (8041697*X - 3049000*Y - 1591847*Z)/3400850
green = (-1752003*X + 4851000*Y + 301853*Z)/3400850
blue = (17697*X - 49000*Y + 3432153*Z)/3400850


blackbody_texture = np.zeros((resolution, resolution, 3))
blackbody_arr = np.zeros((resolution**2, 3))

# Filling in texture
for i in range(resolution):
    for j in range(resolution):
        x = i * resolution + j
        blackbody_texture[i, j, 0] = np.log10(red[x]) + 10
        blackbody_texture[i, j, 1] = np.log10(green[x]) + 10
        blackbody_texture[i, j, 2] = np.log10(blue[x]) + 10


for i in range(resolution):
    for j in range(resolution):
        for k in range(3):
            if isnan(blackbody_texture[i, j, k]):
                blackbody_texture[i, j, k] = 0



# Test plots
# ===================================================================================================
def Filter_plot():
    plt.plot(filters[:,0], filters[:,1])
    plt.plot(filters[:,0], filters[:,2])
    plt.plot(filters[:,0], filters[:,3])
    plt.show()
#Filter_plot()


def Test_plot():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    #T_high = np.logspace(4, 6)

    plt.plot(T, red, color='#DD3333', linewidth=2)
    plt.plot(T, green, color='#33DD33', linewidth=2)
    plt.plot(T, blue, color='#3333FF', linewidth=2)

    #plt.plot(T_high, red[-1] * T_high*1e-6, color='#D1D1D1', linestyle = ':')
    #plt.plot(T_high, green[-1] * T_high*1e-6, color='#D1D1D1', linestyle = ':')
    #plt.plot(T_high, blue[-1] * T_high*1e-6, color='#D1D1D1', linestyle = ':')

    plt.semilogx()
    plt.semilogy()
    plt.xlabel('$\\log T [k]$', fontsize=16, color='#D1D1D1')
    plt.ylabel('$\\log RGB$', fontsize=16, color='#D1D1D1')
    fig.patch.set_facecolor('#1C1C1C')
    ax.set_facecolor('#1C1C1C')
    ax.spines['bottom'].set_color('#D1D1D1')
    ax.spines['top'].set_color('#D1D1D1') 
    ax.spines['right'].set_color('#D1D1D1')
    ax.spines['left'].set_color('#D1D1D1')
    ax.tick_params(axis='both', colors='#D1D1D1', labelsize=16)
    plt.tight_layout()
    #plt.savefig('Blackbody Colors Plot.png', dpi=300)
    plt.show()
#Test_plot()
# ===================================================================================================



# Saving Texture
# ===================================================================================================
def saveTexture():
    bb_arr = blackbody_texture.astype('float32')

    # Flip the y direction
    bb_arr = np.flip(bb_arr, 0)

    # Write to disk
    imageio.imwrite('blackbody_color.tiff', bb_arr)

#saveTexture()
# ===================================================================================================

print('-- DONE! --')
