import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
import imageio


def bracket(w, M, b):
    return 1 - w**2 * (1 - 2 * M / b * w)

def integrand(w, M, b):
    return 1 / np.sqrt(bracket(w, M, b))

def integral(w, M, b):
    return integrate.quad(integrand, 0, w, (M, b))[0]

def turning_point(b, M=1):
    w1 = fsolve(bracket, 1, (M, b))[0]
    return integral(w1, M, b)

# Resolution of texture
resolution = 64

# 2D array for the texture
phi_tp_texture = np.zeros((resolution, resolution))

b = 5.1962 / np.linspace(0, 1, resolution**2)

phi_TP = np.array([turning_point(b[i]) for i in range(len(b))])
phi_TP[0] = np.pi/2

# Filling in texture
for i in range(resolution):
    for j in range(resolution):
        k = i * resolution + j
        phi_tp_texture[i, j] = phi_TP[k]


# Plotting phi_TP
# ===================================================================================================
def plot1D():
    plt.scatter(1/b, phi_TP, s=2)

    plt.xlabel('1/b')
    plt.ylabel('$\\phi_{TP}$')
    plt.show()

#plot1D()
# ===================================================================================================



# Plotting Texture
# ===================================================================================================
def textureImshow(log=False):
    # Creating figure
    # -------------------------------------
    plt.figure(figsize=(8, 6), facecolor="#1c1c1c")
    ax = plt.axes()
    ax.set_facecolor("#1c1c1c")
    mpl.rcParams.update({'font.size': 15})
    # -------------------------------------

    # Plotting the 2d array as an image
    if log:
        plt.imshow(np.log10(phi_tp_texture), origin='lower', cmap='gray')
    else:
        plt.imshow(phi_tp_texture, origin='lower', cmap='gray')

    # Plot layout
    # -------------------------------------
    cb = plt.colorbar()
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    if log:
        plt.title("$\\log(\\Delta\\phi)$", color='white')
    else:
        plt.title("$\\Delta\\phi$", color='white')

    ax.set_xticks([0, int(resolution/2), resolution-1])

    ax.set_yticks([0, int(resolution/2), resolution-1])

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['bottom'].set_color('white')
    # -------------------------------------

    plt.show()

#textureImshow(log=True)
# ===================================================================================================


# Saving Texture
# ===================================================================================================
def saveTexture():
    phi_arr = phi_tp_texture.astype('float32')

    # Flip the y direction
    phi_arr = np.flip(phi_arr, 0)

    # Write to disk
    imageio.imwrite('phi_turning_point.tiff', phi_arr)

saveTexture()
# ===================================================================================================

print('-- DONE! --')