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

def integral_r(r, M, b):
    return integrate.quad(dphi_dr, r, np.inf, (b, M))[0]

def delta_phi_func(b, M=1):
    w1 = fsolve(bracket, 1, (M, b))[0]
    return 2 * integral(w1, M, b) - np.pi

def dr_dphi(r, b, M=1):
    return -r**2 * np.sqrt(1 / b**2 - 1 / r**2 * (1 - 2 * M / r))

def dphi_dr(r, b, M=1):
    return 1/r**2 / np.sqrt(1 / b**2 - 1 / r**2 * (1 - 2*M/r))

def dy_dx(w, b, M=1):
    w1 = fsolve(bracket, 1, (M, b))[0]
    w_int = w

    # Invalid input, just use w1, this will only happen in
    # pixels of the texture that we'll never access.
    if w > w1:
        w_int = w1
    
    r = b / w_int
    phi = integral(w_int, M, b)
    return (r * np.cos(phi) + np.sin(phi) * dr_dphi(r, b)) / (-r * np.sin(phi) + np.cos(phi) * dr_dphi(r, b))

def dy_dx_r(r, b, M=1):
    if b > 5.1962:
        raise ValueError("Only use dy_dx_r for b < 5.1962!")
    
    phi = integral_r(r, M, b)
    return (r * np.cos(phi) + np.sin(phi) * dr_dphi(r, b)) / (-r * np.sin(phi) + np.cos(phi) * dr_dphi(r, b))

def check_valid(w, b):
    w1 = fsolve(bracket, 1, (1, b))[0]
    if w > w1:
        return False
    return True

def calculate_alpha(slope, phi):
    if phi > 3 * np.pi / 2 and slope > 0:
        return (np.arctan(slope) + np.pi) % np.pi + np.pi
    
    return (np.arctan(slope) + np.pi) % np.pi

def mapRange(x, from_low, from_high, to_low, to_high):
    return (x - from_low) / (from_high - from_low) * (to_high - to_low) + to_low

def b_max(r, M):
    return r**2 / np.sqrt(r**2 - 2*M*r)

def b_calc(u_r, u_phi, r, M):
    l = -r**2 * u_phi
    e = np.sqrt(u_r**2 + (r**2 - 2*M*r) * u_phi**2)
    return np.abs(l/e)


# Calculating Texture
# ===================================================================================================

resolution = 512

M = 1

x = np.linspace(0, 1, resolution) # pixel column
y = np.linspace(0, 1, resolution) # pixel row

# Preparing a 2d array, this is what will be saved to an image texture
phi_texture = np.zeros((resolution, resolution))

# Filling in texture
for i in range(1, resolution):
    r = 3/y[i]
    for j in range(resolution):
        # Calculate u_r and u_phi
        u_r = np.cos(x[j]*np.pi)
        u_phi = np.sin(x[j]*np.pi)/r

        # Calculate b
        b = b_calc(u_r, u_phi, r, M)

        phi_texture[i, j] = integral_r(r, M, b)

# r = 3M, gamma = pi/2 results in phi = infinity, so just put that pixel to a large value instead
phi_texture[-1, -1] = 10

# Towards infinite r, it seems to converge to phi = view angle
for j in range(1, resolution):
    phi_texture[0, j] = x[j] * np.pi


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
        plt.imshow(np.log10(phi_texture), origin='lower', cmap='gray')
    else:
        plt.imshow(phi_texture, origin='lower', cmap='gray')

    # Plot layout
    # -------------------------------------
    cb = plt.colorbar()
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    if log:
        plt.title("log $\\phi(r, \\gamma)$", color='white')
    else:
        plt.title("$\\phi(r, \\gamma)$", color='white')
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("M/r")

    ax.set_xticks([0, int(resolution/2), resolution-1])
    ax.set_xticklabels([0, "", "$\\pi$/2"])

    ax.set_yticks([0, int(resolution/2), resolution-1])
    ax.set_yticklabels(["0", "", "1/3"])

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


# Plotting Texture as 3D Surface
# ===================================================================================================
def surface3Dplot(log=False):
    # Creating figure
    # -------------------------------------
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    mpl.rcParams.update({'font.size': 15})
    # -------------------------------------

    # Plotting the 2d array as a 3D surface
    X, Y = np.meshgrid(x, y)
    if log:
        ax.plot_surface(X, Y, np.log10(phi_texture), linewidth=2, antialiased=True, alpha=0.5)
        ax.scatter(X, Y, np.log10(phi_texture), s=1)
    else:
        ax.plot_surface(X, Y, phi_texture, linewidth=2, antialiased=True, alpha=0.5)
        ax.scatter(X, Y, phi_texture, s=1)

    # Plot layout
    # -------------------------------------
    ax.set_xlabel("$\\gamma$")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([0, "", "$\\pi$/2"])

    ax.set_ylabel("M/r")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "", "1/3"])

    if log:
        ax.set_zlabel("log $\\phi$")
    else:
        ax.set_zlabel("$\\phi$")
    # -------------------------------------

    plt.show()


#surface3Dplot(log=False)
# ===================================================================================================


# Saving Texture
# ===================================================================================================
def saveTexture():
    phi_arr = phi_texture.astype('float32')

    # Flip the y direction
    phi_arr = np.flip(phi_arr, 0)

    # Write to disk
    imageio.imwrite('phi_r_viewangle.tiff', phi_arr)

saveTexture()
# ===================================================================================================

print('-- DONE! --')