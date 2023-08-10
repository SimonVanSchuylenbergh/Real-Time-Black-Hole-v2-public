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

def delta_phi_func(b):
    w1 = fsolve(bracket, 1, (M, b))[0]
    
    return 2 * integrate.quad(integrand, 0, w1, (M, b))[0] - np.pi

def dr_dphi(r, b, M = 1):
    return -r**2 * np.sqrt(1 / b**2 - 1 / r**2 * (1 - 2 * M / r))

def dy_dx(w, b):
    w1 = fsolve(bracket, 1, (1, b))[0]
    # Invalid input, just return zero, this will only happen in
    # pixels of the texture that we'll never access.
    if w > w1:
        return 0
    
    r = b / w
    phi = integrate.quad(integrand, 0, w, (1, b))[0]
    
    return (r * np.cos(phi) + np.sin(phi) * dr_dphi(r, b)) / (-r * np.sin(phi) + np.cos(phi) * dr_dphi(r, b))

def calculate_alpha(slope, phi):
    if phi > 3 * np.pi / 2 and slope > 0:
        return (np.arctan(slope) + np.pi) % np.pi + np.pi
    
    return (np.arctan(slope) + np.pi) % np.pi

def mapRange(x, from_low, from_high, to_low, to_high):
    return (x - from_low) / (from_high - from_low) * (to_high - to_low) + to_low

def b_max(r, M):
    return r**2 / np.sqrt(r**2 - 2*r*M)

def check_valid(w, b):
    w1 = fsolve(bracket, 1, (1, b))[0]
    if w > w1:
        return False
    return True

# Calculating Texture
# ===================================================================================================

resolution = 256

M = 1

x = np.linspace(1, resolution, resolution) / resolution # pixel column
y = np.linspace(1, resolution, resolution) / resolution # pixel row

b = 1 / mapRange(x, 0, 1, 0, 1 / 5.1962)
r = 1 / mapRange(y, 0, 1, 0, 1 / 3)

# This is the deflection angle as we calculated it previously
Delta_phi = np.array([delta_phi_func(b[i]) for i in range(len(b))])

# Preparing a 2d array for the distance corrected deflection angles,
# this is what will be saved to an image texture
corrected_delta_phi = np.zeros((len(b), len(r)))

# Filling in the 2d array
# Fill in infinite r, which is just Delta_phi
for j in range(1, len(b)):
    corrected_delta_phi[0, j] = Delta_phi[j]

# Fill in the minimum deflection angle possible for this value of r
for i in range(1, len(r)):
    delta_phi_min = delta_phi_func(b_max(r[i], M)) / 2
    for j in range(len(b)):
        corrected_delta_phi[i, j] = delta_phi_min

# For every combination of b and r, we calculate alpha and save Delta_phi - alpha
for j in range(1, len(r)): # rows
    for i in range(1, len(b)): # columns
        if check_valid(b[j] / r[i], b[j]):
            alpha = calculate_alpha(dy_dx(b[i] / r[j], b[i]), integrate.quad(integrand, 0, b[i] / r[j], (M, b[i]))[0])
        
            corrected_delta_phi[j, i] = Delta_phi[i] - alpha

# ===================================================================================================



# Plotting Texture
# ===================================================================================================

# Creating figure
# -------------------------------------
plt.figure(figsize=(8, 6), facecolor="#1c1c1c")
ax = plt.axes()
ax.set_facecolor("#1c1c1c")
mpl.rcParams.update({'font.size': 15})
# -------------------------------------

# Plotting the 2d array as an image
plt.imshow(np.log10(corrected_delta_phi), origin='lower', cmap='gray')

# Plot layout
# -------------------------------------
cb = plt.colorbar()
cb.ax.yaxis.set_tick_params(color='white')
cb.outline.set_edgecolor('white')
plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

plt.title("$\\log(\\delta\\phi_{def})$", color='white')
ax.set_xlabel("$M/b$") # x -> $b/M$ = 5.1962 / x
ax.set_ylabel("$\\log_{10}(r) / M$") # y -> $\\log_{10}(r)$ = mapRange(y, 0, 1, $\\log_{10}(3)$, 3)

ax.set_xticks([0, 127, 255])
ax.set_xticklabels([0, "", 5.1962])

ax.set_yticks([0, 127, 255])
ax.set_yticklabels(["$\\log_{10}(3)$", "", 3])

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

# ===================================================================================================


# Saving Texture
# ===================================================================================================

arr = corrected_delta_phi.astype('float32')

# Flip the y direction
arr = np.flip(arr, 0)

# Write to disk
imageio.imwrite('Delta_phi.tiff', arr)

# ===================================================================================================
