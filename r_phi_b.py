import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
import imageio
from math import isnan


def bracket(w, M, b):
    return 1 - w**2 * (1 - 2 * M / b * w)

def integrand(w, M, b):
    return 1 / np.sqrt(bracket(w, M, b))

def integral(w, M, b):
    return integrate.quad(integrand, 0, w, (M, b))[0]

def integral_r(r, M, b):
    return integrate.quad(dphi_dr, r, np.inf, (b, M))[0]

def turning_point(b, M=1):
    w1 = fsolve(bracket, 1, (M, b))[0]
    return integral(w1, M, b)

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

def get_r_phi(phi, b, r_guess=10, tol=1e-6, max_iter=30, print_warnings=False):
    # KEY POINT: r is a monotonically decreasing function of phi
    #   => if phi(r_guess) < phi we know that r_guess > r
    #   => if phi(r_guess) > phi we know that r_guess < r

    # A guaranteed lower limit on r is 3M
    r_low = 3
    # Still need to find an upper limit
    r_high = np.inf

    # First find a lower and upper limit on r
    # Step 1: find phi(r_guess)
    phi_r_guess = integral_r(r_guess, 1, b)

    # Keeping track of how many times we have calculated phi_r_guess. If this 
    # exceeds max_iter, we stop regardless of whether the requested has been reached.
    n_iter = 1

    # Step 2: if phi_r_guess < phi: we have found an upper and lower limit
    if phi_r_guess < phi:
        r_high = r_guess
    
    # Step 2: if phi_r_guess > phi: we have a new lower limit, but no upper limit yet.
    #         We can find an upper limit on r by doubling r_guess until phi_r_guess > phi
    else:
        r_low = r_guess
        while n_iter <= max_iter:
            r_guess *= 2
            phi_r_guess = integral_r(r_guess, 1, b)
            n_iter += 1
            if phi_r_guess < phi: # We have found our upper limit
                r_high = r_guess
                break
            else:
                r_low = r_guess

    # Step 3: Now we have an upper and lower limit on r.
    #         We take a new r_guess, in the middle of r_low and r_high
    #         We keep halfing our range until the requested tolerance has been reached,
    #         or n_iter exceeds max_iter
    while n_iter <= max_iter:
        # Take new r_guess
        r_guess = (r_high + r_low)/2

        # Calculate phi(r_guess)
        phi_r_guess = integral_r(r_guess, 1, b)
        n_iter += 1

        # Adjust range
        if phi_r_guess > phi or isnan(phi_r_guess):
            r_low = r_guess
        else:
            r_high = r_guess
            
        # Check if tolerance has been reached
        if abs(phi_r_guess - phi) / phi < tol:
            return r_guess
        
    # If we exited the while loop, we have reached the max number of iterations
    # We return our guess, but write a warning message to the console
    if print_warnings:
        print("get_r_phi: max number of iterations reached, tol = " + str(tol)
              + ", relative error = " + str((phi_r_guess - phi) / phi) + ", range = ["
              + str(r_low) + ", " + str(r_high) + "]")
    return (r_high + r_low)/2



# Calculating Texture
# ===================================================================================================

resolution = 512

M = 1

x = np.linspace(0, 1, resolution+2)[1:resolution+1] # pixel column
y = np.linspace(0, 1, resolution+1)[1:] # pixel row

phi = y * np.pi
b = 20 * (1/x - 1)

# Here we keep track of the phi we reach at r=3, for rays with b < sqrt(27).
# Elements in this array corresponding to b > sqrt(27) are not used
# When filling in the texture, we don't go to r smaller than 3
phi_max = np.zeros_like(b)
for i in range(len(b)):
    if b[i] < np.sqrt(27):
        phi_max[i] = integral_r(3, M, b[i])

# Preparing a 2d array, this is what will be saved to an image texture
r_texture = np.zeros((resolution, resolution))


# Filling in texture
for i in range(resolution):
    for j in range(resolution):
        if b[j] > np.sqrt(27):
            if phi[i] < turning_point(b[j]):
                r_texture[i, j] = get_r_phi(phi[i], b[j])
            else:
                r_texture[i, j] = get_r_phi(turning_point(b[j]), b[j])
        else:
            if phi[i] < phi_max[j]:
                r_texture[i, j] = get_r_phi(phi[i], b[j])
            else:
                r_texture[i, j] = 3
    msg = "Completed row "
    for _ in range(len(str(resolution)) - len(str(i+1))):
        msg += " "
    msg += str(i+1) + "/" + str(resolution) + "  "
    for _ in range(3 - len(str(int((i+1)/resolution*100)))):
        msg += " "
    msg += "(" + str(int((i+1)/resolution*100)) + "%)"
    print(msg)


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
        plt.imshow(np.log10(r_texture), origin='lower', cmap='gray')
    else:
        plt.imshow(r_texture, origin='lower', cmap='gray')

    # Plot layout
    # -------------------------------------
    cb = plt.colorbar()
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='white')

    if log:
        plt.title("log r", color='white')
    else:
        plt.title("r", color='white')
    ax.set_xlabel("f(b)")
    ax.set_ylabel("$\\phi$")

    ax.set_xticks([0, int(resolution/2), resolution-1])
    ax.set_xticklabels(["b=$\\infty$", "", "b=0"])

    ax.set_yticks([0, int(resolution/2), resolution-1])
    ax.set_yticklabels(["0", "", "$\\pi$"])

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
        ax.plot_surface(X, Y, np.log10(r_texture), linewidth=2, antialiased=True, alpha=0.5)
        ax.scatter(X, Y, np.log10(r_texture), s=1)
    else:
        ax.plot_surface(X, Y, r_texture, linewidth=2, antialiased=True, alpha=0.5)
        ax.scatter(X, Y, r_texture, s=1)

    # Plot layout
    # -------------------------------------
    ax.set_xlabel("f(b)")
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["b=$\\infty$", "", "b=0"])

    ax.set_ylabel("$\\phi$")
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "", "$\\pi$"])

    if log:
        ax.set_zlabel("log r")
    else:
        ax.set_zlabel("r")
    # -------------------------------------

    plt.show()

#surface3Dplot(log=True)
# ===================================================================================================


# Saving Texture
# ===================================================================================================
def saveTexture(log=True):
    if log:
        r_arr = np.log10(r_texture).astype('float32')
        # Flip the y direction
        r_arr = np.flip(r_arr, 0)
        # Write to disk
        imageio.imwrite('log_r_phi_b.tiff', r_arr)

    else:
        r_arr = r_texture.astype('float32')
        # Flip the y direction
        r_arr = np.flip(r_arr, 0)
        # Write to disk
        imageio.imwrite('r_phi_b.tiff', r_arr)

saveTexture(log=True)
# ===================================================================================================

print('-- DONE! --')