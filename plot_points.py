"""
import matplotlib.pyplot as plt
import matplotlib
from calc_plot_points import get_solution, calc_tortoise, calc_T_X, calc_p_varphi, nu
import numpy

# --- VECTORIZATION WRAPPERS ---
# This forces functions using scipy.integrate.quad to process arrays element-by-element automatically
calc_tortoise_vectorized = numpy.vectorize(calc_tortoise)
calc_T_X_vectorized = numpy.vectorize(calc_T_X)
# ------------------------------

all_T_lists = []
all_X_lists = []

for r_0 in range(15, 50):
  initial_conditions = [r_0, 0, calc_p_varphi(nu, 1/r_0), 0]
  solution = get_solution(initial_conditions)
  
  # 1. Create a smooth grid of time points across the actual simulation lifetime
  t_smooth = numpy.linspace(0, solution.t[-1], 300)
  
  # 2. Evaluate the continuous solution at these smooth time points
  y_smooth = solution.sol(t_smooth)
  r_smooth = y_smooth[0] # Grab the 'r' vector
  
  # 3. Call the vectorized versions of your transformation functions
  R_smooth, _ = calc_tortoise_vectorized(r_smooth, nu, 15)
  T_smooth, X_smooth = calc_T_X_vectorized(t_smooth, R_smooth)
  
  # 4. Save results for plotting
  all_T_lists.append(T_smooth)
  all_X_lists.append(X_smooth)

# --- PLOTTING ---
plt.figure(figsize=(6, 8))
ax = plt.gca()

for idx in range(len(all_T_lists)):
  T_arr = all_T_lists[idx]
  X_arr = all_X_lists[idx]
    
  label_name = f'Initial r_0 = {15 + idx}' if idx == 0 else None
  plt.plot(X_arr, T_arr, color='blue', alpha=0.7, label=label_name)


# Define the 4 corners of your Penrose diamond
points = [
    (-numpy.pi/2, 0),       # Left (i^0)
    (0, numpy.pi/2),        # Top (i^+)
    (numpy.pi/2, 0),        # Right (i^0)
    (0, -numpy.pi/2)        # Bottom (i^-)
]

# Create the polygon
# fill=False ensures it's just the black outline
diamond = matplotlib.patches.Polygon(
    points, 
    closed=True, 
    fill=False, 
    edgecolor='black', 
    linewidth=2
)
ax.add_patch(diamond)


plt.xlabel('X')
plt.ylabel('T')
plt.title('Penrose Diagram: EOB Worldlines')
plt.legend()
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
"""


import matplotlib.pyplot as plt
from calc_plot_points import get_solution, calc_tortoise, calc_T_X, calc_p_varphi, nu
import numpy, scipy, matplotlib

all_T_lists = []
all_X_lists = []
start, end = 12, 50
mid = (start + end)/2
nu_range = mid + (end-mid) * numpy.sin(numpy.linspace(-numpy.pi / 2, numpy.pi / 2, 50))**3
# nu_range = mid + (end-mid) * numpy.linspace(-1, 1, 33)**3
# for r_0 in numpy.arange(15, 50, 0.25):
# for r_0 in numpy.arange(4, 20):
for r_0 in nu_range:
# for r_0 in numpy.arange(start, end):
  # plot_coords_list = []
  T_list, X_list = [], []
  # initial_conditions = [r_0, varphi_0, p_varphi_0, p_r_0]
  # initial_conditions = [15, 0, calc_p_varphi(nu, 1/15), 0]
  initial_conditions = [r_0, 0, calc_p_varphi(nu, 1/r_0), 0]
  solution = get_solution(initial_conditions)
  for i in range(len(solution.t)):
    R, _ = calc_tortoise(solution.y[0][i], nu, mid)
    # R, _ = calc_tortoise(solution.y[0][i], nu, start)
    # plot_coords_list.append(calc_T_X(solution.t[i], R))
    T, X = calc_T_X(solution.t[i], R)
    T_list.append(T)
    X_list.append(X)
  all_T_lists.append(T_list)
  all_X_lists.append(X_list)

# T, X = zip(*plot_coords_list)

plt.figure(figsize=(6, 8))
ax = plt.gca()

for idx in range(len(all_T_lists)):
  T_arr = numpy.array(all_T_lists[idx])
  X_arr = numpy.array(all_X_lists[idx])
    
  # --- INTERPOLATION STEP ---
  # We create a parametric variable 's' from 0 to 1 based on the index.
  # This is much safer than interpolating T vs X directly, as it handles 
  # any curve shape without throwing "duplicate x values" errors.
  s = numpy.linspace(0, 1, len(T_arr))  
  s_smooth = numpy.linspace(0, 1, 300) # Generate 300 smooth points
    
    # k=3 means cubic spline interpolation
  T_smooth = scipy.interpolate.make_interp_spline(s, T_arr, k=3)(s_smooth)
  X_smooth = scipy.interpolate.make_interp_spline(s, X_arr, k=3)(s_smooth)
    # --------------------------
    
    # Plot the smoothed line (only add to legend on the first iteration)
  label_name = f'Initial r_0 = {start + idx}' if idx == 0 else None
  # plt.plot(X_smooth, T_smooth, color='blue', alpha=0.7, label=label_name)
  plt.plot(all_X_lists[idx], all_T_lists[idx], color='blue', alpha=0.7, label=label_name)
  plt.plot(all_X_lists[idx], numpy.multiply(-1, all_T_lists[idx]), color='blue', alpha=0.7)

# Define the 4 corners of your Penrose diamond
points = [
    (-numpy.pi/2, 0),       # Left (i^0)
    (0, numpy.pi/2),        # Top (i^+)
    (numpy.pi/2, 0),        # Right (i^0)
    (0, -numpy.pi/2)        # Bottom (i^-)
]

# Create the polygon
# fill=False ensures it's just the black outline
diamond = matplotlib.patches.Polygon(
    points, 
    closed=True, 
    fill=False, 
    edgecolor='black', 
    linewidth=2
)
ax.add_patch(diamond)

# PENROSE DIAGRAM LABELS
text_kwargs = dict(fontsize=12, zorder=5, va='center', ha='center')

# i+-0
plt.text(0, numpy.pi/2 + 0.08, r'$i^+$', **text_kwargs)        # Future Timelike Infinity
plt.text(0, -numpy.pi/2 - 0.08, r'$i^-$', **text_kwargs)       # Past Timelike Infinity
plt.text(numpy.pi/2 + 0.08, 0, r'$i^0$', **text_kwargs)        # Spatial Infinity (r -> oo)

# SCRI
plt.text(numpy.pi/4 + 0.12, numpy.pi/4 + 0.05, r'$\mathcal{I}^+$', **text_kwargs)  # Future Null Infinity
plt.text(numpy.pi/4 + 0.12, -numpy.pi/4 - 0.05, r'$\mathcal{I}^-$', **text_kwargs) # Past Null Infinity

# event horizon labels
plt.text(-numpy.pi/4 - 0.18, numpy.pi/4 + 0.02, 'Future Event Horizon', fontsize=10, color='darkred', va='center', ha='center', rotation=45)
plt.text(-numpy.pi/4 - 0.18, -numpy.pi/4 - 0.02, 'Past Event Horizon', fontsize=10, color='darkred', va='center', ha='center', rotation=-45)

plt.xlabel('X')
plt.ylabel('T')
plt.title('Penrose Diagram: EOB Worldlines')
plt.legend()
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

"""
plt.figure(figsize=(6, 8))
plt.plot(X, T, color='blue', label='worldline')
plt.xlabel('X')
plt.ylabel('T')
plt.title('Penrose diagram')
plt.legend()
plt.axis('equal')
plt.show()
"""
"""
# clip to diamond (removes any points outside the boundary)
mask = (np.abs(X_wl) + np.abs(T_wl)) <= half * 1.01

ax.plot(X_wl[mask], T_wl[mask],
        color='#00ffcc', linewidth=2.2, zorder=7, label='worldline')

# start dot
ax.scatter([X_wl[mask][0]], [T_wl[mask][0]],
           color='#00ffcc', s=40, zorder=8)
ax.text(X_wl[mask][0]+0.04, T_wl[mask][0]-0.04, '$r_0=15$',
        color='#00ffcc', fontsize=8, zorder=8)
"""

"""


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
T_wl = np.array(T_list)
X_wl = np.array(X_list)

# ── figure setup ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), facecolor='#0d0d0d')
ax.set_facecolor('#0d0d0d')
 
half = np.pi / 2
 
# ── helper: diamond boundary ─────────────────────────────────────────────────
diamond_x = [0, half, 0, -half, 0]
diamond_y = [-half, 0, half, 0, -half]
 
# shaded regions ─────────────────────────────────────────────────────────────
# exterior (right wedge)
ext_x = [0, half, 0]
ext_y = [0,  0,  half]  # wrong, fix below
# exterior = right diamond quadrant
ext = plt.Polygon([[0, -half], [half, 0], [0, half]],
                  closed=True, facecolor='#1a2a3a', edgecolor='none', zorder=1)
ax.add_patch(ext)
 
# interior / black hole (top wedge)
interior = plt.Polygon([[0, half], [-half, 0], [half, 0]],
                       closed=True, facecolor='#1a0d0d', edgecolor='none', zorder=1)
ax.add_patch(interior)
 
# left wedge (other exterior)
left = plt.Polygon([[0, -half], [-half, 0], [0, half]],
                   closed=True, facecolor='#1a2a3a', edgecolor='none', alpha=0.5, zorder=1)
ax.add_patch(left)
 
# bottom wedge (white hole)
bottom = plt.Polygon([[0, -half], [half, 0], [-half, 0]],
                     closed=True, facecolor='#0d1a0d', edgecolor='none', zorder=1)
ax.add_patch(bottom)
 
# ── singularity (top edge) ───────────────────────────────────────────────────
ax.plot([-half, half], [half, half],
        color='#ff4444', linewidth=3, zorder=5, solid_capstyle='round')
ax.fill_between([-half, half], [half, half], [half+0.04, half+0.04],
                color='#ff4444', alpha=0.3, zorder=4)
ax.text(0, half + 0.06, 'singularity  $r = 0$',
        color='#ff6666', fontsize=9, ha='center', va='bottom',
        fontfamily='monospace', zorder=6)
 
# ── horizon lines ────────────────────────────────────────────────────────────
# future horizon
ax.plot([0, half], [0, half],
        color='#ffcc00', linewidth=1.8, linestyle='--', zorder=4, alpha=0.9)
# past horizon
ax.plot([0, half], [0, -half],
        color='#ffcc00', linewidth=1.8, linestyle='--', zorder=4, alpha=0.9)
ax.plot([0, -half], [0, half],
        color='#ffcc00', linewidth=1.8, linestyle='--', zorder=4, alpha=0.9)
ax.plot([0, -half], [0, -half],
        color='#ffcc00', linewidth=1.8, linestyle='--', zorder=4, alpha=0.9)
 
# horizon label
ax.text(half*0.55, half*0.45, '$\\mathcal{H}^+$',
        color='#ffcc00', fontsize=11, ha='left', zorder=6)
 
# ── diamond outline ───────────────────────────────────────────────────────────
ax.plot(diamond_x, diamond_y, color='#888888', linewidth=1.2, zorder=5)
 
# ── null infinity labels ──────────────────────────────────────────────────────
ax.text(half*0.78, half*0.55,  '$\\mathcal{I}^+$',
        color='#aaddff', fontsize=13, ha='center', zorder=6)
ax.text(half*0.78, -half*0.55, '$\\mathcal{I}^-$',
        color='#aaddff', fontsize=13, ha='center', zorder=6)
ax.text(-half*0.85, half*0.55,  '$\\mathcal{I}^+$',
        color='#aaddff', fontsize=13, ha='center', zorder=6, alpha=0.4)
ax.text(-half*0.85, -half*0.55, '$\\mathcal{I}^-$',
        color='#aaddff', fontsize=13, ha='center', zorder=6, alpha=0.4)
 
# region labels
ax.text(half*0.62, 0,   'I\nexterior',
        color='#88bbdd', fontsize=8.5, ha='center', va='center',
        fontfamily='monospace', zorder=6)
ax.text(0, half*0.62,   'II\nblack hole',
        color='#dd8888', fontsize=8.5, ha='center', va='center',
        fontfamily='monospace', zorder=6)
ax.text(-half*0.62, 0,  'III\nexterior',
        color='#88bbdd', fontsize=8.5, ha='center', va='center',
        fontfamily='monospace', alpha=0.5, zorder=6)
ax.text(0, -half*0.62,  'IV\nwhite hole',
        color='#88dd88', fontsize=8.5, ha='center', va='center',
        fontfamily='monospace', zorder=6)

# ── worldline placeholder ─────────────────────────────────────────────────────
# replace these with your actual T, X arrays from the solution
t_vals = np.linspace(0, 1, 300)
# example curve: starts in exterior, crosses horizon, hits singularity
X_wl = half * (0.6 - 0.6*t_vals**0.4)
T_wl = -half*0.6 + (half + half*0.6) * t_vals**0.7
# clip to diamond
mask = (np.abs(X_wl) + np.abs(T_wl)) <= half * 1.01
ax.plot(X_wl[mask], T_wl[mask],
        color='#00ffcc', linewidth=2.2, zorder=7, label='worldline')

# clip to diamond (removes any points outside the boundary)
mask = (np.abs(X_wl) + np.abs(T_wl)) <= half * 1.01

ax.plot(X_wl[mask], T_wl[mask],
        color='#00ffcc', linewidth=2.2, zorder=7, label='worldline')

# start dot
ax.scatter([X_wl[mask][0]], [T_wl[mask][0]],
           color='#00ffcc', s=40, zorder=8)
ax.text(X_wl[mask][0]+0.04, T_wl[mask][0]-0.04, '$r_0=15$',
        color='#00ffcc', fontsize=8, zorder=8)
 
# start dot
ax.scatter([X_wl[0]], [T_wl[0]],
           color='#00ffcc', s=40, zorder=8)
ax.text(X_wl[0]+0.04, T_wl[0]-0.04, '$r_0=15$',
        color='#00ffcc', fontsize=8, zorder=8)
 
# ── axes / frame ──────────────────────────────────────────────────────────────
ax.set_xlim(-half - 0.15, half + 0.15)
ax.set_ylim(-half - 0.15, half + 0.25)
ax.set_aspect('equal')
ax.axis('off')
 
ax.set_title('Penrose–Carter diagram\nEOB binary black hole  ($\\nu = 1/4$)',
             color='#cccccc', fontsize=11, pad=12, fontfamily='monospace')
 
# legend
wl_patch = mpatches.Patch(color='#00ffcc', label='EOB worldline')
hor_patch = mpatches.Patch(color='#ffcc00', label='event horizon $\\mathcal{H}^\\pm$')
sing_patch = mpatches.Patch(color='#ff4444', label='singularity $r=0$')
ax.legend(handles=[wl_patch, hor_patch, sing_patch],
          loc='lower right', fontsize=8,
          facecolor='#1a1a1a', edgecolor='#444444',
          labelcolor='white', framealpha=0.8)
 
plt.tight_layout()
plt.savefig('penrose_diagram.png', dpi=150,
            bbox_inches='tight', facecolor='#0d0d0d')
plt.show()
print("saved")
 """