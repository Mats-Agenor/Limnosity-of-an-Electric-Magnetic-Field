import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib as mpl
import os

# ======================
# STYLE CONFIGURATIONS
# ======================
# These settings control the appearance of all matplotlib plots
plt.rcParams["axes.labelsize"] = 20       # Font size for axis labels
plt.rcParams["xtick.labelsize"] = 18      # Font size for x-axis ticks
plt.rcParams["ytick.labelsize"] = 18      # Font size for y-axis ticks
plt.rcParams['font.size'] = 20            # Base font size
plt.rc('font', **{'family':'serif', 'serif':['Times']})  # Font family
mpl.rcParams['figure.dpi'] = 100          # Figure resolution
# mpl.rcParams['text.usetex'] = True      # Uncomment to use LaTeX rendering
mpl.rcParams['legend.frameon'] = False    # No frame around legend
mpl.rcParams['font.family'] = 'STIXGeneral'  # Math font family
mpl.rcParams['mathtext.fontset'] = 'stix' # Math font style
mpl.rcParams['xtick.direction'] = 'in'    # X ticks inside the plot
mpl.rcParams['ytick.direction'] = 'in'    # Y ticks inside the plot
mpl.rcParams['xtick.top'] = True          # Show ticks on top axis
mpl.rcParams['ytick.right'] = True        # Show ticks on right axis
mpl.rcParams['xtick.major.size'] = 5      # Length of major x ticks
mpl.rcParams['xtick.minor.size'] = 3      # Length of minor x ticks
mpl.rcParams['ytick.major.size'] = 5      # Length of major y ticks
mpl.rcParams['ytick.minor.size'] = 3      # Length of minor y ticks
mpl.rcParams['xtick.major.width'] = 0.79  # Width of major x ticks
mpl.rcParams['xtick.minor.width'] = 0.79  # Width of minor x ticks
mpl.rcParams['ytick.major.width'] = 0.79  # Width of major y ticks
mpl.rcParams['ytick.minor.width'] = 0.79  # Width of minor y ticks
plt.rcParams['figure.constrained_layout.use'] = True  # Better layout management

# =============================================
# 1. Load Data from trajectories_fields.dat
# =============================================
def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        current_time_step = []
        for line in f:
            if line.strip():  # Non-empty line
                parts = list(map(float, line.split()))
                x, y, z, q, Ex, Ey, Ez, Bx, By, Bz = parts
                current_time_step.append({
                    'position': np.array([x, y, z]),
                    'charge': q,
                    'E_field': np.array([Ex, Ey, Ez]),
                    'B_field': np.array([Bx, By, Bz])
                })
            else:  # Empty line indicates new time step
                if current_time_step:
                    data.append(current_time_step)
                    current_time_step = []
        if current_time_step:  # Add last time step if file doesn't end with empty line
            data.append(current_time_step)
    return np.array(data)

# Load the data
print("Loading data...")
simulation_data = load_data('trajectories_fields.dat')
n_timesteps = len(simulation_data)
n_particles = len(simulation_data[0])
print(f"Loaded {n_timesteps} timesteps with {n_particles} particles each")

# Simulation parameters
dt = 1e-6  # Time step
L = 1.0    # Box size
times = np.arange(n_timesteps) * dt

# =============================================
# 2. Precompute Luminosity Maps for Animation
# =============================================
print("\nPrecomputing luminosity maps for animation...")

# Parameters for the luminosity map
grid_size = 128  # Reduced for faster computation
sigma = L / 50
intensity_scale = 1e-20

# Create grid
xx, yy = np.meshgrid(np.linspace(0, L, grid_size), 
                     np.linspace(0, L, grid_size))

# We'll animate every nth frame to make it manageable
animation_stride = max(1, n_timesteps // 100)  # Aim for ~100 frames
animation_frames = range(0, n_timesteps, animation_stride)

# Precompute all luminosity maps
luminosity_maps = []
for t in tqdm(animation_frames, desc="Computing frames"):
    current_map = np.zeros((grid_size, grid_size))
    
    for particle in simulation_data[t]:
        x, y, _ = particle['position']
        intensity = np.linalg.norm(particle['E_field']) * intensity_scale
        dist_sq = (xx - x)**2 + (yy - y)**2
        current_map += intensity * np.exp(-dist_sq / (2 * sigma**2))
    
    # Apply smoothing and store
    luminosity_maps.append(gaussian_filter(current_map, sigma=1))

# Find global min/max for consistent color scaling
vmin = np.min([np.min(m[m > 0]) for m in luminosity_maps])  # smallest non-zero value
vmax = np.max([np.max(m) for m in luminosity_maps])

# =============================================
# 3. Create the Animation
# =============================================
print("\nCreating animation...")

# Set up the figure
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(luminosity_maps[0], extent=[0, L, 0, L],
               origin='lower', cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Luminosity (arb. units)')
ax.set_xlabel('x position (m)')
ax.set_ylabel('y position (m)')
title = ax.set_title(f'Luminosity Map at t = {times[0]:.2e} s')

# Animation update function
def update(frame):
    idx = frame  # Since we're using a subset of frames
    im.set_array(luminosity_maps[idx])
    title.set_text(f'Luminosity Map at t = {times[animation_frames[idx]]:.2e} s')
    return im, title

# Create the animation
ani = FuncAnimation(fig, update, frames=len(luminosity_maps),
                    interval=100, blit=True)

# =============================================
# 4. Save the Animation
# =============================================
output_file = 'luminosity_evolution.mp4'
print(f"\nSaving animation to {output_file}...")

# Ensure the ffmpeg writer is available
try:
    Writer = plt.get_backend().animation.writers['ffmpeg']
except:
    from matplotlib.animation import FFMpegWriter
    Writer = FFMpegWriter

writer = Writer(fps=15, bitrate=1800)
ani.save(output_file, writer=writer, dpi=150)

print(f"Animation successfully saved to {output_file}")
print(f"File size: {os.path.getsize(output_file)/1e6:.1f} MB")

plt.close()
