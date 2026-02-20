import numpy as np
import matplotlib.pyplot as plt

# --- Physical Constants ---
g = 9.81           # Gravity (m/s^2)
sigma = 2.1        # Conductivity (S/m)
V_rail = 630       # Voltage (V)
rho_fluid = 1020   # Density of urine (kg/m^3) approx

# --- Initial Conditions ---
h0 = 1.0           # Source height (m)
theta_deg = 0      # Launch angle (degrees, 0 = horizontal)
theta = np.radians(theta_deg)
v0 = 2.8           # Initial velocity (m/s)
d0 = 0.005         # Initial diameter (m)

# Derived flow constants
A0 = np.pi * (d0/2)**2
Q0 = A0 * v0
vx = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)

# --- Time of Flight Calculation ---
# Solve y(T) = 0 => h0 + vy0*T - 0.5*g*T^2 = 0
# 0.5*g*T^2 - vy0*T - h0 = 0
discriminant = vy0**2 - 4*(0.5*g)*(-h0)
T_impact = (vy0 + np.sqrt(discriminant)) / g

# --- Trajectory Arrays ---
t = np.linspace(0, T_impact, 200)
x = vx * t
y = h0 + vy0*t - 0.5*g*t**2

# Velocity magnitude v(t) = sqrt(vx^2 + (vy0 - gt)^2)
vy_t = vy0 - g*t
v_t = np.sqrt(vx**2 + vy_t**2)

# --- Resistance Integration (The "Curvilinear Integral") ---
# dR = (v(t)^2 / (sigma * Q0)) * dt
dR_dt = (v_t**2) / (sigma * Q0)
R_accumulated = np.cumsum(dR_dt) * (t[1] - t[0]) # Numerical integration
I_profile = V_rail / R_accumulated

# --- Instability Analysis (Plateau-Rayleigh) ---
# Weber Number calculation
gamma = 0.060 # Surface tension of urine (N/m) approx slightly < water
We = (rho_fluid * v0**2 * d0) / gamma
# Breakup length L ~ 20 * d0 * sqrt(We) (Turbulent jet correlation)
L_breakup = 20 * d0 * np.sqrt(We) 
# Find time when arc length > L_breakup
arc_length = np.cumsum(v_t) * (t[1] - t[0])
breakup_index = np.searchsorted(arc_length, 0.25) # Conservative 25cm limit

# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: The Trajectory Arc
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, 'b-', label='Fluid Stream Trajectory $\\vec{r}(t)$')
ax.plot(x[breakup_index:], y[breakup_index:], 'r--', linewidth=2, label='Droplet Breakup Zone (Non-Conductive)')
ax.scatter([x[-1]], [0], color='black', s=100, label='Rail Contact', zorder=5)
ax.fill_between(x, 0, -0.2, color='gray', alpha=0.5)
ax.set_ylim(-0.1, h0 + 0.2)
ax.set_aspect('equal')
ax.set_xlabel('Horizontal Distance $x$ (m)')
ax.set_ylabel('Vertical Height $y$ (m)')
ax.set_title('Projectile Trajectory of the Electrolytic Stream')
ax.legend()
plt.tight_layout()
plt.savefig('trajectory_arc.png', dpi=300)

# Plot 2: Current vs Arc Length
plt.figure(figsize=(8, 5))
valid_arc = arc_length[:breakup_index]
valid_I = I_profile[:breakup_index] * 1000 # to mA
full_I = I_profile * 1000

plt.plot(arc_length, full_I, 'k:', alpha=0.5, label='Theoretical Continuous Current')
plt.plot(valid_arc, valid_I, 'r-', linewidth=2, label='Realistic Current (Before Breakup)')
plt.axhline(50, color='orange', linestyle='--', label='Lethal Threshold (50mA)')

# Cap the y-axis because R->0 at t=0 implies I->inf
plt.ylim(0, 500)
plt.xlabel('Stream Arc Length $s$ (m)')
plt.ylabel('Current $I$ (mA)')
plt.title('Current vs. Stream Length (Curvilinear Integration)')
plt.legend()
plt.tight_layout()
plt.savefig('current_integration.png', dpi=300)

print(f"Total Resistance at impact: {R_accumulated[-1]:.2f} Ohms")
print(f"Theoretical Max Current: {630/R_accumulated[-1]*1000:.2f} mA")