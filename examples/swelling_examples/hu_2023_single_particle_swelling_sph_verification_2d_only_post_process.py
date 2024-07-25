from matplotlib import pyplot as plt
import os
from pysph.solver.utils import iter_output, get_files
import numpy as np

directory = "hu_2023_single_particle_swelling_sph_verification_2d_output"
files = get_files(directory)
# print(files)
files = files[::50]
t_simulated, rad_s_simulated = [], []

initial_radius = 1.9 * 1e-3
for sd, body in iter_output(files, 'rigid_body_combined_master'):
    _t = sd['t']
    print(_t)
    t_simulated.append(_t)
    rad_s_simulated.append(body.rad_s[0])

rad_ratio = np.asarray(rad_s_simulated) / initial_radius


# Get the analytical values
rho_w = 1000.
rad_p = 1.9 * 1e-3
d_p = 2 * rad_p
d_p_0 = 2 * rad_p
rad_p_0 = rad_p
rho_p = 1486.
V_p_0 = np.pi * rad_p**2
V_p = V_p_0
S_p = 2. * np.pi * rad_p
m_p_0 = rho_p * V_p_0
m_p_t = rho_p * V_p_0
m_w_t = 0.
v_s_t = V_p_0
v_s_0 = V_p_0
v_w_t = 0.

c_wl = rho_w
c_wp = 0.
c_max = c_wl
m_w = 0.
D_0 = 2. * 1e-7
decay_delta = 1.81
D_lp = D_0 * np.exp(-decay_delta * c_wp / c_max)
dt = 1e-2
rad_data = []
t_data = []
concentration_data = []
diffusion = []
t_current = 0.
# time_limit = 120 * 60
time_limit = 300. * 60

steps = 0
print_freq = int(30 / dt)
# print_freq = int(dt / dt)

rad_data.append(d_p / d_p_0)
# rad_data.append(d_p/2.)
t_data.append(t_current)
concentration_data.append(c_wp)
diffusion.append(D_lp)

# while c_wp < 999 and t < time_limit:
while t_current < t_simulated[-1]:
    D_lp = D_0 * np.exp(-decay_delta * c_wp / c_max)
    S_lp = 2. * np.pi * d_p / 2.
    m_dot_lp = 2 * S_lp * D_lp * (c_wl - c_wp) / d_p

    # update the mass of the spherical particle
    delta_m = m_dot_lp * dt
    m_w_t += delta_m
    m_p_t = m_p_0 + m_w_t

    # update the volume of the spherical particle
    v_w_t = m_w_t / rho_w
    v_s_t = v_s_0 + v_w_t

    # update the diameter
    rad_p = (v_s_t / np.pi)**(1/2)
    d_p = 2 * rad_p

    # update the concentration
    c_wp = (m_p_t - m_p_0) / v_s_t

    t_current = t_current + dt
    # t_data.append(t/60.)
    # rad_data.append(rad_p / rad_p_0)
    # concentration_data.append(c_wp)
    steps += 1
    # print(t)
    # print(steps)
    # if steps % print_freq == 0:
        # print(c_wp)

    # rad_data.append(rad_p / rad_p_0)
    print(steps)
    rad_data.append(d_p / d_p_0)
    # rad_data.append(d_p/ 2.)
    t_data.append(t_current)
    concentration_data.append(c_wp)
    diffusion.append(D_lp)

plt.clf()

plt.plot(t_simulated, rad_ratio, "^", label='Current')
plt.plot(t_data, rad_data, label='Analytical')

plt.xlabel('t (seconds)')
plt.ylabel('Radius ratio')
plt.legend()
fig = os.path.join(directory, "rad_ratio_vs_t.png")
plt.savefig(fig, dpi=300)
plt.show()

# plt.plot(x, y, label='Simulated')
# plt.show()
