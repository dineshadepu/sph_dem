import matplotlib.pyplot as plt
import numpy as np
import os

rho_w = 1000.
rad_p = 1.9 * 1e-3
d_p = 2 * rad_p
d_p_0 = 2 * rad_p
rad_p_0 = rad_p
rho_p = 1486.
V_p_0 = 4. / 3. * np.pi * rad_p**3.
V_p = V_p_0
S_p = 4. * np.pi * rad_p**2.
m_p_0 = rho_p * V_p_0
m_p_t = rho_p * V_p_0
m_w_t = 0.
v_s_t = V_p_0
v_s_0 = V_p_0
v_w_t = 0.

c_wl = 1000.
c_wp = 0.
c_max = c_wl
m_w = 0.
D_0 = 4. * 1e-7
decay_delta = 1.81
D_lp = D_0 * np.exp(-decay_delta * c_wp / c_max)
dt = 1e-2
rad_data = []
t_data = []
concentration_data = []
diffusion = []
t = 0.
time_limit = 300 * 60
# time_limit = 300. * dt

steps = 0
print_freq = int(30 / dt)
# print_freq = int(dt / dt)

rad_data.append(d_p / d_p_0)
# rad_data.append(d_p/2.)
t_data.append(t / 60.)
concentration_data.append(c_wp)
diffusion.append(D_lp)

# while c_wp < 999 and t < time_limit:
while t < time_limit:
    D_lp = D_0 * np.exp(-decay_delta * c_wp / c_max)
    S_lp = np.pi * d_p**2.
    m_dot_lp = 2 * S_lp * D_lp * (c_wl - c_wp) / d_p

    # update the mass of the spherical particle
    delta_m = m_dot_lp * dt
    m_w_t += delta_m
    m_p_t = m_p_0 + m_w_t

    # update the volume of the spherical particle
    v_w_t = m_w_t / rho_w
    v_s_t = v_s_0 + v_w_t

    # update the diameter
    rad_p = (3 * v_s_t / (4 * np.pi))**(1/3)
    d_p = 2 * rad_p

    # update the concentration
    c_wp = (m_p_t - m_p_0) / v_s_t

    t = t + dt
    # t_data.append(t/60.)
    # rad_data.append(rad_p / rad_p_0)
    # concentration_data.append(c_wp)
    steps += 1
    # print(t)
    # print(steps)
    # if steps % print_freq == 0:
        # print(c_wp)

    # rad_data.append(rad_p / rad_p_0)
    if steps % print_freq == 0:
        print(steps)
        rad_data.append(d_p / d_p_0)
        # rad_data.append(d_p/ 2.)
        t_data.append(t / 60.)
        concentration_data.append(c_wp)
        diffusion.append(D_lp)

# print(rad_data)
# print(rad_data[1]/ rad_data[0])


data_swelling_dia_ratio_vs_time_exp = np.loadtxt(
    os.path.join('single_particle_swelling_data_experimental.csv'), delimiter=',')
t_exp, diameter_ratio_exp = data_swelling_dia_ratio_vs_time_exp[:, 0], data_swelling_dia_ratio_vs_time_exp[:, 1]

plt.scatter(t_exp, diameter_ratio_exp, label='exp')
plt.scatter(t_data, rad_data, label='current')
# plt.scatter(t_data, concentration_data, label='code')
# plt.scatter(t_data, diffusion, label='code')
plt.legend()
plt.show()
plt.savefig("concentration_vs_time.pdf")
# plt.plot(t_data, rad_data)
# plt.xlabel("Time (minutes)")
# plt.ylabel("d/d_0")
# plt.savefig("radius_vs_time.pdf")
# plt.clf()
