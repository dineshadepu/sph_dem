import matplotlib.pyplot as plt
import numpy as np


rho_w = 1000.
rad_p = 1.9 * 1e-3
rad_p_0 = rad_p
rho_p = 1486.
V_p_0 = 4. / 3. * np.pi * rad_p**3.
V_p = V_p_0
S_p = 4. * np.pi * rad_p**2.
m_p_0 = rho_p * V_p_0

c_w_l = 1000.
c_w_p = 0.
c_max = c_w_l
m_w = 0.
D_0 = 4. * 1e-5
D = D_0
D_decay_fac = 1.81
dt = 1e-2
rad_data = []
t_data = []
concentration_data = []
t = 0.
time_limit = 120 * 60
while c_w_p < 999 and t < time_limit:
    m_dot_lp = S_p * D * (c_w_l - c_w_p) / rad_p
    m_w = m_w + m_dot_lp * dt
    V_w = m_w / rho_w
    V_p = V_p_0 + V_w
    m_p = m_p_0 + m_w

    # Compute the new radius
    rad_p = (3. * V_p / (4 * np.pi))**1/3.
    S_p = 4. * np.pi * rad_p**2.
    c_w_p = (m_p - m_p_0) / V_p
    D = D_0 * np.exp(-D_decay_fac * c_w_p / c_max)

    t = t + dt
    t_data.append(t/60.)
    rad_data.append(rad_p / rad_p_0)
    concentration_data.append(c_w_p)


plt.plot(t_data, rad_data)
plt.xlabel("Time (minutes)")
plt.ylabel("d/d_0")
plt.savefig("radius_vs_time.pdf")
plt.clf()
plt.plot(t_data, concentration_data)
plt.show()
plt.savefig("concentration_vs_time.pdf")
