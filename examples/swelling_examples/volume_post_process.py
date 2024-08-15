from matplotlib import pyplot as plt
import os
from pysph.solver.utils import iter_output, get_files
from pysph.solver.utils import load
import numpy as np

directory = "test_add_radial_velocity_equation_with_different_swell_amounts_output"
files = get_files(directory)
# print(files)
files = files[::5]
t, total_energy = [], []
x, y = [], []
R = []
ang_mom = []

fluid_volume_increase = []
solid_volume_increase = []

rigid_body_diameter = 1.9 * 1e-3
dx = rigid_body_diameter / 6
# fluid_length = 3 * 1.5 * self.rigid_body_diameter
# fluid_height = 3 * self.rigid_body_diameter

fname = files[0]
data = load(fname)
fluid = data['arrays']['fluid']
fluid_length = max(fluid.x) - min(fluid.x)
fluid_height = max(fluid.y) - min(fluid.y)
fluid_volume_0 = fluid_height * fluid_length

fname = files[0]
data = load(fname)
body = data['arrays']['rigid_body_combined_master']
solid_volume_t_0 = np.pi * (body.rad_s[0])**2.
print("solid volume is", solid_volume_t_0)

for sd, body, fluid in iter_output(files, 'rigid_body_combined_master', 'fluid'):
    _t = sd['t']
    # print(_t)
    t.append(_t)
    rad_s_t_later = body.rad_s[0]
    solid_volume_t_later = np.pi * rad_s_t_later**2.
    fluid_y_later = np.max(fluid.y)
    fluid_volume_later = fluid_y_later * fluid_length

    solid_volume_increase_ = solid_volume_t_later - solid_volume_t_0
    # print(solid_volume_t_later)
    # print("solids volume increase", solid_volume_increase)

    fluid_volume_increase_ = fluid_volume_later - fluid_volume_0 - dx / 2. * fluid_length

    fluid_volume_increase.append(fluid_volume_increase_)
    solid_volume_increase.append(solid_volume_increase_)

# print(ang_mom)

import matplotlib
import os
# matplotlib.use('Agg')

from matplotlib import pyplot as plt

# res = os.path.join(self.output_dir, "results.npz")
# np.savez(res, t=t, amplitude=amplitude)

# gtvf data
# data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
# t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

plt.clf()

# plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
# plt.plot(t, total_energy, "-", label='Simulated')
# plt.plot(t, ang_mom, "-", label='Angular momentum')
plt.plot(t, fluid_volume_increase, "-", label='Fluid volume increase')
plt.plot(t, solid_volume_increase, "*", label='Solid volume increase')

plt.xlabel('t')
plt.ylabel('Volume increase (m^3)')
plt.legend()
fig = os.path.join("t_vs_volume.pdf")
plt.savefig(fig, dpi=300)
# plt.show()

# plt.plot(x, y, label='Simulated')
# plt.show()
