"""Change this

"""

import numpy as np
import matplotlib.pyplot as plt

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from sph_dem.rigid_body.rigid_body_3d import (RigidBody3DScheme,
                                              setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)
from sph_dem.dem import (setup_wall_dem)

from pysph_dem.swelling import (add_swelling_properties_to_rigid_body,
                                get_master_and_slave_swelling_rb_from_combined)
from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def add_user_options(self, group):
        # group.add_argument(
        #     "--re", action="store", type=float, dest="re", default=0.0125,
        #     help="Reynolds number of flow."
        # )

        group.add_argument("--no-of-bodies", action="store", type=int, dest="no_of_bodies",
                           default=1,
                           help="Number of freely moving bodies")

        # group.add_argument(
        #     "--remesh", action="store", type=float, dest="remesh", default=0,
        #     help="Remeshing frequency (setting it to zero disables it)."
        # )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        self.no_of_bodies = self.options.no_of_bodies
        # ======================
        # ======================

        self.rigid_body_rho = 2700.0
        self.hdx = 1.0
        self.dy = 0.1
        self.kn = 1e4
        self.mu = 0.5
        self.en = 0.2
        self.dim = 2
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.rigid_body_diameter = 1.
        self.max_rigid_body_diameter = 5. * self.rigid_body_diameter
        self.dx = self.rigid_body_diameter / 20

        self.dt = 5e-4
        self.tf = 3.

    def create_particles(self):
        from pysph_dem.geometry import create_circle_1
        dx = self.dx
        x1, y1 = create_circle_1(self.rigid_body_diameter, dx)
        x1 = x1.ravel()
        y1 = y1.ravel()
        x = np.array([])
        y = np.array([])
        sign = -1
        for i in range(self.no_of_bodies):
            sign *= -1
            x = np.concatenate((x, x1[:] + i * self.rigid_body_diameter + i * 2. * dx))
            y = np.concatenate((y, y1[:] + sign * self.rigid_body_diameter / 4.))

        m = np.ones_like(x) * dx * dx * self.rigid_body_rho
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='rigid_body', x=x, y=y, h=h, m=m,
                                  rho=self.rigid_body_rho,
                                  rad_s=rad_s,
                                  E=69 * 1e5,
                                  nu=0.3)
        body_id = np.array([])
        dem_id = np.array([])
        for i in range(self.no_of_bodies):
            body_id = np.concatenate((body_id, i * np.ones_like(x1, dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x1, dtype='int')))
        # print(len(body.x))
        # print(len(body_id))
        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)

        # setup the properties
        setup_rigid_body(body, self.dim)

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -1
        for i in range(self.no_of_bodies):
            sign *= -0
            lin_vel = np.concatenate((lin_vel, np.array([sign * 1., 0., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        set_linear_velocity(body, lin_vel)
        set_angular_velocity(body, ang_vel)

        # This is # 3,
        rigid_body_master, rigid_body_slave = get_master_and_slave_swelling_rb_from_combined(
            body, self.max_rigid_body_diameter, self.dx)
        # rigid_body_master, rigid_body_slave = get_master_and_slave_rb(body)
        add_contact_properties_body_master(rigid_body_master, 6, 2)
        rigid_body_master.rad_s[:] = self.rigid_body_diameter / 2.
        rigid_body_master.h[:] = self.rigid_body_diameter * 2.

        # ======================
        # create wall for rigid body
        # ======================
        # left right bottom
        x = np.array([min(rigid_body_slave.x) - 4. * self.rigid_body_diameter,
                      max(rigid_body_slave.x) + 4. * self.rigid_body_diameter,
                      rigid_body_master.x[0]])
        y = np.array([2 * max(rigid_body_slave.y),
                      2 * max(rigid_body_slave.y),
                      min(rigid_body_slave.y) - 2. * self.dx,
                      ])
        normal_x = np.array([1., -1., 0.])
        normal_y = np.array([0., 0., 1.])
        normal_z = np.array([0., 0., 0.])
        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  normal_x=normal_x,
                                  normal_y=normal_y,
                                  normal_z=normal_z,
                                  h=self.rigid_body_diameter/2.,
                                  rho_b=self.rigid_body_rho,
                                  rad_s=self.rigid_body_diameter/2.,
                                  E=69. * 1e9,
                                  nu=0.3,
                                  G=69. * 1e5)
        dem_id = np.array([0, 0, 0])
        wall.add_property('dem_id', type='int', data=dem_id)
        wall.add_constant('no_wall', [3])
        setup_wall_dem(wall)

        return [rigid_body_master, rigid_body_slave, wall]

    def create_scheme(self):
        rb3d_ms = RigidBody3DScheme(rigid_bodies_master=['rigid_body_master'],
                                    rigid_bodies_slave=['rigid_body_slave'],
                                    boundaries=['wall'], dim=2)
        s = SchemeChooser(default='rb3d_ms', rb3d_ms=rb3d_ms)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            gy=self.gy,
            en=self.en,
            dim=self.dim)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=5)
        print("dt = %g"%self.dt)

    def post_step(self, solver):
        from pysph_dem.geometry import create_circle_of_three_layers
        t = solver.t
        dt = solver.dt
        readjust_freq = 100
        for pa in self.particles:
            if pa.name == 'rigid_body_master':
                master = pa

            if pa.name == 'rigid_body_slave':
                slave = pa

        for i in range(len(master.x)):
            if master.rad_s[i] < self.max_rigid_body_diameter / 2. - 2. * self.dx:
                master.rad_s[i] += (self.dx / 1.) / readjust_freq
                master.m[i] = np.pi * master.rad_s[i]**2. * self.rigid_body_rho
                master.m_b[i] = master.m[i]

            if (int(t / dt) % readjust_freq) == 0:
                # re-adjust the SPH particles
                x_new, y_new = create_circle_of_three_layers(master.rad_s[i] * 2. - self.dx / 8.,
                                                             self.dx)
                left_limit = master.body_limits[2 * i]
                slave.dx0[left_limit:left_limit+len(x_new)] = x_new[:]
                slave.dy0[left_limit:left_limit+len(x_new)] = y_new[:]

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        files = files[:]
        t, total_energy = [], []
        x, y = [], []
        R = []
        ang_mom = []
        for sd, body in iter_output(files, 'rigid_body_master'):
            _t = sd['t']
            # print(_t)
            t.append(_t)
            total_energy.append(0.5 * np.sum(body.m[:] * (body.u[:]**2. +
                                                          body.v[:]**2.)))
            R.append(body.R[0])
            # print("========================")
            # print("R is", body.R)
            # # print("ang_mom x is", body.ang_mom_x)
            # # print("ang_mom y is", body.ang_mom_y)
            # print("ang_mom z is", body.ang_mom_z)
            # # print("omega x is", body.omega_x)
            # # print("omega y is", body.omega_y)
            # print("omega z is", body.omega_z)
            # print("moi global master ", body.inertia_tensor_inverse_global_frame)
            # # print("moi body master ", body.inertia_tensor_inverse_body_frame)
            # # print("moi global master ", body.inertia_tensor_global_frame)
            # # print("moi body master ", body.inertia_tensor_body_frame)
            # # x.append(body.xcm[0])
            # # y.append(body.xcm[1])
            # # print(body.ang_mom_z[0])
            ang_mom.append(body.ang_mom_z[0])
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
        plt.plot(t, R, "-", label='R[0]')

        plt.xlabel('t')
        plt.ylabel('ang energy')
        plt.legend()
        fig = os.path.join(self.output_dir, "ang_mom_vs_t.png")
        plt.savefig(fig, dpi=300)
        # plt.show()

        # plt.plot(x, y, label='Simulated')
        # plt.show()


if __name__ == '__main__':
    app = Case0()
    app.run()
    # app.post_process(app.info_filename)
