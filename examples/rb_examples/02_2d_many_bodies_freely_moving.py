"""A cube translating and rotating freely without the influence of gravity.
This is used to test the rigid body dynamics equations.
"""

import numpy as np

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser
from pysph_dem.rigid_body.rigid_body_3d import (RigidBody3DScheme,
                                                setup_rigid_body,
                                                set_linear_velocity,
                                                set_angular_velocity,
                                                get_master_and_slave_rb,
                                                add_contact_properties_body_master)
from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def add_user_options(self, group):
        # group.add_argument(
        #     "--re", action="store", type=float, dest="re", default=0.0125,
        #     help="Reynolds number of flow."
        # )

        group.add_argument("--no-of-bodies", action="store", type=int, dest="no_of_bodies",
                           default=4,
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

        self.rho0 = 2700.0
        self.hdx = 1.0
        self.dx = 0.1
        self.dy = 0.1
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

        self.dt = 1e-3
        self.tf = 5.

    def create_particles(self):
        from pysph.tools.geometry import get_2d_block
        dx = self.dx
        x1, y1 = get_2d_block(dx, 1., 1.)
        x1 = x1.flat
        y1 = y1.flat
        x = np.array([])
        y = np.array([])
        for i in range(self.no_of_bodies):
            x = np.concatenate((x, x1[:] + 2. * i))

        for i in range(self.no_of_bodies):
            y = np.concatenate((y, y1))

        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                                  rho=self.rho0,
                                  rad_s=rad_s,
                                  E=69 * 1e9,
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
        for i in range(self.no_of_bodies):
            lin_vel = np.concatenate((lin_vel, np.array([1., 1., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 2. * np.pi])))

        set_linear_velocity(body, lin_vel)
        set_angular_velocity(body, ang_vel)

        body_master, body_slave = get_master_and_slave_rb(body)
        add_contact_properties_body_master(body_master, 6, 3)
        body_master.rad_s[:] = 1. / 2.
        print("Body id is", body_slave.body_id)
        print("Body limits is", body_master.body_limits)

        return [body_master, body_slave]

    def create_scheme(self):
        rb3d_ms = RigidBody3DScheme(rigid_bodies_master=['body_master'],
                                    rigid_bodies_slave=['body_slave'],
                                    boundaries=None, dim=2)
        s = SchemeChooser(default='rb3d_ms', rb3d_ms=rb3d_ms)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=30)
        print("dt = %g"%self.dt)


if __name__ == '__main__':
    app = Case0()
    app.run()
    app.post_process(app.info_filename)
