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
from sph_dem.rigid_body.rigid_body_3d import (RigidBody3DScheme,
                                                setup_rigid_body,
                                                set_linear_velocity,
                                                set_angular_velocity,
                                                get_master_and_slave_rb,
                                                add_contact_properties_body_master)
from pysph.examples.solid_mech.impact import add_properties


class Case0(Application):
    def initialize(self):
        self.rho0 = 2700.0
        self.hdx = 1.0
        self.dx = 0.1
        self.dy = 0.1
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.dim = 2

        self.dt = 1e-3
        self.tf = 3.

    def create_particles(self):
        from pysph.tools.geometry import get_2d_block
        dx = self.dx
        x, y = get_2d_block(dx, 1., 1.)
        x = x.flat
        y = y.flat
        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                                  rho=self.rho0,
                                  rad_s=rad_s,
                                  E=69 * 1e9,
                                  nu=0.3)
        body_id = np.zeros(len(x), dtype=int)
        dem_id = np.zeros(len(x), dtype=int)
        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)

        # setup the properties
        setup_rigid_body(body, self.dim)

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        set_linear_velocity(body, np.array([1., 1., 0.]))
        set_angular_velocity(body, np.array([0., 0., 2. * np.pi]))

        body_master, body_slave = get_master_and_slave_rb(body)
        add_contact_properties_body_master(body_master, 6, 3)
        body_master.rad_s[:] = 1. / 2.

        return [body_master, body_slave]

    def create_scheme(self):
        rb3d_ms = RigidBody3DScheme(rigid_bodies_master=['body_master'],
                                    rigid_bodies_slave=['body_slave'],
                                    boundaries=None, dim=self.dim)
        s = SchemeChooser(default='rb3d_ms', rb3d_ms=rb3d_ms)
        return s

    def configure_scheme(self):
        dt = self.dt
        tf = self.tf
        self.scheme.configure_solver(dt=dt, tf=tf, pfreq=10)


if __name__ == '__main__':
    app = Case0()
    app.run()
    app.post_process(app.info_filename)
