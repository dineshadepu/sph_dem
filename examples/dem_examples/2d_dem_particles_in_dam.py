"""In this example we will test our formulation of handling many particles in
contact. We take 5 particles and allow them to collide.

Run the simulation to understand the example better.
"""
import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.examples.solid_mech.impact import add_properties

from pysph.tools.geometry import get_2d_block

from sph_dem.dem import DEMScheme, setup_dem_particles, setup_wall_dem


class B1SSCollideNormalElastic(Application):
    def add_user_options(self, group):
        # group.add_argument(
        #     "--re", action="store", type=float, dest="re", default=0.0125,
        #     help="Reynolds number of flow."
        # )

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=10,
                           help="Number of particles in diamter of a rigid cylinder")

        # group.add_argument(
        #     "--remesh", action="store", type=float, dest="remesh", default=0,
        #     help="Remeshing frequency (setting it to zero disables it)."
        # )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        # self.re = self.options.re
        # ======================
        # ======================

        self.length = 1.
        self.height = 1.
        self.spacing = 0.1
        self.sphere_radius = self.spacing / 2.
        self.sphere_h = 1. * self.sphere_radius * 2.
        self.sphere_rho = 2800
        self.sphere_velocity = 10
        self.en = 1.

        self.h = self.sphere_h

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2
        self.tf = 1.
        self.dt = 2e-4

    def create_particles(self):
        r = self.sphere_radius
        x, y = get_2d_block(dx=self.spacing,
                            length=self.length,
                            height=self.height)
        x[:] += self.length / 2. + self.spacing
        y[:] += self.height / 2. + self.spacing
        x[:] += np.random.rand(len(x)) * self.spacing / 10
        y[:] += np.random.rand(len(y)) * self.spacing / 10
        m_b = 4/3. * np.pi * self.sphere_radius**3. * self.sphere_rho
        _I = 2. / 5. * m_b * self.sphere_radius**2
        I_inverse = 1. / _I
        E = 1e7
        nu = 0.2

        spheres = get_particle_array(name='spheres',
                                     x=x,
                                     y=y,
                                     h=self.h,
                                     m_b=m_b,
                                     I_inverse=I_inverse,
                                     rho_b=self.sphere_rho,
                                     rad_s=self.sphere_radius - self.sphere_radius / 10,
                                     E=E,
                                     nu=nu)
        dem_id = np.ones_like(x, dtype=int) * 0
        spheres.add_property('dem_id', type='int', data=dem_id)

        # create wall
        # [bottom, left, right]
        x = np.array([self.length/2., 0., self.length * 3.])
        y = np.array([0., self.height/2., self.height/2.])
        normal_x = np.array([0., 1., -1.])
        normal_y = np.array([1., 0., 0.])
        normal_z = np.array([0., 0., 0.])
        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  normal_x=normal_x,
                                  normal_y=normal_y,
                                  normal_z=normal_z,
                                  h=self.h,
                                  rho_b=self.sphere_rho,
                                  rad_s=self.sphere_radius,
                                  E=E,
                                  nu=nu,
                                  G=E  # this will be corrected in a functino
                                       # call below (#1)
                                  )
        dem_id = np.array([0, 0, 0])
        wall.add_property('dem_id', type='int', data=dem_id)
        wall.add_constant('no_wall', [len(wall.x)])

        setup_dem_particles(spheres, 8, wall.no_wall[0])
        setup_wall_dem(wall) #  #1)
        print(spheres.max_no_walls, "sphere 1")
        print(wall.no_wall, "wall 1")

        return [spheres, wall]

    def create_scheme(self):
        dem = DEMScheme(
            dem_particles=['spheres'],
            boundaries=['wall'],
            dim=0,
            gx=0.0,
            gy=0.0,
            gz=0.0,
        )
        s = SchemeChooser(default='dem', dem=dem)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            gx=self.gx,
            gy=self.gy,
            gz=self.gz,
           )

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=100)
        print("dt = %g"%self.dt)


if __name__ == '__main__':
    app = B1SSCollideNormalElastic()
    app.run()
    app.post_process(app.info_filename)
