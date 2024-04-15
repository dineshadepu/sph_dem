"""A 2D fluid dam break

Run it using:

python 2d_dam_break.py --openmp --tf 1 --alpha 0.1 --tank-length-ratio 3.
"""
import numpy as np

# imports related to the application class
from pysph.solver.application import Application
from pysph.sph.scheme import (SchemeChooser,
                              add_bool_argument)

# geometry imports
import pysph.tools.geometry as G
from sph_dem.geometry import (
    hydrostatic_tank_2d,
    translate_system_with_left_corner_as_origin,
    create_tank_2d_from_block_2d,
    get_2d_block)

from sph_dem.particle_shifting.fluid import (FluidsScheme,
                                             get_particle_array_fluid,
                                             get_particle_array_boundary)


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=4,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=8,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=1,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=1.4,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=20,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=1500,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=1e-3,
                           help="Diameter of each particle")

        group.add_argument("--no-of-bodies", action="store", type=int,
                           dest="no_of_bodies", default=1,
                           help="Total no of rigid bodies")

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        # self.re = self.options.re
        # ======================
        # ======================

        # ======================
        # Dimensions
        # ======================
        # rigid body dimensions
        # All the particles are in circular or spherical shape
        self.rigid_body_diameter = 0.11
        self.rigid_body_velocity = 0.
        self.no_of_bodies = self.options.no_of_bodies

        # x - axis
        self.fluid_length = self.options.fluid_length_ratio * self.rigid_body_diameter
        # y - axis
        self.fluid_height = self.options.fluid_height_ratio * self.rigid_body_diameter
        # z - axis
        self.fluid_depth = 0.

        # x - axis
        self.tank_length = self.options.tank_length_ratio * self.fluid_length
        # y - axis
        self.tank_height = self.options.tank_height_ratio * self.fluid_height
        # z - axis
        self.tank_depth = 0.0

        self.tank_layers = 3

        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.fluid_nu = 0.0

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        self.N = self.options.N
        self.dx = self.fluid_length / self.N
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = self.fluid_nu
        self.tf = 1.0
        # self.tf = 0.56 - 0.3192
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.00

        self.gy = 0.0

        # Setup default parameters.
        dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 1e5
        if self.nu > 1e-12:
            dt_viscous = 0.125 * self.h**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h/(np.abs(self.gy)))
        print("dt_cfl", dt_cfl, "dt_viscous", dt_viscous, "dt_force", dt_force)

        self.dt = min(dt_cfl, dt_force)
        print("Computed stable dt is: ", self.dt)
        # self.dt = 1e-4
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_fluid_and_tank_particle_arrays(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                             self.tank_height, self.tank_layers,
                                             self.dx, self.dx, False)
        if self.options.tank_length_ratio > 1.:
            xt, yt = create_tank_2d_from_block_2d(xf, yf,
                                                  self.tank_length,
                                                  self.tank_height,
                                                  self.dx,
                                                  self.tank_layers)

        zt = np.zeros_like(xt)
        zf = np.zeros_like(xf)

        # move fluid such that the left corner is at the origin of the
        # co-ordinate system
        translation = translate_system_with_left_corner_as_origin(xf, yf, zf)
        xt[:] = xt[:] - translation[0]
        yt[:] = yt[:] - translation[1]
        zt[:] = zt[:] - translation[2]

        # xf, yf, zf = np.array([0.02]), np.array([self.fluid_height]), np.array([0.])

        m = self.dx**self.dim * self.fluid_rho
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=self.h, m=m, rho=self.fluid_rho)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=self.h, m=m, rho=self.fluid_rho)

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_rho * self.gy * (max(fluid.y) - fluid.y[:])
        fluid.c0_ref[0] = self.c0
        fluid.p0_ref[0] = self.p0
        return fluid, tank

    def create_particles(self):
        from pysph.base.kernels import QuinticSpline
        # This will create full particle array required for the scheme
        fluid, tank = self.create_fluid_and_tank_particle_arrays()

        kernel = QuinticSpline(dim=self.dim)
        fluid.add_constant('wdeltap', -1.)
        wdeltap = kernel.kernel(rij=self.dx, h=self.h)
        fluid.wdeltap[0] = wdeltap

        # create a solid boundary block
        xb, yb = get_2d_block(self.dx, 8. * self.dx, 8. * self.dx)
        m = self.dx**self.dim * self.fluid_rho
        moving_boundary = get_particle_array_boundary(name='moving_boundary',
                                                      x=xb, y=yb, h=self.h, m=m,
                                                      rho=self.fluid_rho)
        moving_boundary.x[:] -= min(moving_boundary.x) - min(fluid.x)
        moving_boundary.x[:] += self.fluid_length / 2.
        moving_boundary.x[:] -= 4. * self.dx
        moving_boundary.y[:] -= min(moving_boundary.y) - min(fluid.y)
        moving_boundary.y[:] += self.fluid_height / 8.
        G.remove_overlap_particles(
            fluid, moving_boundary, self.dx, dim=self.dim
        )
        # moving_boundary.y[:] += self.dx / 4.

        # add output properties
        fluid.add_output_arrays(['auhat', 'avhat', 'awhat', 'is_boundary', 'normal_norm'])

        return [fluid, tank, moving_boundary]

    def create_scheme(self):
        fluid = FluidsScheme(
            fluids=['fluid'],
            boundaries=['tank', 'moving_boundary'],
            dim=2,
            rho0=0.,
            mach_no=0.,
            u_max=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)

        s = SchemeChooser(default='fluid', fluid=fluid)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            rho0=self.fluid_rho,
            mach_no=self.mach_no,
            u_max=self.vref,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200)
        print("dt = %g"%self.dt)

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        if (int(t / dt) % 200) == 0:
            # print("inside")
            for pa in self.particles:
                if pa.name == 'moving_boundary':
                    pa.y[:] += self.dx / 4.


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
