"""3D dam break

Run it using:

python 3d_dam_break.py --openmp --tf 1 --alpha 0.1 --tank-length-ratio 3.

"""
import numpy as np

# imports related to the application class
from pysph.solver.application import Application
from pysph.sph.scheme import (SchemeChooser,
                              add_bool_argument)

# geometry imports
import pysph.tools.geometry as G
from sph_dem.geometry import (
    get_fluid_tank_3d, get_truncated_circle_from_3d_block,
    hydrostatic_tank_2d,
    translate_system_with_left_corner_as_origin,
    create_tank_2d_from_block_2d,
    get_cylindrical_fluid_tank_3d,
    get_3d_block)

from sph_dem.fluids import (FluidsScheme,
                            get_particle_array_fluid,
                            get_particle_array_boundary)


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=8,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=6,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--fluid-depth-ratio", action="store", type=float,
                           dest="fluid_depth_ratio", default=3,
                           help="Ratio between the fluid depth and rigid body diameter (z-direction)")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=3,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=1.4,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--tank-depth-ratio", action="store", type=float,
                           dest="tank_depth_ratio", default=1,
                           help="Ratio between the tank depth and fluid depth (z-direction)")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=7,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--dimension", action="store", type=int, dest="dim",
                           default=3,
                           help="Dimension of the problem")

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
        self.dim = self.options.dim

        # dimensions rigid body dimensions
        # All the particles are in circular or spherical shape
        self.rigid_body_diameter = 1.

        # x - axis
        self.fluid_length = self.options.fluid_length_ratio * self.rigid_body_diameter
        # y - axis
        self.fluid_height = self.options.fluid_height_ratio * self.rigid_body_diameter
        # z - axis
        self.fluid_depth = self.options.fluid_depth_ratio * self.rigid_body_diameter

        # x - axis
        self.tank_length = self.options.tank_length_ratio * self.fluid_length
        # y - axis
        self.tank_height = self.options.tank_height_ratio * self.fluid_height
        # z - axis
        self.tank_depth = self.options.tank_depth_ratio * self.fluid_depth

        self.tank_layers = 3

        # x - axis
        self.stirrer_length = self.fluid_length * 0.1
        # y - axis
        self.stirrer_height = self.fluid_height * 0.5
        # z - axis
        self.stirrer_depth = self.fluid_depth * 0.5
        self.stirrer_velocity = 1.
        # time period of stirrer
        self.T = 0.1 * self.fluid_length / self.stirrer_velocity

        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        self.N = self.options.N
        self.dx = self.rigid_body_diameter / self.N
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 1.0
        # self.tf = 0.56 - 0.3192
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.00

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
        xf, yf, zf, xt, yt, zt = get_cylindrical_fluid_tank_3d(
            self.fluid_length,
            self.fluid_height,
            self.fluid_depth,
            self.tank_length,
            self.tank_height,
            self.tank_layers,
            self.dx, self.dx, False)

        m = self.dx**self.dim * self.fluid_rho
        # print(self.dim, "dim")
        # print("m", m)
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=self.h, m=m, rho=self.fluid_rho)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=self.h, m=m, rho=self.fluid_rho)

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_rho * self.gy * (max(fluid.y) - fluid.y[:])
        fluid.c0_ref[0] = self.c0
        fluid.p0_ref[0] = self.p0
        return fluid, tank

    def create_stirrer(self):
        x_stirrer, y_stirrer, z_stirrer = get_3d_block(dx=self.dx,
                                                       length=self.stirrer_length,
                                                       height=self.stirrer_height,
                                                       depth=self.stirrer_depth
                                                       )
        m = self.dx**self.dim * self.fluid_rho
        stirrer_sph = get_particle_array_boundary(name='stirrer',
                                                  x=x_stirrer, y=y_stirrer,
                                                  z=z_stirrer,
                                                  u=self.stirrer_velocity,
                                                  h=self.h, m=m,
                                                  rho=self.fluid_rho)
        return stirrer_sph

    def create_particles(self):
        # This will create full particle array required for the scheme
        fluid, tank = self.create_fluid_and_tank_particle_arrays()

        stirrer_sph = self.create_stirrer()
        # set the position of the stirrer
        stirrer_sph.x[:] += ((min(fluid.x) - min(stirrer_sph.x)) +
                             self.fluid_length * 0.5) - self.stirrer_length * 0.5
        stirrer_sph.y[:] += ((min(fluid.y) - min(stirrer_sph.y)) +
                             self.fluid_height) - self.stirrer_height * 0.5
        G.remove_overlap_particles(
            fluid, stirrer_sph, self.dx, dim=self.dim
        )
        return [fluid, tank, stirrer_sph]

    def create_scheme(self):
        fluid = FluidsScheme(
            fluids=['fluid'],
            boundaries=['tank', 'stirrer'],
            dim=0,
            rho0=0.,
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
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200)
        print("dt = %g"%self.dt)

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'stirrer_sph':
                if (t // self.T) % 2 == 0:
                    pa.u[:] = self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt
                else:
                    pa.u[:] = -self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt



if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
