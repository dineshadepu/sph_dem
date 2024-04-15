"""Rigid body falling in tank but 2D

Run it using:

python rb_falling_in_hs_tank.py --openmp --pfreq 100 --timestep 1e-4 --alpha 0.1 --fric-coeff 0.05 --en 0.05 --N 5 --no-of-bodies 1 --fluid-length-ratio 5 --fluid-height-ratio 10 --rigid-body-rho 2000 --tf 3 --scheme combined -d rb_falling_in_hs_tank_combined_output

python rb_falling_in_hs_tank.py --openmp --pfreq 100 --timestep 1e-4 --alpha 0.1 --fric-coeff 0.05 --en 0.05 --N 5 --no-of-bodies 1 --fluid-length-ratio 5 --fluid-height-ratio 10 --rigid-body-rho 2000 --tf 3 --scheme master -d rb_falling_in_hs_tank_master_output


TODO:

1. Create many particles
2. Run it on HPC
3. Change the speed
4. Change the dimensions
5. Commit SPH-DEM and current repository
6. Validate 1 spherical particle settled in hs tank
7. Validate 2 spherical particles settled in hs tank
8. Complete the manuscript
9. Different particle diameter
10. Different density
"""
import numpy as np
import sys

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.tools.geometry import get_3d_sphere
import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from pysph.sph.scheme import add_bool_argument
sys.path.insert(0, "./../")
from pysph_rfc_new.fluids import (get_particle_array_fluid, get_particle_array_boundary)

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from sph_dem.unresolved_coupling import (
    get_particle_array_fluid,
    get_particle_array_boundary,
    UnresolvedCouplingScheme)

from sph_dem.dem import (
    setup_dem_particles
   )

from sph_dem.geometry import (
    hydrostatic_tank_2d,
    translate_system_with_left_corner_as_origin,
    )


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=10,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=8,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=1,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=1.2,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=6,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--spherical-particle-rho", action="store", type=float,
                           dest="spherical_particle_rho", default=1500,
                           help="Density of spherical particle")

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
        # dimensions rigid body dimensions
        # All the particles are in circular or spherical shape
        self.spherical_particle_diameter = 0.1
        self.spherical_particle_radius = self.spherical_particle_diameter / 2.
        self.rigid_body_velocity = 0.

        # x - axis
        self.fluid_length = self.options.fluid_length_ratio * self.spherical_particle_diameter
        # y - axis
        self.fluid_height = self.options.fluid_height_ratio * self.spherical_particle_diameter
        # z - axis
        self.fluid_depth = 0.

        # x - axis
        self.tank_length = self.options.tank_length_ratio * self.fluid_length
        # y - axis
        self.tank_height = self.options.tank_height_ratio * self.fluid_height
        # z - axis
        self.tank_depth = 0.0

        self.tank_layers = 3

        # x - axis
        self.stirrer_length = self.fluid_length * 0.1
        # y - axis
        self.stirrer_height = self.fluid_height * 0.5
        # z - axis
        self.stirrer_depth = self.fluid_depth * 0.5
        self.stirrer_velocity = 1.
        # time period of stirrer
        self.T = (self.stirrer_length * 3) / self.stirrer_velocity

        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.spherical_particle_rho = self.options.spherical_particle_rho
        self.spherical_particle_E = 1e9
        self.spherical_particle_nu = 0.23

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
        self.dx = self.spherical_particle_diameter / self.N
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

        # ==========================
        # Numerical properties ends
        # ==========================
        # self.follow_combined_rb_solver = self.options.follow_combined_rb_solver
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_fluid_and_tank_particle_arrays(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                             self.tank_height, self.tank_layers,
                                             self.dx, self.dx, False)

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

    def create_spherical_particles(self):
        # =============================
        # DEM particles setup
        # =============================
        rad = self.spherical_particle_diameter / 2.
        m = np.pi * rad**self.dim * self.spherical_particle_rho
        x = np.asarray([self.fluid_length / 2.])
        y = np.asarray([self.fluid_height / 2.])
        z = np.asarray([0.])
        m_b = 4/3. * np.pi * self.spherical_particle_radius**3. * self.spherical_particle_rho
        _I = 2. / 5. * m_b * self.spherical_particle_radius**2
        I_inverse = 1. / _I
        E = 1e7
        nu = 0.2

        spherical_particles = get_particle_array(
            name='spherical_particles',
            x=x,
            y=y,
            h=self.hdx * self.spherical_particle_diameter,
            m_b=m_b,
            I_inverse=I_inverse,
            rho_b=self.spherical_particle_rho,
            rad_s=self.spherical_particle_radius - self.spherical_particle_radius / 10,
            E=E,
            nu=nu,
            dem_volume=self.spherical_particle_radius**2. * np.pi)
        dem_id = np.ones_like(x, dtype=int) * 0
        spherical_particles.add_property('dem_id', type='int', data=dem_id)
        setup_dem_particles(spherical_particles, 8, 4)
        # =============================
        # DEM particles setup ends
        # =============================

        # Add unresolved coupling properties for DEM particle


        return spherical_particles

    def create_particles(self):
        # This will create full particle array required for the scheme
        fluid, tank = self.create_fluid_and_tank_particle_arrays()

        # =========================
        # create spherical particles which follow DEM
        # =========================
        spherical_particles = self.create_spherical_particles()

        # =========================
        # create rigid body ends
        # =========================

        # remove fluid particles overlapping with the rigid body
        G.remove_overlap_particles(
            fluid, spherical_particles, self.spherical_particle_diameter/2., dim=self.dim
        )
        return [fluid, tank, spherical_particles]

    def create_scheme(self):
        scheme = UnresolvedCouplingScheme(
            fluids=['fluid'],
            boundaries=['tank'],
            spherical_particles=["spherical_particles"],
            dim=2,
            rho0=0.,
            # h=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)

        s = SchemeChooser(default='scheme', scheme=scheme)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            rho0=self.fluid_rho,
            # h=self.h,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200)
        print("dt = %g"%self.dt)


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
