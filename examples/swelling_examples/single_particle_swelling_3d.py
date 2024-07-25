"""
Numerical modeling of floating bodies transport for flooding analysis in nuclear
reactor building

3.1. Validation of the PMS model for solid-fluid interaction

https://www.sciencedirect.com/science/article/pii/S0029549318307350#b0140

"""
import numpy as np
import sys

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import (get_2d_tank, get_2d_block, get_3d_block)
from pysph.tools.geometry import get_3d_sphere
import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array
from pysph.examples.solid_mech.impact import add_properties

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

from pysph_dem.rigid_body.boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from pysph_rfc_new.geometry import hydrostatic_tank_2d, create_circle_1, translate_system_with_left_corner_as_origin
# from geometry import hydrostatic_tank_2d, create_circle_1

from sph_dem.rigid_fluid_coupling import (
    ParticlesFluidScheme, add_rigid_fluid_properties_to_rigid_body)
from sph_dem.geometry import (get_fluid_tank_new_rfc_3d, get_3d_block_rfc)

from sph_dem.rigid_body.rigid_body_3d import (setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)

from pysph_dem.dem_simple import (setup_wall_dem)


def swell_condition(t, dt):
    if (int(t / dt) % 50) == 0.:
        return True
    else:
        return False


class MakeForcesZeroOnRigidBody(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.

    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')

        frc = dst.force
        trq = dst.torque

        frc[:] = 0
        trq[:] = 0


class AddRadialVelocityToSlaveBody(Equation):
    def __init__(self, dest, sources, swell_amount, max_radius):
        self.swell_amount = swell_amount
        self.max_radius = max_radius
        super(AddRadialVelocityToSlaveBody, self).__init__(dest, sources)

    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
                        d_au, d_av, d_aw, d_body_id, d_is_boundary, d_normal0, d_normal,
                        s_R, s_omega_x, s_omega_y, s_omega_z, s_u, s_v, s_w,
                        s_au, s_av, s_aw, s_x, s_y, s_z, s_rad_s, dt):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly
        bid, i9, i3, = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (s_R[i9 + 0] * d_dx0[d_idx] + s_R[i9 + 1] * d_dy0[d_idx] +
              s_R[i9 + 2] * d_dz0[d_idx])
        dy = (s_R[i9 + 3] * d_dx0[d_idx] + s_R[i9 + 4] * d_dy0[d_idx] +
              s_R[i9 + 5] * d_dz0[d_idx])
        dz = (s_R[i9 + 6] * d_dx0[d_idx] + s_R[i9 + 7] * d_dy0[d_idx] +
              s_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = s_x[bid] + dx
        d_y[d_idx] = s_y[bid] + dy
        d_z[d_idx] = s_z[bid] + dz

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm

        # Radial direction
        dist = (dx**2. + dy**2. + dz**2.)**0.5
        if dist > 1e-12:
            nx = dx / dist
            ny = dy / dist
            nz = dz / dist
        else:
            nx = 0.
            ny = 0.
            nz = 0.

        du = s_omega_y[bid] * dz - s_omega_z[bid] * dy
        dv = s_omega_z[bid] * dx - s_omega_x[bid] * dz
        dw = s_omega_x[bid] * dy - s_omega_y[bid] * dx

        d_u[d_idx] = s_u[bid] + du
        d_v[d_idx] = s_v[bid] + dv
        d_w[d_idx] = s_w[bid] + dw
        if s_rad_s[bid] < self.max_radius:
            d_u[d_idx] += nx * self.swell_amount / dt
            d_v[d_idx] += ny * self.swell_amount / dt
            d_w[d_idx] += nz * self.swell_amount / dt

        # d_u[d_idx] = s_u[bid] + du + nx * self.swell_amount / dt**0.5
        # d_v[d_idx] = s_v[bid] + dv + ny * self.swell_amount / dt**0.5
        # d_w[d_idx] = s_w[bid] + dw + nz * self.swell_amount / dt**0.5


class RBTankForce(Equation):
    def __init__(self, dest, sources, kn, en, fric_coeff):
        self.kn = kn
        self.en = en
        self.fric_coeff = fric_coeff
        super(RBTankForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, d_rad_s,
             s_idx, s_rad_s, d_dem_id, s_dem_id,
             d_nu, s_nu, d_E, s_E, d_G, s_G,
             d_m, s_m,
             d_body_id,
             XIJ, RIJ, R2IJ, VIJ):
        overlap = 0
        if RIJ > 1e-9:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        if overlap > 1e-12:
            # normal vector passing from particle i to j
            nij_x = -XIJ[0] / RIJ
            nij_y = -XIJ[1] / RIJ
            nij_z = -XIJ[2] / RIJ

            # overlap speed: a scalar
            vijdotnij = VIJ[0] * nij_x + VIJ[1] * nij_y + VIJ[2] * nij_z

            # normal velocity
            vijn_x = vijdotnij * nij_x
            vijn_y = vijdotnij * nij_y
            vijn_z = vijdotnij * nij_z

            kn = self.kn

            # normal force with conservative and dissipation part
            fn_x = -kn * overlap * nij_x
            fn_y = -kn * overlap * nij_y
            fn_z = -kn * overlap * nij_z
            # fn_x = -kn * overlap * nij_x
            # fn_y = -kn * overlap * nij_y
            # fn_z = -kn * overlap * nij_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=8,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=6,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=1,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=1.4,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=10,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=2120,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=1e-3,
                           help="Diameter of each particle")

        add_bool_argument(
            group,
            'follow-combined-rb-solver',
            dest='follow_combined_rb_solver',
            default=True,
            help='Use combined particle array solver for rigid body dynamics')

        group.add_argument("--no-of-layers", action="store", type=int,
                           dest="no_of_layers", default=2,
                           help="Total no of rigid bodies layers")

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
        self.rigid_body_velocity = 0.
        # x - axis
        self.rigid_body_width = 20. * 1e-3
        # y - axis
        self.rigid_body_length = 20. * 1e-3
        # z - axis
        self.rigid_body_depth = 20. * 1e-3
        # Rigid body diameter
        self.rigid_body_diameter = 1.9 * 1e-3

        self.no_of_layers = self.options.no_of_layers
        self.no_of_bodies = 1

        # x - axis (out of the plane)
        self.fluid_width = 150. * 1e-3
        # y - axis
        self.fluid_length = 140. * 1e-3
        # z - axis
        self.fluid_depth = 131. * 1e-3

        # x - axis
        self.tank_width = 150. * 1e-3
        # y - axis
        self.tank_length = 140. * 1e-3
        # z - axis
        self.tank_depth = 145. * 1e-3

        self.tank_layers = 3

        # x - axis
        self.stirrer_width = self.fluid_width * 0.1
        # y - axis
        self.stirrer_length = self.fluid_length * 0.5
        # z - axis
        self.stirrer_depth = self.stirrer_length
        self.stirrer_velocity = 20 * 1e-3
        # time period of stirrer
        self.T = self.stirrer_length / self.stirrer_velocity

        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = self.options.rigid_body_rho
        self.rigid_body_E = 1e9
        self.rigid_body_nu = 0.23

        self.gx = 0.
        self.gy = 0.
        self.gz = -9.81
        self.dim = 3
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        self.N = self.options.N
        self.dx = self.rigid_body_length / self.N
        print("Spacing is", self.dx)
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gz) * self.fluid_depth)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.wall_time = 0.3
        self.tf = 0.5 + self.wall_time
        # self.tf = 0.56 - 0.3192
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.00

        # Setup default parameters.
        dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 1e5
        if self.nu > 1e-12:
            dt_viscous = 0.125 * self.h**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h/(np.abs(self.gz)))
        print("dt_cfl", dt_cfl, "dt_viscous", dt_viscous, "dt_force", dt_force)

        self.dt = min(dt_cfl, dt_force)
        self.dt *= 5
        print("Computed stable dt is: ", self.dt)
        self.total_steps = self.tf / self.dt
        print("Total steps in this simulation are", self.total_steps)
        self.pfreq = int(self.total_steps / 100)
        print("Pfreq is", self.pfreq)
        # self.dt = 1e-4
        # ==========================
        # Numerical properties ends
        # ==========================

        # ==========================
        # Numerical properties ends
        # ==========================
        self.follow_combined_rb_solver = self.options.follow_combined_rb_solver
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_fluid_and_tank_particle_arrays(self):
        xf, yf, zf, xt, yt, zt = get_fluid_tank_new_rfc_3d(
            self.fluid_width,
            self.fluid_length,
            self.fluid_depth,
            self.tank_length,
            self.tank_depth,
            self.tank_layers,
            self.dx, self.dx, True)

        m = self.dx**self.dim * self.fluid_rho
        # print(self.dim, "dim")
        # print("m", m)
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=self.h, m=m, rho=self.fluid_rho)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=self.h, m=m, rho=self.fluid_rho,
                                           E=1e9,
                                           nu=0.3,
                                           G=1e9,
                                           rad_s=self.dx / 2.
                                           )
        dem_id = np.ones_like(xt, dtype='int')
        tank.add_property('dem_id', type='int', data=dem_id)

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_rho * self.gy * (max(fluid.y) - fluid.y[:])
        fluid.c0_ref[0] = self.c0
        fluid.p0_ref[0] = self.p0
        return fluid, tank

    def create_rb_geometry_particle_array(self):
        x, y, z = get_3d_block_rfc(dx=self.dx,
                                   width=self.rigid_body_width - self.dx,
                                   length=self.rigid_body_length - self.dx,
                                   depth=self.rigid_body_depth - self.dx)

        # y[:] += self.fluid_depth + self.rigid_body_diameter
        # # x[:] += self.fluid_length/2. + self.rigid_body_diameter
        # x[:] += self.fluid_length/2. - self.rigid_body_diameter * 4.
        # x[:] += self.rigid_body_diameter * 4.
        # x[:] += self.rigid_body_diameter * 4.
        # # z = np.zeros_like(x)

        m = self.rigid_body_rho * self.dx**self.dim * np.ones_like(x)
        h = self.h
        rad_s = self.dx / 2.
        # This is # 1
        rigid_body_combined = get_particle_array(name='rigid_body_combined',
                                                 x=x,
                                                 y=y,
                                                 z=z,
                                                 h=1.2 * h,
                                                 m=m,
                                                 E=1e9,
                                                 nu=0.23,
                                                 rho=self.fluid_rho)

        x_block, y_block, z_block = get_3d_block_rfc(dx=self.dx,
                                                     width=self.rigid_body_width - self.dx,
                                                     length=self.rigid_body_length - self.dx,
                                                     depth=self.rigid_body_depth - self.dx)

        body_id = np.array([])
        dem_id = np.array([])
        for i in range(1):
            body_id = np.concatenate((body_id, i * np.ones_like(x_block,
                                                                dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x_block,
                                                              dtype='int')))

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)
        return rigid_body_combined

    def create_stirrer(self):
        x_stirrer, y_stirrer, z_stirrer = get_3d_block(
            dx=self.dx,
            length=self.stirrer_width,
            height=self.stirrer_length,
            depth=self.stirrer_depth)

        m = self.dx**self.dim * self.fluid_rho
        stirrer = get_particle_array_boundary(name='stirrer',
                                              x=x_stirrer,
                                              y=y_stirrer,
                                              z=z_stirrer,
                                              u=self.stirrer_velocity,
                                              h=self.h, m=m,
                                              rho=self.fluid_rho,
                                              E=1e9,
                                              nu=0.3,
                                              G=1e9,
                                              rad_s=self.dx)
        dem_id = np.ones_like(x_stirrer, dtype='int')
        stirrer.add_property('dem_id', type='int', data=dem_id)
        return stirrer

    def create_particles(self):
        # This will create full particle array required for the scheme
        fluid, tank = self.create_fluid_and_tank_particle_arrays()
        fluid.p[:] = - self.fluid_rho * self.gz * (max(fluid.z) - fluid.z[:])

        # =========================
        # create rigid body
        # =========================
        # Steps in creating the the right rigid body
        # 1. Create a particle array
        # 2. Get the combined rigid body with properties computed
        # 3. Seperate out center of mass and particles into two particle arrays
        # 4. Add dem contact properties to the master particle array
        # 5. Add rigid fluid coupling properties to the slave particle array
        # Some important notes.
        # 1. Make sure the mass is consistent for all the equations, since
        # we use 'm_b' for some equations and 'm' for fluid coupling
        rigid_body_combined = self.create_rb_geometry_particle_array()
        rigid_body_combined.z[:] += min(fluid.z) - min(rigid_body_combined.z)
        rigid_body_combined.z[:] += max(fluid.z) - min(rigid_body_combined.z)
        rigid_body_combined.z[:] -= self.rigid_body_length / 2.

        # This is # 2, (Here we create a rigid body which is compatible with
        # combined rigid body solver formulation)
        setup_rigid_body(rigid_body_combined, self.dim)
        rigid_body_combined.h[:] = self.h
        rigid_body_combined.rad_s[:] = self.dx / 2.

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -1
        for i in range(self.no_of_bodies):
            sign *= -1
            lin_vel = np.concatenate((lin_vel, np.array([sign * 0., 0., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        set_linear_velocity(rigid_body_combined, lin_vel)
        set_angular_velocity(rigid_body_combined, ang_vel)

        # This is # 3,
        rigid_body_master, rigid_body_slave = get_master_and_slave_rb(
            rigid_body_combined
        )

        # This is # 4
        rigid_body_master.rad_s[:] = self.dx
        rigid_body_master.h[:] = self.dx
        add_contact_properties_body_master(rigid_body_master, 6, 3)

        # This is # 5
        add_rigid_fluid_properties_to_rigid_body(rigid_body_slave)
        # set mass and density to correspond to fluid
        rigid_body_slave.m[:] = self.fluid_rho * self.dx**self.dim
        rigid_body_slave.rho[:] = self.fluid_rho
        # similarly for combined rb particle arra
        add_rigid_fluid_properties_to_rigid_body(rigid_body_combined)
        # set mass and density to correspond to fluid
        rigid_body_combined.m[:] = self.fluid_rho * self.dx**self.dim
        rigid_body_combined.rho[:] = self.fluid_rho

        # =========================
        # create rigid body ends
        # =========================

        # ======================
        # create wall for rigid body
        # ======================
        # left right bottom
        x = np.array([min(tank.x) + self.tank_layers * self.dx,
                      max(tank.x) - self.tank_layers * self.dx,
                      max(tank.x) / 2
                     ])
        # x[:] += disp_x
        y = np.array([max(tank.y) / 2.,
                      max(tank.y) / 2.,
                      min(tank.y) + self.tank_layers * self.dx
                      ])
        normal_x = np.array([1., -1., 0.])
        normal_y = np.array([0., 0., 1.])
        normal_z = np.array([0., 0., 0.])
        rigid_body_wall = get_particle_array(name='rigid_body_wall',
                                             x=x,
                                             y=y,
                                             normal_x=normal_x,
                                             normal_y=normal_y,
                                             normal_z=normal_z,
                                             h=self.h,
                                             rho_b=self.rigid_body_rho,
                                             rad_s=self.rigid_body_width/2.,
                                             E=69. * 1e9,
                                             nu=0.3,
                                             G=69. * 1e5)
        dem_id = np.array([0, 0, 0])
        rigid_body_wall.add_property('dem_id', type='int', data=dem_id)
        rigid_body_wall.add_constant('no_wall', [3])
        setup_wall_dem(rigid_body_wall)

        # remove fluid particles overlapping with the rigid body
        G.remove_overlap_particles(
            fluid, rigid_body_combined, self.dx, dim=self.dim
        )
        # remove fluid particles overlapping with the rigid body
        G.remove_overlap_particles(
            fluid, rigid_body_slave, self.dx, dim=self.dim
        )

        # Add stirrer
        stirrer = self.create_stirrer()
        # set the position of the stirrer
        stirrer.x[:] += ((min(fluid.x) - min(stirrer.x)) +
                         self.fluid_length * 0.5) - self.stirrer_length * 0.5
        # stirrer.x[:] -= self.rigid_body_diameter
        stirrer.y[:] += ((min(fluid.y) - min(stirrer.y)) +
                         self.fluid_depth) - self.stirrer_depth * 0.5
        stirrer.y[:] -= self.rigid_body_width
        # G.remove_overlap_particles(
        #     fluid, stirrer, self.dx, dim=self.dim
        # )

        # add extra output properties
        rigid_body_slave.add_output_arrays(['fx', 'fy', 'fz'])

        # Add properties to rigid body to hold the body still until some time
        add_properties(rigid_body_master, 'hold_x', 'hold_y', 'hold_z')
        rigid_body_master.hold_x[:] = rigid_body_master.x[:]
        rigid_body_master.hold_y[:] = rigid_body_master.y[:]
        rigid_body_master.hold_z[:] = rigid_body_master.z[:]

        return [fluid, tank, rigid_body_master, rigid_body_slave,
                rigid_body_wall]

    def create_scheme(self):
        master = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['tank'],
            # rigid_bodies_combined=[],
            rigid_bodies_master=["rigid_body_combined_master"],
            rigid_bodies_slave=["rigid_body_combined_slave"],
            rigid_bodies_wall=["rigid_body_wall"],
            stirrer=[],
            dim=3,
            rho0=0.,
            h=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)

        s = SchemeChooser(default='master', master=master)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            rho0=self.fluid_rho,
            h=self.h,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy,
            gz=self.gz
        )

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=self.pfreq)
        print("dt = %g"%self.dt)

    def create_equations(self):
        # print("inside equations")
        eqns = self.scheme.get_equations()

        rb_interactions = eqns.groups[-1][-2].equations.pop(2)
        rb_interactions = eqns.groups[-1][-2].equations.pop(1)
        print(rb_interactions)

        # Apply external force
        zero_frc = []
        zero_frc.append(
            RBTankForce("rigid_body_combined_slave", sources=['tank'], kn=1e4, en=0.1, fric_coeff=0.4))

        # print(eqns.groups)
        eqns.groups[-1].insert(-1, Group(equations=zero_frc))

        return eqns

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'stirrer':
                if (t // self.T) % 2 == 0:
                    pa.u[:] = self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt
                else:
                    pa.u[:] = -self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt
            if pa.name == 'rigid_body_combined_master':
                if t < self.wall_time:
                    pa.x[:] = pa.hold_x[:]
                    pa.y[:] = pa.hold_y[:]
                    pa.z[:] = pa.hold_z[:]

                    pa.u[:] = 0.
                    pa.v[:] = 0.
                    pa.w[:] = 0.

                    pa.omega_x[:] = 0.
                    pa.omega_y[:] = 0.
                    pa.omega_z[:] = 0.


    def customize_output(self):
        self._mayavi_config('''
        # b = particle_arrays['rigid_body']
        # b.scalar = 'm'
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output
        import os

        info = self.read_info(fname)
        files = self.output_files

        # initial position of the cylinder
        fname = files[0]
        data = load(fname)
        rigid_body = data['arrays']['rigid_body_combined_master']
        y_0 = rigid_body.x[0]

        t = []
        z = []

        for sd, rigid_body in iter_output(files, 'rigid_body_combined_master'):
            _t = sd['t']
            if _t > self.wall_time:
                # y.append(rigid_body.xcm[1])
                # u.append(rigid_body.vcm[0])
                # v.append(rigid_body.vcm[1])
                z.append(rigid_body.z[0])
                t.append(_t - self.wall_time)
        # set the z value
        z[:] += 131 * 1e-3 - z[0]
        # print(t, "t is ")
        # non dimentionalize it
        # penetration_current = (np.asarray(y)[::1] - y_0) / self.rigid_body_diameter
        # t_current = np.asarray(t)[::1] * (9.81 / self.rigid_body_diameter)**0.5

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        # We use Sun 2018 accurate and efficient water entry paper data for validation
        data_y_position_wu_2014_exp = np.loadtxt(os.path.join(
            directory, 'wu_2014_falling_solid_y_position_exp_data.csv'), delimiter=',')

        t_exp, z_position_exp = data_y_position_wu_2014_exp[:, 0], data_y_position_wu_2014_exp[:, 1]
        # t_BEM, penetration_BEM = data_y_penetration_sun_2018_BEM[:, 0], data_y_penetration_sun_2018_BEM[:, 1]
        # t_SPH, penetration_SPH = data_y_penetration_sun_2018_SPH[:, 0], data_y_penetration_sun_2018_SPH[:, 1]
        # # =================
        # # sort webplot data
        # p = t_SPH.argsort()
        # t_SPH = t_SPH[p]
        # penetration_SPH = penetration_SPH[p]
        # # t_SPH = np.delete(t_SPH, np.where(t_SPH > 1.65 and t_SPH < 1.75))
        # # penetration_SPH = np.delete(penetration_SPH,  np.where(t_SPH > 1.65 and t_SPH < 1.75))
        # t_SPH = np.delete(t_SPH, -4)
        # penetration_SPH = np.delete(penetration_SPH, -4)

        # p = t_BEM.argsort()
        # t_BEM = t_BEM[p]
        # penetration_BEM = penetration_BEM[p]
        # # sort webplot data
        # # =================

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 t_exp=t_exp,
                 z_position_exp=z_position_exp,
                 t_current=t,
                 z_position_current=z)
        data = np.load(res)

        # ========================
        # Variation of y penetration
        # ========================
        plt.clf()
        # plt.plot(t_SPH, penetration_SPH, "-+", label='SPH')
        # plt.plot(t_BEM, penetration_BEM, "--", label='BEM')
        # plt.plot(t_current, -penetration_current, "-", label='Current')
        # print("len of t is", len(t))
        # print("len of v is", len(fy))

        plt.plot(t, z, label='Current')
        plt.plot(t_exp, z_position_exp, "^", label='Experimental')

        plt.title('Variation in z-position (meters)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Vertical displacement (m)')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "z_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
