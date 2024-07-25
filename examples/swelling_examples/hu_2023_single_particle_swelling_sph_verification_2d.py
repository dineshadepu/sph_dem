"""Rigid body falling in tank but 2D

Run it using:

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

from pysph_rfc_new.geometry import hydrostatic_tank_2d, create_circle_1, translate_system_with_left_corner_as_origin
# from geometry import hydrostatic_tank_2d, create_circle_1

from sph_dem.rigid_body.rigid_body_3d import (setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)

from pysph_dem.dem_simple import (setup_wall_dem)
from sph_dem.swelling.rfc_with_swelling import (
    ParticlesFluidScheme, add_rigid_fluid_properties_to_rigid_body,
    add_boundary_identification_properties,
    add_swelling_properties_to_rigid_body_hu_2d
)
from pysph_dem.swelling import (add_swelling_properties_to_rigid_body,
                                get_master_and_slave_swelling_rb_from_combined)


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
                        s_au, s_av, s_aw, s_x, s_y, s_z, s_rad_s,
                        s_radial_vel_count,
                        s_radial_vel_count_max,
                        dt):
        # Update the velocities to 1/2. time step
        # some variables to update the positions seamlessly
        bid, i9, i3, = declare('int', 3)
        bid = d_body_id[d_idx]
        i9 = 9 * bid
        i3 = 3 * bid

        if s_radial_vel_count[bid] < s_radial_vel_count_max[bid]:
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
            d_u[d_idx] += nx * self.swell_amount / dt
            d_v[d_idx] += ny * self.swell_amount / dt
            d_w[d_idx] += nz * self.swell_amount / dt

            # d_u[d_idx] = s_u[bid] + du + nx * self.swell_amount / dt**0.5
            # d_v[d_idx] = s_v[bid] + dv + ny * self.swell_amount / dt**0.5
            # d_w[d_idx] = s_w[bid] + dw + nz * self.swell_amount / dt**0.5


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=3,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=2.5,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=1,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=1.5,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=6,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=1500,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=2. * 1.9*1e-3,
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

        add_bool_argument(
            group,
            'use-pst',
            dest='use_pst',
            default=False,
            help='Use particle shifting scheme')

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
        self.rigid_body_diameter = self.options.rigid_body_diameter
        self.max_rigid_body_diameter = 3. * self.rigid_body_diameter
        # self.max_rigid_body_diameter = 1.11 * self.rigid_body_diameter
        self.rigid_body_velocity = 0.
        self.no_of_layers = self.options.no_of_layers
        self.no_of_bodies = 3 * 6 * self.options.no_of_layers

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
        self.rigid_body_rho = self.options.rigid_body_rho
        self.rigid_body_E = 1e9
        self.rigid_body_nu = 0.23

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
        self.dx = self.rigid_body_diameter / self.N
        self.pst_y_pos_limit = self.fluid_height - 5. * self.dx
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.use_pst = self.options.use_pst
        #     self.mach_no = self.vref / self.c0
        # else:
        #     self.mach_no = 0.
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 1.0
        # self.tf = 0.56 - 0.3192
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.00
        self.swell_freq = 200

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
        self.follow_combined_rb_solver = self.options.follow_combined_rb_solver
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

    def create_six_bodies(self):
        x1, y1 = create_circle_1(self.rigid_body_diameter, self.dx)
        x2, y2 = create_circle_1(self.rigid_body_diameter, self.dx)
        x3, y3 = create_circle_1(self.rigid_body_diameter, self.dx)

        x2 += self.rigid_body_diameter + self.dx * 2
        x3 += 2. * self.rigid_body_diameter + self.dx * 4

        # x2 += x1 + self.rigid_body_diameter + self.dx * 2
        # x3 += x2 + self.rigid_body_diameter + self.dx * 2

        x_three_bottom = np.concatenate((x1, x2, x3))
        y_three_bottom = np.concatenate((y1, y2, y3))

        x_three_top = np.copy(x_three_bottom)
        y_three_top = np.copy(y_three_bottom)
        y_three_top += self.rigid_body_diameter + self.dx * 2

        x = np.concatenate((x_three_bottom, x_three_top))
        y = np.concatenate((y_three_bottom, y_three_top))
        return x, y

    def create_rb_geometry_particle_array(self):
        x, y = create_circle_1(self.rigid_body_diameter, self.dx)
        z = np.zeros_like(x)

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

        x_circle, y_circle = create_circle_1(self.rigid_body_diameter, self.dx)

        body_id = np.array([])
        dem_id = np.array([])
        for i in range(1):
            body_id = np.concatenate((body_id, i * np.ones_like(x_circle,
                                                                dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x_circle,
                                                              dtype='int')))

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)
        return rigid_body_combined

    def create_stirrer(self):
        x_stirrer, y_stirrer = get_2d_block(dx=self.dx,
                                            length=self.stirrer_length,
                                            height=self.stirrer_height)
        m = self.dx**self.dim * self.fluid_rho
        stirrer = get_particle_array_boundary(name='stirrer',
                                              x=x_stirrer, y=y_stirrer,
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

        # =========================
        # Add boundary identification and pst properties to fluid
        # =========================
        add_boundary_identification_properties(fluid)
        kernel = QuinticSpline(dim=self.dim)
        fluid.add_constant('wdeltap', -1.)
        wdeltap = kernel.kernel(rij=self.dx, h=self.h)
        fluid.wdeltap[0] = wdeltap
        # add output properties
        fluid.add_output_arrays(['auhat', 'avhat', 'awhat', 'is_boundary', 'normal_norm'])
        # =========================
        # Add boundary identification and pst properties to fluid ends
        # =========================

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
        rigid_body_extent = max(rigid_body_combined.x) - min(rigid_body_combined.x)
        rigid_body_combined.x[:] -= min(rigid_body_combined.x) - min(fluid.x)
        rigid_body_combined.x[:] += self.fluid_length / 2.
        rigid_body_combined.x[:] -= self.rigid_body_diameter / 2.
        rigid_body_combined.y[:] += min(fluid.y) - min(rigid_body_combined.y)
        rigid_body_combined.y[:] += self.rigid_body_diameter / 4.

        # This is # 2, (Here we create a rigid body which is compatible with
        # combined rigid body solver formulation)
        setup_rigid_body(rigid_body_combined, self.dim)
        rigid_body_combined.h[:] = self.h
        rigid_body_combined.rad_s[:] = self.dx / 2.

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -1
        for i in range(1):
            sign *= -1
            lin_vel = np.concatenate((lin_vel, np.array([sign * 0., 0., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        set_linear_velocity(rigid_body_combined, lin_vel)
        set_angular_velocity(rigid_body_combined, ang_vel)

        # This is # 3,
        rigid_body_master, rigid_body_slave = get_master_and_slave_swelling_rb_from_combined(
            rigid_body_combined, self.max_rigid_body_diameter/2., self.dx)

        # This is # 4
        rigid_body_master.rad_s[:] = self.rigid_body_diameter / 2.
        rigid_body_master.h[:] = self.rigid_body_diameter * 2.
        add_contact_properties_body_master(rigid_body_master, 6, 3)

        # This is # 5
        add_rigid_fluid_properties_to_rigid_body(rigid_body_slave)
        # set mass and density to correspond to fluid
        rigid_body_slave.m[:] = self.fluid_rho * self.dx**2.
        rigid_body_slave.rho[:] = self.fluid_rho
        # similarly for combined rb particle arra
        add_rigid_fluid_properties_to_rigid_body(rigid_body_combined)
        # set mass and density to correspond to fluid
        rigid_body_combined.m[:] = self.fluid_rho * self.dx**2.
        rigid_body_combined.rho[:] = self.fluid_rho

        # Add swelling properties to master particle array
        add_swelling_properties_to_rigid_body(rigid_body_master)
        add_swelling_properties_to_rigid_body_hu_2d(rigid_body_master)

        rigid_body_master.D_0[:] = 2. * 1e-7
        rigid_body_master.D_lp[:] = 2. * 1e-7
        rigid_body_master.decay_delta[:] = 1.81
        rigid_body_master.c_wp[:] = 0.
        rigid_body_master.c_wl[:] = self.fluid_rho
        rigid_body_master.c_max[:] = rigid_body_master.c_wl[:]
        rigid_body_master.S_lp[:] = 2. * np.pi * self.rigid_body_diameter / 2.
        rigid_body_master.c_wp[:] = 0.
        rigid_body_master.m_dot_lp[:] = 0.
        rigid_body_master.m_w_t[:] = 0.
        rigid_body_master.m_p_0[:] = rigid_body_master.m_b[:]
        rigid_body_master.rho_w[:] = self.fluid_rho
        rigid_body_master.v_s_0[:] = np.pi * (self.rigid_body_diameter * 0.5)**2.
        rigid_body_master.v_s_t[:] = np.pi * (self.rigid_body_diameter * 0.5)**2.
        rigid_body_master.v_w_t[:] = 0.
        rigid_body_master.d_p[:] = self.rigid_body_diameter
        rigid_body_master.rad_s_prev[:] = self.rigid_body_diameter / 2.
        rigid_body_master.radial_vel_count_max[:] = 200
        rigid_body_master.radial_vel_count[:] = rigid_body_master.radial_vel_count_max[:]
        rigid_body_master.add_output_arrays(['rad_s', 'c_wp', 'v_w_t', 'd_p', 'rad_s_prev'])

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
                      min(tank.y) + self.tank_layers * self.dx + self.dx
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
                                             h=self.rigid_body_diameter/2.,
                                             rho_b=self.rigid_body_rho,
                                             rad_s=self.rigid_body_diameter/2.,
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

        rigid_body_slave.add_output_arrays([
            'fx_p',
            'fy_p',
            'fz_p',
            'fx_v',
            'fy_v',
            'fz_v',
            'fx',
            'fy',
            'fz'])

        rigid_body_master.add_output_arrays(['m_b'])

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
            dim=2,
            rho0=0.,
            mach_no=0.,
            u_max=0.,
            h=0.,
            pst_y_pos_limit=0.,
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
            pst_y_pos_limit=self.pst_y_pos_limit,
            mach_no=self.mach_no,
            u_max=self.vref,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200)
        print("dt = %g"%self.dt)

    def create_equations(self):
        # print("inside equations")
        eqns = self.scheme.get_equations()

        # # Apply external force
        # zero_frc = []
        # zero_frc.append(
        #     MakeForcesZeroOnRigidBody("rigid_body", sources=None))

        # # print(eqns.groups)
        # eqns.groups[-1].append(Group(equations=zero_frc,
        #                              condition=check_time_make_zero))
        # rb_interactions = eqns.groups[-1].pop(-1)
        # rb_interactions = eqns.groups[-1].pop(-1)

        # Add radial velocity to the slave body
        tmp = []
        tmp.append(
            AddRadialVelocityToSlaveBody(
                dest='rigid_body_combined_slave',
                sources=['rigid_body_combined_master'],
                swell_amount=self.dx / 12. * 1 / 200,
                max_radius=self.max_rigid_body_diameter / 2. - 2. * self.dx))

        # print(eqns.groups)
        # eqns.groups[0].insert(1, Group(equations=tmp, condition=swell_condition))
        if self.use_pst is False:
            print("inside use pst")
            eqns.groups[0].insert(1, Group(equations=tmp))

        # Add to the end so that we can see the velocities
        # print(eqns.groups)
        # eqns.groups[1].append(Group(equations=tmp, condition=swell_condition))
            eqns.groups[1].insert(1, Group(equations=tmp))

        return eqns

    def post_step(self, solver):
        from pysph_dem.geometry import create_circle_of_three_layers
        t = solver.t
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'rigid_body_combined_master':
                master = pa

            if pa.name == 'rigid_body_combined_slave':
                slave = pa

        for i in range(len(master.x)):
            # Compute the diffusion coefficient
            master.D_lp[i] = master.D_0[i] * np.exp(
                -master.decay_delta[i] * master.c_wp[i] / master.c_max[i])

            # Compute the surface area
            master.S_lp[i] = 2. * np.pi * master.rad_s[i]

            # Compute the rate of mass influx due to surrounding water
            conc_diff = master.c_wl[i] - master.c_wp[i]
            master.m_dot_lp[i] = (
                2. * master.S_lp[i] * master.D_lp[i] * conc_diff) / (
                    2. * master.rad_s[i])

            # update the mass of the particle
            delta_m = master.m_dot_lp[i] * self.dt
            master.m_w_t[i] += delta_m
            # TODO choose this carefully
            master.m_b[i] = master.m_p_0[i] + master.m_w_t[i]

            # update the volume of the spherical particle
            master.v_w_t[i] = master.m_w_t[i] / master.rho_w[i]
            master.v_s_t[i] = master.v_s_0[i] + master.v_w_t[i]

            # update the diameter
            master.rad_s[i] = (master.v_s_t[i] / np.pi)**(1/2)
            master.d_p[i] = 2 * master.rad_s[i]

            # update the moment of inertia of the particle
            # TODO
            master.rad_s[i] = (master.v_s_t[i] / np.pi)**(1/2)
            master.d_p[i] = 2 * master.rad_s[i]

            # update the concentration
            master.c_wp[i] = (master.m_b[i] - master.m_p_0[i]) / master.v_s_t[i]

            master.radial_vel_count[i] += 1

            if master.rad_s[i] - master.rad_s_prev[i] >= self.dx / 12:
                # master.rad_s[i] += self.dx / 8.
                # master.m_b[i] = np.pi * master.rad_s[i]**2. * self.rigid_body_rho
                # re-adjust the SPH particles
                x_new, y_new = create_circle_of_three_layers(
                    master.rad_s[i] * 2., self.dx)
                left_limit = master.body_limits[2 * i]
                slave.dx0[left_limit:left_limit+len(x_new)] = x_new[:]
                slave.dy0[left_limit:left_limit+len(x_new)] = y_new[:]

                slave.u[left_limit:left_limit+len(x_new)] = x_new[:]
                slave.v[left_limit:left_limit+len(x_new)] = y_new[:]

                # reset the previous radius
                master.rad_s_prev[i] = master.rad_s[i]

                # reset the radial velocity count
                master.radial_vel_count[i] = 0

    def customize_output(self):
        self._mayavi_config('''
        # b = particle_arrays['rigid_body']
        # b.scalar = 'm'
        b = particle_arrays['fluid']
        b.scalar = 'm'
        b.plot.module_manager.scalar_lut_manager.lut_mode = 'gist_yarg'
        b = particle_arrays['rigid_body_combined_slave']
        b.scalar = 'm'
        b.plot.module_manager.scalar_lut_manager.lut_mode = 'jet'
        ''')


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
