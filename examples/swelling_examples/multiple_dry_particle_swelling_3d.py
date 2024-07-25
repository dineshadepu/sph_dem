"""
Numerical modeling of floating bodies transport for flooding analysis in nuclear
reactor building

3.1. Validation of the PMS model for solid-fluid interaction

https://www.sciencedirect.com/science/article/pii/S0029549318307350#b0140

"""
import numpy as np
import sys
from math import pi

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

from sph_dem.geometry import (get_fluid_tank_new_rfc_3d, get_3d_block_rfc)

from sph_dem.rigid_body.rigid_body_3d import (RigidBody3DScheme,
                                              setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)
from pysph_dem.swelling import (add_swelling_properties_to_rigid_body,
                                get_master_and_slave_swelling_rb_from_combined_3d)

from pysph_dem.dem_simple import (setup_wall_dem)
from pysph_dem.geometry import get_regular_3d_sphere


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
        group.add_argument("--N", action="store", type=int, dest="N",
                           default=3,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=2120,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=1e-3,
                           help="Diameter of each particle")

        group.add_argument("--no-of-bodies", action="store", type=int, dest="no_of_bodies",
                           default=50,
                           help="Number of spherical particles")

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
        # Rigid body diameter
        self.rigid_body_diameter = 1.9 * 1e-3
        self.rigid_body_radius = self.rigid_body_diameter / 2.
        self.max_rigid_body_diameter = 3. * self.rigid_body_diameter
        self.max_rigid_body_radius = self.max_rigid_body_diameter / 2.

        # x - axis
        self.tank_width = 40. * 1e-3
        # y - axis
        self.tank_length = 20. * 1e-3
        # z - axis
        self.tank_depth = 110. * 1e-3

        self.tank_layers = 3

        self.swell_time_delay = 0.03
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
        self.dx = self.rigid_body_diameter / self.N
        print("Spacing is", self.dx)
        self.h = self.hdx * self.dx
        self.wall_time = 0.3
        self.tf = 0.5 + self.wall_time
        self.swell_freq = 100

        # Setup default parameters.
        # Set as per DEM
        self.dt = 1e-5
        print("Computed stable dt is: ", self.dt)
        self.total_steps = self.tf / self.dt
        print("Total steps in this simulation are", self.total_steps)
        self.pfreq = int(self.total_steps / 100)
        print("Pfreq is", self.pfreq)
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
        # x, y, z = get_3d_block_rfc(dx=self.dx,
        #                            width=self.rigid_body_diameter,
        #                            length=self.rigid_body_diameter,
        #                            depth=self.rigid_body_diameter)

        # =================================================
        # For this specific example set the particles again
        # =================================================
        # re-adjust the SPH particles
        area_sphere = 4. * pi * self.rigid_body_radius**2.
        num_point = area_sphere / self.dx**2.
        x1, y1, z1 = get_regular_3d_sphere(self.rigid_body_radius, num_point)
        x1, y1, z1 = np.asarray(x1), np.asarray(y1), np.asarray(z1)
        # print("len of sphere particles", len(x))
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for i in range(5):
            x_new = x1 + (1.5 * i) * self.rigid_body_diameter
            x = np.concatenate((x, x_new))
            y = np.concatenate((y, y1))
            z = np.concatenate((z, z1))

        # left_limit = rigid_body_master.body_limits[2 * 0]
        # rigid_body_slave.dx0[left_limit:left_limit+len(x_new)] = x_new[:]
        # rigid_body_slave.dy0[left_limit:left_limit+len(x_new)] = y_new[:]
        # =================================================
        # For this specific example set the particles again
        # =================================================

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

        # x_block, y_block, z_block = get_3d_block_rfc(dx=self.dx,
        #                                              width=self.rigid_body_diameter,
        #                                              length=self.rigid_body_diameter,
        #                                              depth=self.rigid_body_diameter)
        x_block, y_block, z_block = get_regular_3d_sphere(
            self.rigid_body_radius, num_point)

        body_id = np.array([])
        dem_id = np.array([])
        for i in range(5):
            body_id = np.concatenate((body_id, i * np.ones_like(x_block,
                                                                dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x_block,
                                                              dtype='int')))

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)
        return rigid_body_combined

    def create_particles(self):
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
        # rigid_body_combined.z[:] += min(fluid.z) - min(rigid_body_combined.z)
        # rigid_body_combined.z[:] += max(fluid.z) - min(rigid_body_combined.z)
        # rigid_body_combined.z[:] -= self.rigid_body_length / 2.

        # This is # 2, (Here we create a rigid body which is compatible with
        # combined rigid body solver formulation)
        setup_rigid_body(rigid_body_combined, self.dim)
        rigid_body_combined.h[:] = self.h
        rigid_body_combined.rad_s[:] = self.dx / 2.

        # This is # 3,
        # rigid_body_master, rigid_body_slave = get_master_and_slave_rb(
        #     rigid_body_combined
        # )
        rigid_body_master, rigid_body_slave = get_master_and_slave_swelling_rb_from_combined_3d(
            rigid_body_combined, 1.2 * self.max_rigid_body_radius, self.dx)

        # =================================================
        # For this specific example set the particles again
        # =================================================
        # re-adjust the SPH particles
        # area_sphere = 4. * pi * self.rigid_body_radius
        # num_point = area_sphere / self.dx
        # x_new, y_new, z_new = get_regular_3d_sphere(
        #     self.rigid_body_radius, num_point)
        # print("len of sphere particles", len(x_new))

        # left_limit = rigid_body_master.body_limits[2 * 0]
        # rigid_body_slave.dx0[left_limit:left_limit+len(x_new)] = x_new[:]
        # rigid_body_slave.dy0[left_limit:left_limit+len(x_new)] = y_new[:]
        # =================================================
        # For this specific example set the particles again
        # =================================================

        # This is # 4
        rigid_body_master.rad_s[:] = self.dx
        rigid_body_master.h[:] = self.rigid_body_diameter * 2.
        add_contact_properties_body_master(rigid_body_master, 18, 5)

        # # This is # 5
        # add_rigid_fluid_properties_to_rigid_body(rigid_body_slave)
        # # set mass and density to correspond to fluid
        # rigid_body_slave.m[:] = self.fluid_rho * self.dx**self.dim
        # rigid_body_slave.rho[:] = self.fluid_rho
        # # similarly for combined rb particle arra
        # add_rigid_fluid_properties_to_rigid_body(rigid_body_combined)
        # # set mass and density to correspond to fluid
        # rigid_body_combined.m[:] = self.fluid_rho * self.dx**self.dim
        # rigid_body_combined.rho[:] = self.fluid_rho

        # =========================
        # create rigid body ends
        # =========================

        # ======================
        # create wall for rigid body
        # ======================
        # left right bottom front back
        x = np.array([min(rigid_body_master.x) - self.rigid_body_diameter * 5.,
                      max(rigid_body_master.x) + self.rigid_body_diameter * 5.,
                      min(rigid_body_master.x) + (max(rigid_body_master.x) -
                                                  min(rigid_body_master.x)) / 2 - self.rigid_body_diameter * 5.,
                      min(rigid_body_master.x) + (max(rigid_body_master.x) -
                                                  min(rigid_body_master.x)) / 2,
                      min(rigid_body_master.x) + (max(rigid_body_master.x) -
                                                  min(rigid_body_master.x)) / 2])
        # x[:] += disp_x
        y = np.array([self.rigid_body_diameter * 4.,
                      self.rigid_body_diameter * 4.,
                      min(rigid_body_master.y) - self.rigid_body_diameter * 1.,
                      self.rigid_body_diameter * 8.,
                      # self.rigid_body_diameter * 4.,
                      self.rigid_body_diameter * 4.,
                      ])
        z = np.array([self.rigid_body_diameter * 2,
                      self.rigid_body_diameter * 2,
                      self.rigid_body_diameter * 2,
                      # 0.,
                      self.rigid_body_diameter * 2,
                      # self.rigid_body_diameter * 2])
                      min(rigid_body_master.z) - self.rigid_body_diameter * 1.])
        normal_x = np.array([1., -1., 0., 0., 0.])
        normal_y = np.array([0., 0., 1., -1., 0.])
        normal_z = np.array([0., 0., 0., 0., 1.])
        rigid_body_wall = get_particle_array(name='rigid_body_wall',
                                             x=x,
                                             y=y,
                                             z=z,
                                             normal_x=normal_x,
                                             normal_y=normal_y,
                                             normal_z=normal_z,
                                             h=self.rigid_body_diameter/2.,
                                             rho_b=self.rigid_body_rho,
                                             rad_s=self.rigid_body_diameter/2.,
                                             E=69. * 1e6,
                                             nu=0.3,
                                             G=69. * 1e5)
        dem_id = np.array([0, 0, 0, 0, 0])
        rigid_body_wall.add_property('dem_id', type='int', data=dem_id)
        rigid_body_wall.add_constant('no_wall', [5])
        setup_wall_dem(rigid_body_wall)

        # # remove fluid particles overlapping with the rigid body
        # G.remove_overlap_particles(
        #     fluid, rigid_body_combined, self.dx, dim=self.dim
        # )
        # # remove fluid particles overlapping with the rigid body
        # G.remove_overlap_particles(
        #     fluid, rigid_body_slave, self.dx, dim=self.dim
        # )

        # add extra output properties
        rigid_body_slave.add_output_arrays(['fx', 'fy', 'fz'])

        # Add properties to rigid body to hold the body still until some time
        add_properties(rigid_body_master, 'hold_x', 'hold_y', 'hold_z')
        rigid_body_master.hold_x[:] = rigid_body_master.x[:]
        rigid_body_master.hold_y[:] = rigid_body_master.y[:]
        rigid_body_master.hold_z[:] = rigid_body_master.z[:]

        return [rigid_body_master, rigid_body_slave, rigid_body_wall]

    def create_scheme(self):
        rb = RigidBody3DScheme(
            rigid_bodies_master=["rigid_body_combined_master"],
            rigid_bodies_slave=["rigid_body_combined_slave"],
            boundaries=["rigid_body_wall"],
            dim=3,
            gy=0.)

        s = SchemeChooser(default='rb', rb=rb)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            gz=self.gz
        )

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=self.pfreq)
        print("dt = %g"%self.dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['rigid_body_combined_slave']
        b.scalar = 'm'
        b.plot.module_manager.scalar_lut_manager.lut_mode = 'gist_yarg'
        ''')

    def post_step(self, solver):
        from pysph_dem.geometry import create_circle_of_three_layers
        t = solver.t
        dt = solver.dt
        if t > self.swell_time_delay:
            # print("inside")
            # for pa in self.particles:
            #     if pa.name == 'rigid_body_combined_master' or pa.name == 'moving_boundary':
            #         pa.y[:] += self.dx / 4.

            for pa in self.particles:
                if pa.name == 'rigid_body_combined_master':
                    master = pa

                if pa.name == 'rigid_body_combined_slave':
                    slave = pa

            for i in range(len(master.x)):
                if master.rad_s[i] < self.max_rigid_body_diameter / 2.:
                    master.rad_s[i] += self.dx / 1000.
                    master.m_b[i] = 4. / 3. * np.pi * master.rad_s[i]**3. * self.rigid_body_rho
                    master.h[:] = master.rad_s[i] * 2.

                # re-adjust the SPH particles
                if (int(t / dt) % self.swell_freq) == 0:
                    area_sphere = 4. * pi * master.rad_s[i]**2.
                    num_point = area_sphere / self.dx**2.
                    x_new, y_new, z_new = get_regular_3d_sphere(
                        master.rad_s[i], num_point)

                    left_limit = master.body_limits[2 * i]
                    slave.dx0[left_limit:left_limit+len(x_new)] = x_new[:]
                    slave.dy0[left_limit:left_limit+len(x_new)] = y_new[:]
                    slave.dz0[left_limit:left_limit+len(x_new)] = z_new[:]

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        files = files[:]
        t, total_energy = [], []
        x, y = [], []
        fx, fy, fz = [], [], []
        R = []
        ang_mom = []
        for sd, body in iter_output(files, 'rigid_body_combined_master'):
            _t = sd['t']
            # print(_t)
            t.append(_t)
            fz.append(body.fz[0])
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
        plt.plot(t, fz, "-", label='Force in z')

        plt.xlabel('t')
        plt.ylabel('Force z')
        plt.legend()
        fig = os.path.join(self.output_dir, "fz_vs_t.png")
        plt.savefig(fig, dpi=300)
        # plt.show()

        # plt.plot(x, y, label='Simulated')
        # plt.show()


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
