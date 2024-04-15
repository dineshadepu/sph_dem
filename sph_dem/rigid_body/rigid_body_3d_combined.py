import numpy as np

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import add_bool_argument
from pysph.sph.equation import Group, MultiStageEquations
from pysph.base.kernels import QuinticSpline
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

from pysph.sph.wc.gtvf import GTVFIntegrator
from pysph.examples.solid_mech.impact import add_properties
from numpy import sqrt, log
from math import pi

M_PI = pi

class GTVFRigidBody3DCombinedStep(IntegratorStep):
    def py_stage1(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.acm[i3 + j] = dst.force[i3 + j] / dst.total_mass[i]
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.acm[i3 + j])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

            # compute the angular acceleration
            # https://physics.stackexchange.com/questions/688426/compute-angular-acceleration-from-torque-in-3d
            omega_cross_L = np.cross(dst.omega[i3:i3 + 3],
                                     dst.ang_mom[i3:i3 + 3])
            tmp = dst.torque[i3:i3 + 3] - omega_cross_L
            dst.ang_acc[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), tmp)

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_au, d_av, d_aw, d_xcm, d_vcm, d_acm, d_ang_acc, d_R, d_omega,
               d_body_id):
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
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

        # for particle acceleration we follow this
        # https://www.brown.edu/Departments/Engineering/Courses/En4/notes_old/RigidKinematics/rigkin.htm
        omega_omega_cross_x = d_omega[i3 + 1] * dw - d_omega[i3 + 2] * dv
        omega_omega_cross_y = d_omega[i3 + 2] * du - d_omega[i3 + 0] * dw
        omega_omega_cross_z = d_omega[i3 + 0] * dv - d_omega[i3 + 1] * du
        ang_acc_cross_x = d_ang_acc[i3 + 1] * dz - d_ang_acc[i3 + 2] * dy
        ang_acc_cross_y = d_ang_acc[i3 + 2] * dx - d_ang_acc[i3 + 0] * dz
        ang_acc_cross_z = d_ang_acc[i3 + 0] * dy - d_ang_acc[i3 + 1] * dx
        d_au[d_idx] = d_acm[i3 + 0] + omega_omega_cross_x + ang_acc_cross_x
        d_av[d_idx] = d_acm[i3 + 1] + omega_omega_cross_y + ang_acc_cross_y
        d_aw[d_idx] = d_acm[i3 + 2] + omega_omega_cross_z + ang_acc_cross_z

    def py_stage2(self, dst, t, dt):
        # move positions to t + dt time step
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.xcm[i3 + j] = dst.xcm[i3 + j] + dt * dst.vcm[i3 + j]

            # angular velocity in terms of matrix
            omega_mat = np.array([[0, -dst.omega[i3 + 2], dst.omega[i3 + 1]],
                                  [dst.omega[i3 + 2], 0, -dst.omega[i3 + 0]],
                                  [-dst.omega[i3 + 1], dst.omega[i3 + 0], 0]])

            # Currently the orientation is at time t
            R = dst.R[i9:i9 + 9].reshape(3, 3)

            # Rate of change of orientation is
            r_dot = np.matmul(omega_mat, R)
            r_dot = r_dot.ravel()

            # update the orientation to next time step
            dst.R[i9:i9 + 9] = dst.R[i9:i9 + 9] + r_dot * dt

            # normalize the orientation using Gram Schmidt process
            normalize_R_orientation(dst.R[i9:i9 + 9])

            # update the moment of inertia
            R = dst.R[i9:i9 + 9].reshape(3, 3)
            R_t = R.transpose()
            tmp = np.matmul(
                R,
                dst.inertia_tensor_inverse_body_frame[i9:i9 + 9].reshape(3, 3))
            dst.inertia_tensor_inverse_global_frame[i9:i9 + 9] = (np.matmul(
                tmp, R_t)).ravel()[:]

    def stage2(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_xcm, d_vcm, d_R, d_omega, d_body_id, d_normal0, d_normal,
               d_is_boundary, d_rho, dt):
        # some variables to update the positions seamlessly
        bid, i9, i3, idx3 = declare('int', 4)
        bid = d_body_id[d_idx]
        idx3 = 3 * d_idx
        i9 = 9 * bid
        i3 = 3 * bid

        ###########################
        # Update position vectors #
        ###########################
        # rotate the position of the vector in the body frame to global frame
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        d_x[d_idx] = d_xcm[i3 + 0] + dx
        d_y[d_idx] = d_xcm[i3 + 1] + dy
        d_z[d_idx] = d_xcm[i3 + 2] + dz

        # update normal vectors of the boundary
        if d_is_boundary[d_idx] == 1:
            d_normal[idx3 + 0] = (d_R[i9 + 0] * d_normal0[idx3] +
                                  d_R[i9 + 1] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 2] * d_normal0[idx3 + 2])
            d_normal[idx3 + 1] = (d_R[i9 + 3] * d_normal0[idx3] +
                                  d_R[i9 + 4] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 5] * d_normal0[idx3 + 2])
            d_normal[idx3 + 2] = (d_R[i9 + 6] * d_normal0[idx3] +
                                  d_R[i9 + 7] * d_normal0[idx3 + 1] +
                                  d_R[i9 + 8] * d_normal0[idx3 + 2])

    def py_stage3(self, dst, t, dt):
        dtb2 = dt / 2.
        for i in range(dst.nb[0]):
            i3 = 3 * i
            i9 = 9 * i
            for j in range(3):
                # using velocity at t, move position
                # to t + dt/2.
                dst.acm[i3 + j] = dst.force[i3 + j] / dst.total_mass[i]
                dst.vcm[i3 + j] = dst.vcm[i3 + j] + (dtb2 * dst.acm[i3 + j])

            # move angular velocity to t + dt/2.
            # omega_dot is
            dst.ang_mom[i3:i3 +
                        3] = dst.ang_mom[i3:i3 + 3] + (dtb2 *
                                                       dst.torque[i3:i3 + 3])

            dst.omega[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), dst.ang_mom[i3:i3 + 3])

            # compute the angular acceleration
            # https://physics.stackexchange.com/questions/688426/compute-angular-acceleration-from-torque-in-3d
            omega_cross_L = np.cross(dst.omega[i3:i3 + 3],
                                     dst.ang_mom[i3:i3 + 3])
            tmp = dst.torque[i3:i3 + 3] - omega_cross_L
            dst.ang_acc[i3:i3 + 3] = np.matmul(
                dst.inertia_tensor_inverse_global_frame[i9:i9 + 9].reshape(
                    3, 3), tmp)

    def stage3(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
               d_au, d_av, d_aw, d_xcm, d_vcm, d_acm, d_ang_acc, d_R, d_omega,
               d_body_id, d_is_boundary):
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
        dx = (d_R[i9 + 0] * d_dx0[d_idx] + d_R[i9 + 1] * d_dy0[d_idx] +
              d_R[i9 + 2] * d_dz0[d_idx])
        dy = (d_R[i9 + 3] * d_dx0[d_idx] + d_R[i9 + 4] * d_dy0[d_idx] +
              d_R[i9 + 5] * d_dz0[d_idx])
        dz = (d_R[i9 + 6] * d_dx0[d_idx] + d_R[i9 + 7] * d_dy0[d_idx] +
              d_R[i9 + 8] * d_dz0[d_idx])

        ###########################
        # Update velocity vectors #
        ###########################
        # here du, dv, dw are velocities due to angular velocity
        # dV = omega \cross dr
        # where dr = x - cm
        du = d_omega[i3 + 1] * dz - d_omega[i3 + 2] * dy
        dv = d_omega[i3 + 2] * dx - d_omega[i3 + 0] * dz
        dw = d_omega[i3 + 0] * dy - d_omega[i3 + 1] * dx

        d_u[d_idx] = d_vcm[i3 + 0] + du
        d_v[d_idx] = d_vcm[i3 + 1] + dv
        d_w[d_idx] = d_vcm[i3 + 2] + dw

        # for particle acceleration we follow this
        # https://www.brown.edu/Departments/Engineering/Courses/En4/notes_old/RigidKinematics/rigkin.htm
        omega_omega_cross_x = d_omega[i3 + 1] * dw - d_omega[i3 + 2] * dv
        omega_omega_cross_y = d_omega[i3 + 2] * du - d_omega[i3 + 0] * dw
        omega_omega_cross_z = d_omega[i3 + 0] * dv - d_omega[i3 + 1] * du
        ang_acc_cross_x = d_ang_acc[i3 + 1] * dz - d_ang_acc[i3 + 2] * dy
        ang_acc_cross_y = d_ang_acc[i3 + 2] * dx - d_ang_acc[i3 + 0] * dz
        ang_acc_cross_z = d_ang_acc[i3 + 0] * dy - d_ang_acc[i3 + 1] * dx
        d_au[d_idx] = d_acm[i3 + 0] + omega_omega_cross_x + ang_acc_cross_x
        d_av[d_idx] = d_acm[i3 + 1] + omega_omega_cross_y + ang_acc_cross_y
        d_aw[d_idx] = d_acm[i3 + 2] + omega_omega_cross_z + ang_acc_cross_z


def normalize_R_orientation(orien):
    a1 = np.array([orien[0], orien[3], orien[6]])
    a2 = np.array([orien[1], orien[4], orien[7]])
    a3 = np.array([orien[2], orien[5], orien[8]])
    # norm of col0
    na1 = np.linalg.norm(a1)

    b1 = a1 / na1

    b2 = a2 - np.dot(b1, a2) * b1
    nb2 = np.linalg.norm(b2)
    b2 = b2 / nb2

    b3 = a3 - np.dot(b1, a3) * b1 - np.dot(b2, a3) * b2
    nb3 = np.linalg.norm(b3)
    b3 = b3 / nb3

    orien[0] = b1[0]
    orien[3] = b1[1]
    orien[6] = b1[2]
    orien[1] = b2[0]
    orien[4] = b2[1]
    orien[7] = b2[2]
    orien[2] = b3[0]
    orien[5] = b3[1]
    orien[8] = b3[2]

class SumUpExternalForcesCombined(Equation):
    def __init__(self, dest, sources,
                 gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(SumUpExternalForcesCombined, self).__init__(dest, sources)

    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        dx0 = declare('object')
        dy0 = declare('object')
        dz0 = declare('object')
        xcm = declare('object')
        R = declare('object')
        total_mass = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')
        i9 = declare('int')

        frc = dst.force
        trq = dst.torque
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        dx0 = dst.dx0
        dy0 = dst.dy0
        dz0 = dst.dz0
        xcm = dst.xcm
        R = dst.R
        total_mass = dst.total_mass
        body_id = dst.body_id

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            i9 = 9 * i
            frc[i3] += fx[j]
            frc[i3 + 1] += fy[j]
            frc[i3 + 2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i

            # get the local vector from particle to center of mass
            dx = (R[i9 + 0] * dx0[j] + R[i9 + 1] * dy0[j] +
                  R[i9 + 2] * dz0[j])
            dy = (R[i9 + 3] * dx0[j] + R[i9 + 4] * dy0[j] +
                  R[i9 + 5] * dz0[j])
            dz = (R[i9 + 6] * dx0[j] + R[i9 + 7] * dy0[j] +
                  R[i9 + 8] * dz0[j])

            # dx = x[j] - xcm[i3]
            # dy = y[j] - xcm[i3 + 1]
            # dz = z[j] - xcm[i3 + 2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3 + 1] += (dz * fx[j] - dx * fz[j])
            trq[i3 + 2] += (dx * fy[j] - dy * fx[j])

        # add body force
        for i in range(max(body_id) + 1):
            i3 = 3 * i
            frc[i3] += total_mass[i] * self.gx
            frc[i3 + 1] += total_mass[i] * self.gy
            frc[i3 + 2] += total_mass[i] * self.gz


class ResetForceRigidBody(Equation):
    def __init__(self, dest, sources):
        super(ResetForceRigidBody, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.


class RBRBCombinedContactForce(Equation):
    def __init__(self, dest, sources, en, fric_coeff):
        self.en = en
        self.fric_coeff = fric_coeff
        super(RBRBCombinedContactForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, d_rad_s,
             s_idx, s_rad_s, d_dem_id, s_dem_id,
             d_nu, s_nu, d_E, s_E, d_G, s_G,
             d_m, s_m,
             d_body_id,
             s_total_mass,
             s_body_id,
             XIJ, RIJ, R2IJ, VIJ):
        overlap = 0
        if d_dem_id[d_idx] != s_dem_id[s_idx]:
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

            # Compute stiffness
            # effective Young's modulus
            tmp_1 = (1. - d_nu[d_idx]**2.) / d_E[d_idx]
            tmp_2 = (1. - s_nu[s_idx]**2.) / s_E[s_idx]
            E_eff = 1. / (tmp_1 + tmp_2)
            tmp_1 = 1. / d_rad_s[d_idx]
            tmp_2 = 1. / s_rad_s[s_idx]
            R_eff = 1. / (tmp_1 + tmp_2)
            # Eq 4 [1]
            kn = 4. / 3. * E_eff * R_eff**0.5

            # compute damping coefficient
            tmp_1 = log(self.en)
            tmp_2 = tmp_1**2. + pi**2.
            beta = tmp_1 / (tmp_2)**0.5
            S_n = 2. * E_eff * (R_eff * overlap)**0.5
            tmp_1 = 1. / d_total_mass[d_body_id[d_idx]]
            tmp_2 = 1. / s_total_mass[s_body_id[s_idx]]
            m_eff = 1. / (tmp_1 + tmp_2)
            eta_n = -2. * (5./6.)**0.5 * beta * (S_n * m_eff)**0.5
            # normal force with conservative and dissipation part
            fn_x = -kn * overlap * nij_x - eta_n * vijn_x
            fn_y = -kn * overlap * nij_y - eta_n * vijn_y
            fn_z = -kn * overlap * nij_z - eta_n * vijn_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class RBWCombinedContactForce(Equation):
    def __init__(self, dest, sources, en, fric_coeff):
        self.en = en
        self.fric_coeff = fric_coeff
        super(RBWCombinedContactForce, self).__init__(dest, sources)

    def initialize_pair(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass,
                        s_idx,
                        d_x, d_y, d_z,
                        d_u, d_v, d_w,
                        s_x, s_y, s_z,
                        d_nu, s_nu, d_E, s_E, d_G, s_G,
                        d_m,
                        s_no_wall,
                        s_normal_x, s_normal_y, s_normal_z,
                        d_rad_s,
                        d_body_id,
                        d_max_no_walls):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        no_wall, i, max_no_of_walls = declare('int', 3)
        overlap = -1.

        # get the number of walls available
        no_wall = s_no_wall[0]
        max_no_of_walls = d_max_no_walls[0]

        for i in range(no_wall):
            dx_ij = d_x[d_idx] - s_x[i]
            dy_ij = d_y[d_idx] - s_y[i]
            dz_ij = d_z[d_idx] - s_z[i]
            # rij = (dx_ij**2. + dy_ij**2. + dz_ij**2.)**0.5
            rij = (s_normal_x[i] * dx_ij + s_normal_y[i] * dy_ij + s_normal_z[i] * dz_ij)
            overlap = d_rad_s[d_idx] - rij

            if overlap > 0:
                # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
                # rinv = 1.0 / RIJ
                # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
                # but we use wall normal
                nx = s_normal_x[i]
                ny = s_normal_y[i]
                nz = s_normal_z[i]

                # ---- Relative velocity computation (Eq 2.9) ----
                # relative velocity of particle d_idx w.r.t particle s_idx at
                # contact point. The velocity difference provided by PySPH is
                # only between translational velocities, but we need to
                # consider rotational velocities also.
                # Distance till contact point
                a_i = d_rad_s[d_idx] - overlap
                # a_j = s_rad_s[s_idx] - overlap / 2.

                # velocity of particle i at the contact point
                vi_x = d_u[d_idx]
                vi_y = d_v[d_idx]
                vi_z = d_w[d_idx]

                # just flip the normal and compute the angular velocity
                # contribution
                # vj_x = s_u[s_idx] + (-s_omega_y[s_idx] * nz + s_omega_z[s_idx] * ny) * a_j
                # vj_y = s_v[s_idx] + (-s_omega_z[s_idx] * nx + s_omega_x[s_idx] * nz) * a_j
                # vj_z = s_w[s_idx] + (-s_omega_x[s_idx] * ny + s_omega_y[s_idx] * nx) * a_j
                # Make it zero for the wall
                vj_x = 0.
                vj_y = 0.
                vj_z = 0.

                # Now the relative velocity of particle i w.r.t j at the contact
                # point is
                vij_x = vi_x - vj_x
                vij_y = vi_y - vj_y
                vij_z = vi_z - vj_z

                # normal velocity magnitude
                vij_dot_nij = vij_x * nx + vij_y * ny + vij_z * nz
                vn_x = vij_dot_nij * nx
                vn_y = vij_dot_nij * ny
                vn_z = vij_dot_nij * nz

                ############################
                # normal force computation #
                ############################
                # Compute stiffness
                # effective Young's modulus
                tmp_1 = (1. - d_nu[d_idx]**2.) / d_E[d_idx]
                tmp_2 = (1. - s_nu[i]**2.) / s_E[i]
                E_eff = 1. / (tmp_1 + tmp_2)
                tmp_1 = 1. / d_rad_s[d_idx]
                tmp_2 = 0.
                # tmp_2 = 1. / s_rad_s[s_idx]
                R_eff = 1. / (tmp_1 + tmp_2)
                # Eq 4 [1]
                kn = 4. / 3. * E_eff * R_eff**0.5

                # compute damping coefficient
                tmp_1 = log(self.en)
                tmp_2 = tmp_1**2. + pi**2.
                beta = tmp_1 / (tmp_2)**0.5
                S_n = 2. * E_eff * (R_eff * overlap)**0.5
                tmp_1 = 1. / d_total_mass[d_body_id[d_idx]]
                tmp_2 = 0.
                m_eff = 1. / (tmp_1 + tmp_2)
                eta_n = -2. * (5./6.)**0.5 * beta * (S_n * m_eff)**0.5

                fn = kn * overlap**1.5
                fn_x = fn * nx - eta_n * vn_x
                fn_y = fn * ny - eta_n * vn_y
                fn_z = fn * nz - eta_n * vn_z

                d_fx[d_idx] += fn_x
                d_fy[d_idx] += fn_y
                d_fz[d_idx] += fn_z
