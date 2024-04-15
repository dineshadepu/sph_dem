import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from pysph.base.kernels import (QuinticSpline)
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

# from pysph.sph.rigid_body import (BodyForce)

from pysph.base.kernels import (QuinticSpline)

from pysph.sph.wc.gtvf import GTVFIntegrator
from sph_dem.rigid_body.math import (normalize_R_orientation, find_transpose)
from sph_dem.rigid_body.compute_rigid_body_properties import (add_properties_stride,
                                           set_total_mass,
                                           set_center_of_mass,
                                           set_moment_of_inertia_and_its_inverse,
                                           set_body_frame_position_vectors,
                                           set_body_frame_normal_vectors)
from pysph.sph.wc.linalg import mat_vec_mult, mat_mult
from pysph.examples.solid_mech.impact import add_properties

# compute the boundary particles
from sph_dem.rigid_body.boundary_particles import (get_boundary_identification_etvf_equations,
                                                     add_boundary_identification_properties)

from sph_dem.dem import (
    UpdateTangentialContacts,
    BodyForce,
    SSHertzContactForce,
    SSDMTContactForce,
    SWHertzContactForce)
from pysph_dem.swelling import (ComputeSwelling)

from sph_dem.rigid_body.rigid_body_3d_combined import (SumUpExternalForcesCombined,
                                                       ResetForceRigidBody,
                                                       RBRBCombinedContactForce,
                                                       RBWCombinedContactForce,
                                                       GTVFRigidBody3DCombinedStep)

from numpy import sqrt, log
from math import pi
from numpy import sin, cos


def set_body_limits(pa):
    nb = int(np.max(pa.body_id) + 1)
    body_limits = []
    for i in range(nb):
        limits = np.where(pa.body_id == i)
        body_limits.append(limits[0][0])
        # We add 1 to the limit so that the range operation
        # is done with regard to python's range
        body_limits.append(limits[0][-1]+1)
    pa.add_constant('body_limits', np.asarray(body_limits, dtype=int))


def setup_rigid_body(pa, dim, total_no_of_walls=3):
    add_properties(pa, 'fx', 'fy', 'fz', 'dx0', 'dy0', 'dz0', 'rad_s')

    add_properties(pa, 'G')
    pa.G[:] = pa.E[:] / (2. * (1. + pa.nu[:]))

    # for interaction with the rigid wall particle array
    pa.add_constant('max_no_walls', [total_no_of_walls])

    nb = int(np.max(pa.body_id) + 1)

    # dem_id = props.pop('dem_id', None)

    consts = {
        'total_mass':
        np.zeros(nb, dtype=float),
        'xcm':
        np.zeros(3 * nb, dtype=float),
        'xcm0':
        np.zeros(3 * nb, dtype=float),
        'R': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
        'R0': [1., 0., 0., 0., 1., 0., 0., 0., 1.] * nb,
        # moment of inertia izz (this is only for 2d)
        'izz':
        np.zeros(nb, dtype=float),
        # moment of inertia inverse in body frame
        'inertia_tensor_body_frame':
        np.zeros(9 * nb, dtype=float),
        # moment of inertia inverse in body frame
        'inertia_tensor_inverse_body_frame':
        np.zeros(9 * nb, dtype=float),
        # moment of inertia inverse in body frame
        'inertia_tensor_global_frame':
        np.zeros(9 * nb, dtype=float),
        # moment of inertia inverse in body frame
        'inertia_tensor_inverse_global_frame':
        np.zeros(9 * nb, dtype=float),
        # total force at the center of mass
        'force':
        np.zeros(3 * nb, dtype=float),
        # torque about the center of mass
        'torque':
        np.zeros(3 * nb, dtype=float),
        # velocity, acceleration of CM.
        'vcm':
        np.zeros(3 * nb, dtype=float),
        'vcm0':
        np.zeros(3 * nb, dtype=float),
        # acceleration of CM.
        'acm':
        np.zeros(3 * nb, dtype=float),
        # angular momentum
        'ang_mom':
        np.zeros(3 * nb, dtype=float),
        'ang_mom0':
        np.zeros(3 * nb, dtype=float),
        # angular velocity in global frame
        'omega':
        np.zeros(3 * nb, dtype=float),
        'omega0':
        np.zeros(3 * nb, dtype=float),
        # angular acceleration in global frame
        'ang_acc':
        np.zeros(3 * nb, dtype=float),
        'nb':
        nb
    }

    for key, elem in consts.items():
        pa.add_constant(key, elem)

    # compute the properties of the body
    set_total_mass(pa)
    set_center_of_mass(pa)

    # this function will compute
    # inertia_tensor_body_frame
    # inertia_tensor_inverse_body_frame
    # inertia_tensor_global_frame
    # inertia_tensor_inverse_global_frame
    # of the rigid body
    set_moment_of_inertia_and_its_inverse(pa)

    set_body_frame_position_vectors(pa)

    ####################################################
    # compute the boundary particles of the rigid body #
    ####################################################
    add_boundary_identification_properties(pa)
    # make sure your rho is not zero
    equations = get_boundary_identification_etvf_equations([pa.name],
                                                           [pa.name])
    # print(equations)

    sph_eval = SPHEvaluator(arrays=[pa],
                            equations=equations,
                            dim=dim,
                            kernel=QuinticSpline(dim=dim))

    sph_eval.evaluate(dt=0.1)

    # make normals of particle other than boundary particle as zero
    # for i in range(len(pa.x)):
    #     if pa.is_boundary[i] == 0:
    #         pa.normal[3 * i] = 0.
    #         pa.normal[3 * i + 1] = 0.
    #         pa.normal[3 * i + 2] = 0.

    # normal vectors in terms of body frame
    set_body_frame_normal_vectors(pa)

    # set the body limits
    set_body_limits(pa)

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])


def set_particle_velocities(pa):
    for i in range(max(pa.body_id) + 1):
        fltr = np.where(pa.body_id == i)
        bid = i
        i9 = 9 * bid
        i3 = 3 * bid

        for j in fltr:
            dx = (pa.R[i9 + 0] * pa.dx0[j] + pa.R[i9 + 1] * pa.dy0[j] +
                    pa.R[i9 + 2] * pa.dz0[j])
            dy = (pa.R[i9 + 3] * pa.dx0[j] + pa.R[i9 + 4] * pa.dy0[j] +
                    pa.R[i9 + 5] * pa.dz0[j])
            dz = (pa.R[i9 + 6] * pa.dx0[j] + pa.R[i9 + 7] * pa.dy0[j] +
                    pa.R[i9 + 8] * pa.dz0[j])

            du = pa.omega[i3 + 1] * dz - pa.omega[i3 + 2] * dy
            dv = pa.omega[i3 + 2] * dx - pa.omega[i3 + 0] * dz
            dw = pa.omega[i3 + 0] * dy - pa.omega[i3 + 1] * dx

            pa.u[j] = pa.vcm[i3 + 0] + du
            pa.v[j] = pa.vcm[i3 + 1] + dv
            pa.w[j] = pa.vcm[i3 + 2] + dw


def set_linear_velocity(pa, linear_vel):
    pa.vcm[:] = linear_vel

    set_particle_velocities(pa)


def set_angular_velocity(pa, angular_vel):
    pa.omega[:] = angular_vel[:]

    # set the angular momentum
    for i in range(max(pa.body_id) + 1):
        i9 = 9 * i
        i3 = 3 * i
        pa.ang_mom[i3:i3 + 3] = np.matmul(
            pa.inertia_tensor_global_frame[i9:i9 + 9].reshape(3, 3),
            pa.omega[i3:i3 + 3])[:]

    set_particle_velocities(pa)


def get_master_and_slave_rb(body):
    # rotational components
    # set wdeltap to -1. Which defaults to no self correction
    master = get_particle_array(m=body.total_mass, h=body.h[0], name=body.name+"_master")

    add_properties(master, 'total_mass', 'omega_x', 'omega_y', 'omega_z',
                   'ang_mom_x', 'ang_mom_y', 'ang_mom_z', 'fx', 'fy', 'fz',
                   'torque_x', 'torque_y', 'torque_z', 'E', 'nu', 'rad_s', 'm_b')

    add_properties_stride(master, 9,
                          'R',
                          'inertia_tensor_body_frame',
                          'inertia_tensor_inverse_body_frame',
                          'inertia_tensor_global_frame',
                          'inertia_tensor_inverse_global_frame')

    master.add_property('body_limits', stride=2, type='int')
    master.body_limits[:] = body.body_limits[:]

    # copy properties from body to master
    master.m[:] = body.total_mass[:]
    master.m_b[:] = master.m[:]
    master.x[:] = body.xcm[::3]
    master.y[:] = body.xcm[1::3]
    master.z[:] = body.xcm[2::3]
    master.u[:] = body.vcm[::3]
    master.v[:] = body.vcm[1::3]
    master.w[:] = body.vcm[2::3]
    master.omega_x[:] = body.omega[::3]
    master.omega_y[:] = body.omega[1::3]
    master.omega_z[:] = body.omega[2::3]
    master.ang_mom_x[:] = body.ang_mom[::3]
    master.ang_mom_y[:] = body.ang_mom[1::3]
    master.ang_mom_z[:] = body.ang_mom[2::3]
    master.E[:] = body.E[0]
    master.nu[:] = body.nu[0]
    master.add_property('dem_id', type='int')
    # TODO: set the dem_id properly
    master.dem_id[:] = 0

    master.R[:] = body.R[:]
    master.inertia_tensor_body_frame[:] = body.inertia_tensor_body_frame[:]
    master.inertia_tensor_inverse_body_frame[:] = body.inertia_tensor_inverse_body_frame[:]
    master.inertia_tensor_global_frame[:] = body.inertia_tensor_global_frame[:]
    master.inertia_tensor_inverse_global_frame[:] = body.inertia_tensor_inverse_global_frame[:]
    master.add_output_arrays(['omega_x', 'omega_y', 'omega_z'])

    slave = get_particle_array(m=body.m,
                               x=body.x,
                               y=body.y,
                               z=body.z,
                               u=body.u,
                               v=body.v,
                               w=body.w,
                               E=body.E,
                               G=body.G,
                               nu=body.nu,
                               rad_s=body.rad_s,
                               dx0=body.dx0,
                               dy0=body.dy0,
                               dz0=body.dz0,
                               h=body.h,
                               rho=body.rho,
                               fx=0.,
                               fy=0.,
                               fz=0.,
                               name=body.name+"_slave")
    add_boundary_identification_properties(slave)

    slave.normal[:] = body.normal[:]
    slave.normal0[:] = body.normal0[:]
    slave.normal_tmp[:] = body.normal_tmp[:]
    slave.normal_norm[:] = body.normal_norm[:]

    slave.add_property('is_boundary', type='int', data=body.is_boundary)
    slave.add_property('body_id', type='int', data=body.body_id)
    slave.add_property('dem_id', type='int', data=body.dem_id)
    slave.add_constant('total_mass', body.total_mass)
    slave.add_constant('max_no_walls', [body.max_no_walls[0]])

    return master, slave


def add_contact_properties_body_master(pa, max_no_tng_contacts_limit,
                                       total_no_of_walls):
    add_properties(pa, 'fx', 'fy', 'fz', 'torque_x', 'torque_y', 'torque_z')
    add_properties(pa, 'G')
    pa.G[:] = pa.E[:] / (2. * (1. + pa.nu[:]))

    add_properties_stride(pa, max_no_tng_contacts_limit,
                          'tng_ss_x', 'tng_ss_y', 'tng_ss_z',
                          'ss_fn', 'ss_overlap',
                          'fn_sw', 'overlap_sw'
                          )

    add_properties_stride(pa, total_no_of_walls,
                          'tng_sw_x', 'tng_sw_y', 'tng_sw_z')
    pa.add_constant('max_no_walls', [total_no_of_walls])

    pa.add_constant('max_no_tng_contacts_limit', [max_no_tng_contacts_limit])
    pa.add_property('tng_idx', stride=max_no_tng_contacts_limit, type="int")
    pa.add_property('total_no_tng_contacts', type="int")
    pa.total_no_tng_contacts[:] = 0
    pa.tng_idx[:] = -1
    pa.add_property('tng_idx_dem_id', stride=max_no_tng_contacts_limit, type="int")
    pa.tng_idx_dem_id[:] = -1

    # default property arrays to save out.
    pa.add_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p',
        'fx', 'fy', 'fz', 'torque_x', 'torque_y', 'torque_z',
        'omega_x', 'omega_y', 'omega_z',
        'ss_fn', 'ss_overlap',
        'fn_sw', 'overlap_sw',
        'tng_sw_x', 'tng_sw_y', 'tng_sw_z',
        'tng_ss_x', 'tng_ss_y', 'tng_ss_z'
    ])


class RBStirrerForce(Equation):
    def __init__(self, dest, sources, en, fric_coeff):
        self.en = en
        self.fric_coeff = fric_coeff
        super(RBStirrerForce, self).__init__(dest, sources)

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
            tmp_2 = 0.
            m_eff = 1. / (tmp_1 + tmp_2)
            eta_n = -2. * (5./6.)**0.5 * beta * (S_n * m_eff)**0.5

            # normal force with conservative and dissipation part
            fn_x = -kn * overlap * nij_x - eta_n * vijn_x
            fn_y = -kn * overlap * nij_y - eta_n * vijn_y
            fn_z = -kn * overlap * nij_z - eta_n * vijn_z
            # fn_x = -kn * overlap * nij_x
            # fn_y = -kn * overlap * nij_y
            # fn_z = -kn * overlap * nij_z

            d_fx[d_idx] += fn_x
            d_fy[d_idx] += fn_y
            d_fz[d_idx] += fn_z


class GTVFRigidBody3DMasterStep(IntegratorStep):
    def _get_helpers_(self):
        return [mat_vec_mult, mat_mult, find_transpose,
                normalize_R_orientation]

    def stage1(self, d_idx, d_u, d_v, d_w, d_m,
               d_fx, d_fy, d_fz,
               d_ang_mom_x,
               d_ang_mom_y,
               d_ang_mom_z,
               d_torque_x,
               d_torque_y,
               d_torque_z,
               d_inertia_tensor_inverse_global_frame,
               d_omega_x,
               d_omega_y,
               d_omega_z,
               d_au,
               d_av,
               d_aw,
               dt):
        i, didx9 = declare('int', 2)
        ang_mom, omega = declare('matrix(3)', 2)
        moi = declare('matrix(9)', 1)
        dtb2 = dt / 2.
        didx9 = 9 * d_idx
        # using velocity at t, move position
        # to t + dt/2.
        d_au[d_idx] = d_fx[d_idx] / d_m[d_idx]
        d_av[d_idx] = d_fy[d_idx] / d_m[d_idx]
        d_aw[d_idx] = d_fz[d_idx] / d_m[d_idx]
        d_u[d_idx] = d_u[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_aw[d_idx]

        # move angular velocity to t + dt/2.
        # omega_dot is
        d_ang_mom_x[d_idx] = d_ang_mom_x[d_idx] + (dtb2 * d_torque_x[d_idx])
        d_ang_mom_y[d_idx] = d_ang_mom_y[d_idx] + (dtb2 * d_torque_y[d_idx])
        d_ang_mom_z[d_idx] = d_ang_mom_z[d_idx] + (dtb2 * d_torque_z[d_idx])

        # ang_mom = declare (3)
        ang_mom[0] = d_ang_mom_x[d_idx]
        ang_mom[1] = d_ang_mom_y[d_idx]
        ang_mom[2] = d_ang_mom_z[d_idx]

        for i in range(9):
            moi[i] = d_inertia_tensor_inverse_global_frame[didx9 + i]

        # mat_vec_mult(moi, ang_mom, 3, omega)

        # manually multiply
        omega[0] = moi[0] * ang_mom[0] + moi[1] * ang_mom[1] + moi[2] * ang_mom[2]
        omega[1] = moi[3] * ang_mom[0] + moi[4] * ang_mom[1] + moi[5] * ang_mom[2]
        omega[2] = moi[6] * ang_mom[0] + moi[7] * ang_mom[1] + moi[8] * ang_mom[2]

        d_omega_x[d_idx] = omega[0]
        d_omega_y[d_idx] = omega[1]
        d_omega_z[d_idx] = omega[2]

    def stage2(self, d_idx, d_u, d_v, d_w, d_m,
               d_x, d_y, d_z,
               d_fx, d_fy, d_fz,
               d_ang_mom_x,
               d_ang_mom_y,
               d_ang_mom_z,
               d_torque_x,
               d_torque_y,
               d_torque_z,
               d_inertia_tensor_inverse_global_frame,
               d_inertia_tensor_inverse_body_frame,
               d_omega_x,
               d_omega_y,
               d_omega_z,
               d_R,
               dt):
        didx9, i = declare('int', 3)
        R, R_t, R_dot, omega_mat = declare('matrix(9)', 4)
        tmp_moi_inv, new_moi, R_moi = declare('matrix(9)', 3)
        a1, a2, a3, b1, b2, b3 = declare('matrix(3)', 6)
        didx9 = 9 * d_idx

        # Load the matrices
        for i in range(9):
            R[i] = d_R[didx9 + i]

        # using velocity at t, move position
        # to t + dt/2.
        d_x[d_idx] = d_x[d_idx] + dt * d_u[d_idx]
        d_y[d_idx] = d_y[d_idx] + dt * d_v[d_idx]
        d_z[d_idx] = d_z[d_idx] + dt * d_w[d_idx]

        # angular velocity in terms of matrix
        omega_mat[0] = 0.
        omega_mat[1] = -d_omega_z[d_idx]
        omega_mat[2] = d_omega_y[d_idx]
        omega_mat[3] = d_omega_z[d_idx]
        omega_mat[4] = 0.
        omega_mat[5] = -d_omega_x[d_idx]
        omega_mat[6] = -d_omega_y[d_idx]
        omega_mat[7] = d_omega_x[d_idx]
        omega_mat[8] = 0.

        # Rate of change of orientation is
        # matrix multiply omega mat with R
        R_dot[0] = omega_mat[0] * R[0] + omega_mat[1] * R[3] + omega_mat[2] * R[6]
        R_dot[1] = omega_mat[0] * R[1] + omega_mat[1] * R[4] + omega_mat[2] * R[7]
        R_dot[2] = omega_mat[0] * R[2] + omega_mat[1] * R[5] + omega_mat[2] * R[8]

        R_dot[3] = omega_mat[3] * R[0] + omega_mat[4] * R[3] + omega_mat[5] * R[6]
        R_dot[4] = omega_mat[3] * R[1] + omega_mat[4] * R[4] + omega_mat[5] * R[7]
        R_dot[5] = omega_mat[3] * R[2] + omega_mat[4] * R[5] + omega_mat[5] * R[8]

        R_dot[6] = omega_mat[6] * R[0] + omega_mat[7] * R[3] + omega_mat[8] * R[6]
        R_dot[7] = omega_mat[6] * R[1] + omega_mat[7] * R[4] + omega_mat[8] * R[7]
        R_dot[8] = omega_mat[6] * R[2] + omega_mat[7] * R[5] + omega_mat[8] * R[8]
        # mat_mult(omega_mat, R, 3, R_dot)

        # update the orientation to next time step
        for i in range(9):
            d_R[didx9 + i] = R[i] + R_dot[i] * dt

        # reload the matrix locally
        for i in range(9):
            R[i] = d_R[didx9 + i]
        # ====================================================
        # normalize the orientation using Gram Schmidt process
        # ====================================================
        # normalize_R_orientation(R)
        a1[0] = R[0]
        a1[1] = R[3]
        a1[2] = R[6]

        a2[0] = R[1]
        a2[1] = R[4]
        a2[2] = R[7]

        a3[0] = R[2]
        a3[1] = R[5]
        a3[2] = R[8]

        # norm of col0
        na1 = (a1[0]**2. + a1[1]**2. + a1[2]**2.)**0.5
        if na1 > 1e-12:
            b1[0] = a1[0] / na1
            b1[1] = a1[1] / na1
            b1[2] = a1[2] / na1
        else:
            b1[0] = a1[0]
            b1[1] = a1[1]
            b1[2] = a1[2]

        b1_dot_a2 = b1[0] * a2[0] + b1[1] * a2[1] + b1[2] * a2[2]
        b2[0] = a2[0] - b1_dot_a2 * b1[0]
        b2[1] = a2[1] - b1_dot_a2 * b1[1]
        b2[2] = a2[2] - b1_dot_a2 * b1[2]
        nb2 = (b2[0]**2. + b2[1]**2. + b2[2]**2.)**0.5
        b2[0] = b2[0] / nb2
        b2[1] = b2[1] / nb2
        b2[2] = b2[2] / nb2
        if nb2 > 1e-12:
            b2[0] = b2[0] / nb2
            b2[1] = b2[1] / nb2
            b2[2] = b2[2] / nb2

        b1_dot_a3 = b1[0] * a3[0] + b1[1] * a3[1] + b1[2] * a3[2]
        b2_dot_a3 = b2[0] * a3[0] + b2[1] * a3[1] + b2[2] * a3[2]
        b3[0] = a3[0] - b1_dot_a3 * b1[0] - b2_dot_a3 * b2[0]
        b3[1] = a3[1] - b1_dot_a3 * b1[1] - b2_dot_a3 * b2[1]
        b3[2] = a3[2] - b1_dot_a3 * b1[2] - b2_dot_a3 * b2[2]
        nb3 = (b3[0]**2. + b3[1]**2. + b3[2]**2.)**0.5
        if nb3 > 1e-12:
            b3[0] = b3[0] / nb3
            b3[1] = b3[1] / nb3
            b3[2] = b3[2] / nb3

        R[0] = b1[0]
        R[3] = b1[1]
        R[6] = b1[2]
        R[1] = b2[0]
        R[4] = b2[1]
        R[7] = b2[2]
        R[2] = b3[0]
        R[5] = b3[1]
        R[8] = b3[2]
        # =========================================================
        # normalize the orientation using Gram Schmidt process ends
        # =========================================================
        # copy back to particle array
        for i in range(9):
            d_R[didx9 + i] = R[i]

        # ============================
        # update the moment of inertia
        # ============================
        # find_transpose(R, R_t)
        R_t[0] = R[0]
        R_t[1] = R[3]
        R_t[2] = R[6]

        R_t[3] = R[1]
        R_t[4] = R[4]
        R_t[2] = R[6]

        R_t[6] = R[2]
        R_t[7] = R[5]
        R_t[8] = R[8]

        # =========================================
        # Moment of inertial update using functions
        # =========================================
        # # copy moi to local matrix
        # for i in range(9):
        #     tmp_moi_inv[i] = d_inertia_tensor_inverse_body_frame[didx9 + i]

        # mat_mult(
        #     R,
        #     tmp_moi_inv,
        #     3,
        #     R_moi
        # )

        # mat_mult(
        #     R_moi,
        #     R_t,
        #     3,
        #     new_moi
        # )
        # # update moi to particle array
        # for i in range(9):
        #     d_inertia_tensor_inverse_global_frame[didx9 + i] = new_moi[i]
        # ==============================================
        # Moment of inertial update using functions ends
        # ==============================================
        # =========================================
        # Moment of inertial update without functions
        # =========================================
        # copy moi to local matrix
        for i in range(9):
            tmp_moi_inv[i] = d_inertia_tensor_inverse_body_frame[didx9 + i]

        R_moi[0] = R[0] * tmp_moi_inv[0] + R[1] * tmp_moi_inv[3] + R[2] * tmp_moi_inv[6]
        R_moi[1] = R[0] * tmp_moi_inv[1] + R[1] * tmp_moi_inv[4] + R[2] * tmp_moi_inv[7]
        R_moi[2] = R[0] * tmp_moi_inv[2] + R[1] * tmp_moi_inv[5] + R[2] * tmp_moi_inv[8]

        R_moi[3] = R[3] * tmp_moi_inv[0] + R[4] * tmp_moi_inv[3] + R[5] * tmp_moi_inv[6]
        R_moi[4] = R[3] * tmp_moi_inv[1] + R[4] * tmp_moi_inv[4] + R[5] * tmp_moi_inv[7]
        R_moi[5] = R[3] * tmp_moi_inv[2] + R[4] * tmp_moi_inv[5] + R[5] * tmp_moi_inv[8]

        R_moi[6] = R[6] * tmp_moi_inv[0] + R[7] * tmp_moi_inv[3] + R[8] * tmp_moi_inv[6]
        R_moi[7] = R[6] * tmp_moi_inv[1] + R[7] * tmp_moi_inv[4] + R[8] * tmp_moi_inv[7]
        R_moi[8] = R[6] * tmp_moi_inv[2] + R[7] * tmp_moi_inv[5] + R[8] * tmp_moi_inv[8]

        new_moi[0] = R_moi[0] * R_t[0] + R_moi[1] * R_t[3] + R_moi[2] * R_t[6]
        new_moi[1] = R_moi[0] * R_t[1] + R_moi[1] * R_t[4] + R_moi[2] * R_t[7]
        new_moi[2] = R_moi[0] * R_t[2] + R_moi[1] * R_t[5] + R_moi[2] * R_t[8]

        new_moi[3] = R_moi[3] * R_t[0] + R_moi[4] * R_t[3] + R_moi[5] * R_t[6]
        new_moi[4] = R_moi[3] * R_t[1] + R_moi[4] * R_t[4] + R_moi[5] * R_t[7]
        new_moi[5] = R_moi[3] * R_t[2] + R_moi[4] * R_t[5] + R_moi[5] * R_t[8]

        new_moi[6] = R_moi[6] * R_t[0] + R_moi[7] * R_t[3] + R_moi[8] * R_t[6]
        new_moi[7] = R_moi[6] * R_t[1] + R_moi[7] * R_t[4] + R_moi[8] * R_t[7]
        new_moi[8] = R_moi[6] * R_t[2] + R_moi[7] * R_t[5] + R_moi[8] * R_t[8]

        # update moi to particle array
        for i in range(9):
            d_inertia_tensor_inverse_global_frame[didx9 + i] = new_moi[i]
        # ==============================================
        # Moment of inertial update using functions ends
        # ==============================================

    def stage3(self, d_idx, d_u, d_v, d_w, d_m,
               d_fx, d_fy, d_fz,
               d_ang_mom_x,
               d_ang_mom_y,
               d_ang_mom_z,
               d_torque_x,
               d_torque_y,
               d_torque_z,
               d_inertia_tensor_inverse_global_frame,
               d_omega_x,
               d_omega_y,
               d_omega_z,
               d_au,
               d_av,
               d_aw,
               dt):
        i, didx9 = declare('int', 2)
        ang_mom, omega = declare('matrix(3)', 2)
        moi = declare('matrix(9)', 1)
        dtb2 = dt / 2.
        didx9 = 9 * d_idx
        # using velocity at t, move position
        # to t + dt/2.
        d_au[d_idx] = d_fx[d_idx] / d_m[d_idx]
        d_av[d_idx] = d_fy[d_idx] / d_m[d_idx]
        d_aw[d_idx] = d_fz[d_idx] / d_m[d_idx]
        d_u[d_idx] = d_u[d_idx] + dtb2 * d_au[d_idx]
        d_v[d_idx] = d_v[d_idx] + dtb2 * d_av[d_idx]
        d_w[d_idx] = d_w[d_idx] + dtb2 * d_aw[d_idx]

        # move angular velocity to t + dt/2.
        # omega_dot is
        d_ang_mom_x[d_idx] = d_ang_mom_x[d_idx] + (dtb2 * d_torque_x[d_idx])
        d_ang_mom_y[d_idx] = d_ang_mom_y[d_idx] + (dtb2 * d_torque_y[d_idx])
        d_ang_mom_z[d_idx] = d_ang_mom_z[d_idx] + (dtb2 * d_torque_z[d_idx])

        # ang_mom = declare (3)
        ang_mom[0] = d_ang_mom_x[d_idx]
        ang_mom[1] = d_ang_mom_y[d_idx]
        ang_mom[2] = d_ang_mom_z[d_idx]

        for i in range(9):
            moi[i] = d_inertia_tensor_inverse_global_frame[didx9 + i]

        # mat_vec_mult(moi, ang_mom, 3, omega)

        # manually multiply
        omega[0] = moi[0] * ang_mom[0] + moi[1] * ang_mom[1] + moi[2] * ang_mom[2]
        omega[1] = moi[3] * ang_mom[0] + moi[4] * ang_mom[1] + moi[5] * ang_mom[2]
        omega[2] = moi[6] * ang_mom[0] + moi[7] * ang_mom[1] + moi[8] * ang_mom[2]

        d_omega_x[d_idx] = omega[0]
        d_omega_y[d_idx] = omega[1]
        d_omega_z[d_idx] = omega[2]


class SumUpExternalForces(Equation):
    # def initialize(self, d_idx, d_fx, d_fy, d_fz, d_torque_x, d_torque_y,
    #                d_torque_z):
    #     d_fx[d_idx] = 0.0
    #     d_fy[d_idx] = 0.0
    #     d_fz[d_idx] = 0.0
    #     d_torque_x[d_idx] = 0.0
    #     d_torque_y[d_idx] = 0.0
    #     d_torque_z[d_idx] = 0.0

    def initialize_pair(self, d_idx, d_fx, d_fy, d_fz,
                        d_torque_x, d_torque_y, d_torque_z,
                        d_body_limits, s_x,  s_y,  s_z,
                        s_fx,
                        s_fy,
                        s_fz,
                        d_x,
                        d_y,
                        d_z,
                        t, dt):
        bid, i9, i3, i2, i = declare('int', 5)
        left_limit, right_limit = declare('int', 2)
        i2 = 2 * d_idx
        left_limit = d_body_limits[i2]
        right_limit = d_body_limits[i2+1]
        for i in range(left_limit,  right_limit):
            d_fx[d_idx] += s_fx[i]
            d_fy[d_idx] += s_fy[i]
            d_fz[d_idx] += s_fz[i]

            # torque due to force on particle i
            # (r_i - com) \cross f_i
            dx = s_x[i] - d_x[d_idx]
            dy = s_y[i] - d_y[d_idx]
            dz = s_z[i] - d_z[d_idx]

            # torque due to force on particle i
            # dri \cross fi
            d_torque_x[d_idx] += (dy * s_fz[i] - dz * s_fy[i])
            d_torque_y[d_idx] += (dz * s_fx[i] - dx * s_fz[i])
            d_torque_z[d_idx] += (dx * s_fy[i] - dy * s_fx[i])


class UpdateSlaveBodyState(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_dx0, d_dy0, d_dz0,
                        d_au, d_av, d_aw, d_body_id, d_is_boundary, d_normal0, d_normal,
                        s_R, s_omega_x, s_omega_y, s_omega_z, s_u, s_v, s_w,
                        s_au, s_av, s_aw, s_x, s_y, s_z):
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
        du = s_omega_y[bid] * dz - s_omega_z[bid] * dy
        dv = s_omega_z[bid] * dx - s_omega_x[bid] * dz
        dw = s_omega_x[bid] * dy - s_omega_y[bid] * dx

        d_u[d_idx] = s_u[bid] + du
        d_v[d_idx] = s_v[bid] + dv
        d_w[d_idx] = s_w[bid] + dw

        # for particle acceleration we follow this
        # https://www.brown.edu/Departments/Engineering/Courses/En4/notes_old/RigidKinematics/rigkin.htm
        omega_omega_cross_x = s_omega_y[bid] * dw - s_omega_z[bid] * dv
        omega_omega_cross_y = s_omega_z[bid] * du - s_omega_x[bid] * dw
        omega_omega_cross_z = s_omega_x[bid] * dv - s_omega_y[bid] * du
        # TODO: Skip angular acceleration
        # ang_acc_cross_x = s_ang_acc[i3 + 1] * dz - d_ang_acc[i3 + 2] * dy
        # ang_acc_cross_y = s_ang_acc[i3 + 2] * dx - d_ang_acc[i3 + 0] * dz
        # ang_acc_cross_z = s_ang_acc[i3 + 0] * dy - d_ang_acc[i3 + 1] * dx
        d_au[d_idx] = s_au[bid] + omega_omega_cross_x
        d_av[d_idx] = s_av[bid] + omega_omega_cross_y
        d_aw[d_idx] = s_aw[bid] + omega_omega_cross_z

        # update normal vectors of the boundary
        if d_is_boundary[d_idx] == 1:
            d_normal[i3 + 0] = (s_R[i9 + 0] * d_normal0[i3 + 0] +
                                s_R[i9 + 1] * d_normal0[i3 + 1] +
                                s_R[i9 + 2] * d_normal0[i3 + 2])
            d_normal[i3 + 1] = (s_R[i9 + 3] * d_normal0[i3 + 0] +
                                s_R[i9 + 4] * d_normal0[i3 + 1] +
                                s_R[i9 + 5] * d_normal0[i3 + 2])
            d_normal[i3 + 2] = (s_R[i9 + 6] * d_normal0[i3 + 0] +
                                s_R[i9 + 7] * d_normal0[i3 + 1] +
                                s_R[i9 + 8] * d_normal0[i3 + 2])


class RigidBody3DScheme(Scheme):
    def __init__(self, rigid_bodies_master, rigid_bodies_slave, boundaries, dim,
                 kr=1e5, kf=1e5, en=1.0, gamma=0.0, fric_coeff=0.5, gx=0.0, gy=0.0, gz=0.0):
        self.rigid_bodies_master = rigid_bodies_master
        self.rigid_bodies_slave = rigid_bodies_master

        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        if rigid_bodies_slave is None:
            self.rigid_bodies_slave = []
        else:
            self.rigid_bodies_slave = rigid_bodies_slave

        # rigid body parameters
        self.dim = dim

        self.kernel = QuinticSpline

        self.integrator = "gtvf"

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kr = kr
        self.kf = kf
        self.en = en
        self.gamma = gamma
        self.fric_coeff = fric_coeff

        self.solver = None

    def add_user_options(self, group):
        # choices = ['bui', 'canelas']
        # group.add_argument("--dem",
        #                    action="store",
        #                    dest='dem',
        #                    default="bui",
        #                    choices=choices,
        #                    help="DEM interaction " % choices)

        group.add_argument("--kr-stiffness", action="store",
                           dest="kr", default=1e5,
                           type=float,
                           help="Repulsive spring stiffness")

        group.add_argument("--kf-stiffness", action="store",
                           dest="kf", default=1e3,
                           type=float,
                           help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.5,
                           type=float,
                           help="Friction coefficient")

        group.add_argument("--en", action="store",
                           dest="en", default=1.,
                           type=float,
                           help="Coefficient of restitution")

        group.add_argument("--gamma", action="store",
                           dest="gamma", default=0.0,
                           type=float,
                           help="Surface energy")

    def consume_user_options(self, options):
        _vars = ['kr', 'kf', 'fric_coeff', 'en', 'gamma']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        return self._get_gtvf_equations()

    def _get_gtvf_equations(self):
        # elastic solid equations
        all = self.rigid_bodies_master

        stage1 = []
        # ==============================
        # Stage 2 equations
        # ==============================
        if len(self.rigid_bodies_slave) > 0:
            g5 = []
            for name in self.rigid_bodies_master:
                g5.append(
                    UpdateSlaveBodyState(dest=name[:-6:]+"slave",
                                         sources=[name]))

            stage1.append(Group(equations=g5, real=False))

        stage2 = []
        #######################
        # Handle rigid bodies #
        #######################
        if len(self.rigid_bodies_slave) > 0:
            g5 = []
            for name in self.rigid_bodies_master:
                g5.append(
                    UpdateSlaveBodyState(dest=name[:-6:]+"slave",
                                         sources=[name]))

            stage2.append(Group(equations=g5, real=False))

        if len(self.rigid_bodies_master) > 0:
            g1 = []
            for body in self.rigid_bodies_master:
                g1.append(
                    # see the previous examples and write down the sources
                    UpdateTangentialContacts(
                        dest=body, sources=self.rigid_bodies_master))
            stage2.append(Group(equations=g1, real=False))

            g2 = []
            for body in self.rigid_bodies_master:
                g2.append(
                    BodyForce(dest=body,
                              sources=None,
                              gx=self.gx,
                              gy=self.gy,
                              gz=self.gz))

            for body in self.rigid_bodies_master:
                g2.append(SSHertzContactForce(dest=body,
                                              sources=self.rigid_bodies_master,
                                              en=self.en,
                                              fric_coeff=self.fric_coeff))
                g2.append(SSDMTContactForce(dest=body,
                                            sources=self.rigid_bodies_master,
                                            gamma=self.gamma))
                if len(self.boundaries) > 0:
                    g2.append(SWHertzContactForce(dest=body,
                                                sources=self.boundaries,
                                                en=self.en,
                                                fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g2, real=False))

        if len(self.rigid_bodies_slave) > 0:
            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies_master:
                g6.append(SumUpExternalForces(dest=name, sources=[name[:-6:]+"slave"]))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = QuinticSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        bodystep = GTVFRigidBody3DMasterStep()
        integrator_cls = GTVFIntegrator

        for body in self.rigid_bodies_master:
            if body not in steppers:
                steppers[body] = bodystep

        cls = integrator_cls
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def get_solver(self):
        return self.solver
