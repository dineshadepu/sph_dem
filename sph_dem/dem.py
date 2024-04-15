"""Papers for reference:

1. Smoothed particle hydrodynamics modeling of granular column collapse
https://doi.org/10.1007/s10035-016-0684-3 for the benchmarks (2d column
collapse)

"""
import numpy as np

from pysph.sph.scheme import add_bool_argument

from pysph.sph.equation import Equation
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.equation import Group, MultiStageEquations
from pysph.sph.integrator import EPECIntegrator
from textwrap import dedent
from compyle.api import declare

from pysph.base.utils import get_particle_array
from pysph.sph.integrator_step import IntegratorStep

from pysph.base.kernels import (CubicSpline, QuinticSpline)

from numpy import sqrt, log
from math import pi

# constants
M_PI = pi

from pysph.examples.solid_mech.impact import add_properties

from sph_dem.rigid_body.compute_rigid_body_properties import add_properties_stride
from pysph_dem.swelling import (ComputeSwelling)


def setup_dem_particles(pa,
                        max_no_tng_contacts_limit,
                        total_no_of_walls=6):
    """
    Define the arguments

    max_no_tng_contacts_limit: Maximum no of contacts a particle can possibly make
    """
    add_properties(pa, 'fx', 'fy', 'fz', 'torque_x', 'torque_y', 'torque_z')
    add_properties(pa, 'omega_x', 'omega_y', 'omega_z')
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
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p',
        'fx', 'fy', 'fz', 'torque_x', 'torque_y', 'torque_z',
        'omega_x', 'omega_y', 'omega_z',
        'ss_fn', 'ss_overlap',
        'fn_sw', 'overlap_sw',
        'tng_idx',
        'tng_sw_x', 'tng_sw_y', 'tng_sw_z',
        'tng_ss_x', 'tng_ss_y', 'tng_ss_z'
    ])


def setup_wall_dem(pa):
    add_properties(pa, 'omega_x', 'omega_y', 'omega_z')
    pa.G[:] = pa.E[:] / (2. * (1. + pa.nu[:]))


class DEMStep(IntegratorStep):
    def stage1(self, d_idx, d_m_b, d_I_inverse, d_u, d_v, d_w, d_omega_x,
               d_omega_y, d_omega_z, d_fx, d_fy, d_fz, d_torque_x, d_torque_y,
               d_torque_z, dt):
        dtb2 = 0.5 * dt
        m_inverse = 1. / d_m_b[d_idx]
        d_u[d_idx] += dtb2 * d_fx[d_idx] * m_inverse
        d_v[d_idx] += dtb2 * d_fy[d_idx] * m_inverse
        d_w[d_idx] += dtb2 * d_fz[d_idx] * m_inverse

        I_inverse = d_I_inverse[d_idx]
        d_omega_x[d_idx] += dtb2 * d_torque_x[d_idx] * I_inverse
        d_omega_y[d_idx] += dtb2 * d_torque_y[d_idx] * I_inverse
        d_omega_z[d_idx] += dtb2 * d_torque_z[d_idx] * I_inverse

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, dt):
        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

    def stage3(self, d_idx, d_m_b, d_I_inverse, d_u, d_v, d_w, d_omega_x,
               d_omega_y, d_omega_z, d_fx, d_fy, d_fz, d_torque_x, d_torque_y,
               d_torque_z, dt):
        dtb2 = 0.5 * dt
        m_inverse = 1. / d_m_b[d_idx]
        d_u[d_idx] += dtb2 * d_fx[d_idx] * m_inverse
        d_v[d_idx] += dtb2 * d_fy[d_idx] * m_inverse
        d_w[d_idx] += dtb2 * d_fz[d_idx] * m_inverse

        I_inverse = d_I_inverse[d_idx]
        d_omega_x[d_idx] += dtb2 * d_torque_x[d_idx] * I_inverse
        d_omega_y[d_idx] += dtb2 * d_torque_y[d_idx] * I_inverse
        d_omega_z[d_idx] += dtb2 * d_torque_z[d_idx] * I_inverse


class BodyForce(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(BodyForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_b, d_fx, d_fy, d_fz,
                   d_torque_x,
                   d_torque_y,
                   d_torque_z):
        d_fx[d_idx] = d_m_b[d_idx] * self.gx
        d_fy[d_idx] = d_m_b[d_idx] * self.gy
        d_fz[d_idx] = d_m_b[d_idx] * self.gz
        d_torque_x[d_idx] = 0.
        d_torque_y[d_idx] = 0.
        d_torque_z[d_idx] = 0.


class SWHertzContactForce(Equation):
    """
    Hertz Sphere and Wall

    From TODO
    """
    def __init__(self, dest, sources, en, fric_coeff):
        self.en = en
        self.fric_coeff = fric_coeff
        super(SWHertzContactForce, self).__init__(dest, sources)

    def initialize_pair(self, d_idx, d_m_b, d_u, d_v, d_w, d_omega_x, d_omega_y,
                        d_omega_z,
                        d_fx,
                        d_fy,
                        d_fz,
                        d_tng_sw_x,
                        d_tng_sw_y,
                        d_tng_sw_z,
                        d_torque_x,
                        d_torque_y,
                        d_torque_z,
                        d_fn_sw,
                        d_overlap_sw,
                        d_rad_s,
                        d_max_no_walls,
                        d_x,
                        d_y,
                        d_z,
                        s_x,
                        s_y,
                        s_z,
                        s_no_wall,
                        s_normal_x,
                        s_normal_y,
                        s_normal_z,
                        s_m, s_u, s_v, s_w, s_omega_x, s_omega_y, s_omega_z, s_rad_s,
                        d_nu, s_nu, d_E, s_E, d_G, s_G,
                        dt, t):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        no_wall, i, max_no_of_walls = declare('int', 3)
        overlap = -1.

        # get the number of walls available
        no_wall = s_no_wall[0]
        max_no_of_walls = d_max_no_walls[0]

        # loop over all the walls and compute the force
        for i in range(no_wall):
            dx_ij = d_x[d_idx] - s_x[i]
            dy_ij = d_y[d_idx] - s_y[i]
            dz_ij = d_z[d_idx] - s_z[i]
            # rij = (dx_ij**2. + dy_ij**2. + dz_ij**2.)**0.5
            rij = (s_normal_x[i] * dx_ij + s_normal_y[i] * dy_ij + s_normal_z[i] * dz_ij)
            overlap = d_rad_s[d_idx] - rij

            # ---------- force computation starts ------------
            # We save (saved) the tangential at the following index place
            found_at = d_idx * max_no_of_walls + i
            # print(found_at, "found at")
            # if particles are overlapping
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
                vi_x = d_u[d_idx] + (d_omega_y[d_idx] * nz - d_omega_z[d_idx] * ny) * a_i
                vi_y = d_v[d_idx] + (d_omega_z[d_idx] * nx - d_omega_x[d_idx] * nz) * a_i
                vi_z = d_w[d_idx] + (d_omega_x[d_idx] * ny - d_omega_y[d_idx] * nx) * a_i

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

                # tangential velocity
                vt_x = vij_x - vn_x
                vt_y = vij_y - vn_y
                vt_z = vij_z - vn_z
                # magnitude of the tangential velocity
                # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

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
                tmp_1 = 1. / d_m_b[d_idx]
                tmp_2 = 0.
                m_eff = 1. / (tmp_1 + tmp_2)
                eta_n = -2. * (5./6.)**0.5 * beta * (S_n * m_eff)**0.5

                fn = kn * overlap**1.5
                fn_x = fn * nx - eta_n * vn_x
                fn_y = fn * ny - eta_n * vn_y
                fn_z = fn * nz - eta_n * vn_z

                #################################
                # tangential force computation  #
                #################################
                # if the particle is not been tracked then assign an index in
                # tracking history.
                ft_x = 0.
                ft_y = 0.
                ft_z = 0.
                # tangential velocity
                vij_magn = (vij_x**2. + vij_y**2. + vij_z**2.)**0.5

                if vij_magn < 1e-12:
                    d_tng_sw_x[found_at] = 0.
                    d_tng_sw_y[found_at] = 0.
                    d_tng_sw_z[found_at] = 0.
                else:
                    # print("inside")
                    # project tangential spring on the current plane normal
                    d_tng_sw_x[found_at] += vt_x * dt
                    d_tng_sw_y[found_at] += vt_y * dt
                    d_tng_sw_z[found_at] += vt_z * dt

                    # Compute the tangential stiffness
                    tmp_1 = (2. - d_nu[d_idx]) / d_G[d_idx]
                    tmp_2 = (2. - s_nu[i]) / s_G[i]
                    G_eff = 1. / (tmp_1 + tmp_2)
                    # Eq 12 [1]
                    kt = 8. * G_eff * (R_eff * overlap)**0.5
                    S_t = kt
                    eta_t = -2 * (5/6)**0.5 * beta * (S_t * m_eff)**0.5

                    ft_x_star = -kt * d_tng_sw_x[found_at] - eta_t * vt_x
                    ft_y_star = -kt * d_tng_sw_y[found_at] - eta_t * vt_y
                    ft_z_star = -kt * d_tng_sw_z[found_at] - eta_t * vt_z

                    ft_magn = (ft_x_star**2. + ft_y_star**2. + ft_z_star**2.)**0.5

                    ti_x = 0.
                    ti_y = 0.
                    ti_z = 0.

                    if ft_magn > 1e-12:
                        ti_x = ft_x_star / ft_magn
                        ti_y = ft_y_star / ft_magn
                        ti_z = ft_z_star / ft_magn

                    fn_magn = (fn_x**2. + fn_y**2. + fn_z**2.)**0.5

                    ft_magn_star = min(self.fric_coeff * fn_magn, ft_magn)

                    # compute the tangential force, by equation 17 (Lethe)
                    ft_x = ft_magn_star * ti_x
                    ft_y = ft_magn_star * ti_y
                    ft_z = ft_magn_star * ti_z

                    # Add damping to the limited force
                    ft_x += eta_t * vt_x
                    ft_y += eta_t * vt_y
                    ft_z += eta_t * vt_z

                    # reset the spring length
                    d_tng_sw_x[found_at] = -ft_x / kt
                    d_tng_sw_y[found_at] = -ft_y / kt
                    d_tng_sw_z[found_at] = -ft_z / kt

                d_fx[d_idx] += fn_x + ft_x
                d_fy[d_idx] += fn_y + ft_y
                d_fz[d_idx] += fn_z + ft_z

                # torque = n cross F
                d_torque_x[d_idx] += (ny * ft_z - nz * ft_y) * a_i
                d_torque_y[d_idx] += (nz * ft_x - nx * ft_z) * a_i
                d_torque_z[d_idx] += (nx * ft_y - ny * ft_x) * a_i
            else:
                d_tng_sw_x[found_at] = 0.
                d_tng_sw_y[found_at] = 0.
                d_tng_sw_z[found_at] = 0.


class UpdateTangentialContacts(Equation):
    def initialize_pair(self, d_idx, d_x, d_y, d_z, d_rad_s,
                        d_total_no_tng_contacts, d_tng_idx,
                        d_max_no_tng_contacts_limit, d_tng_ss_x, d_tng_ss_y,
                        d_tng_ss_z, d_tng_idx_dem_id, s_x, s_y, s_z, s_rad_s,
                        s_dem_id):
        p = declare('int')
        count = declare('int')
        k = declare('int')
        xij = declare('matrix(3)')
        last_idx_tmp = declare('int')
        sidx = declare('int')
        dem_id = declare('int')
        rij = 0.0

        idx_total_ctcs = declare('int')
        idx_total_ctcs = d_total_no_tng_contacts[d_idx]
        # particle idx contacts has range of indices
        # and the first index would be
        p = d_idx * d_max_no_tng_contacts_limit[0]
        last_idx_tmp = p + idx_total_ctcs - 1
        k = p
        count = 0

        # loop over all the contacts of particle d_idx
        while count < idx_total_ctcs:
            # The index of the particle with which
            # d_idx in contact is
            sidx = d_tng_idx[k]
            # get the dem id of the particle
            dem_id = d_tng_idx_dem_id[k]

            if sidx == -1:
                break
            else:
                if dem_id == s_dem_id[sidx]:
                    xij[0] = d_x[d_idx] - s_x[sidx]
                    xij[1] = d_y[d_idx] - s_y[sidx]
                    xij[2] = d_z[d_idx] - s_z[sidx]
                    rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] +
                               xij[2] * xij[2])

                    overlap = d_rad_s[d_idx] + s_rad_s[sidx] - rij

                    if overlap <= 0.:
                        # if the swap index is the current index then
                        # simply make it to null contact.
                        if k == last_idx_tmp:
                            d_tng_idx[k] = -1
                            d_tng_idx_dem_id[k] = -1
                            d_tng_ss_x[k] = 0.
                            d_tng_ss_y[k] = 0.
                            d_tng_ss_z[k] = 0.
                        else:
                            # swap the current tracking index with the final
                            # contact index
                            d_tng_idx[k] = d_tng_idx[last_idx_tmp]
                            d_tng_idx[last_idx_tmp] = -1

                            # swap tangential x displacement
                            d_tng_ss_x[k] = d_tng_ss_x[last_idx_tmp]
                            d_tng_ss_x[last_idx_tmp] = 0.

                            # swap tangential y displacement
                            d_tng_ss_y[k] = d_tng_ss_y[last_idx_tmp]
                            d_tng_ss_y[last_idx_tmp] = 0.

                            # swap tangential z displacement
                            d_tng_ss_z[k] = d_tng_ss_z[last_idx_tmp]
                            d_tng_ss_z[last_idx_tmp] = 0.

                            # swap tangential idx dem id
                            d_tng_idx_dem_id[k] = d_tng_idx_dem_id[
                                last_idx_tmp]
                            d_tng_idx_dem_id[last_idx_tmp] = -1

                            # decrease the last_idx_tmp, since we swapped it to
                            # -1
                            last_idx_tmp -= 1

                        # decrement the total contacts of the particle
                        d_total_no_tng_contacts[d_idx] -= 1
                    else:
                        k = k + 1
                else:
                    k = k + 1
                count += 1


class SSHertzContactForce(Equation):
    """
    Hertz

    From TODO
    """
    def __init__(self, dest, sources, en, fric_coeff):
        self.en = en
        self.fric_coeff = fric_coeff
        super(SSHertzContactForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_b, d_u, d_v, d_w, d_omega_x, d_omega_y, d_omega_z, d_fx, d_fy,
             d_fz,
             d_tng_ss_x,
             d_tng_ss_y,
             d_tng_ss_z,
             d_torque_x,
             d_torque_y,
             d_torque_z,
             XIJ, RIJ, d_rad_s,
             s_idx, s_m_b, s_u, s_v, s_w, s_omega_x, s_omega_y, s_omega_z, s_rad_s,
             d_nu, s_nu, d_E, s_E, d_G, s_G,
             d_total_no_tng_contacts,
             d_max_no_tng_contacts_limit,
             d_tng_idx,
             d_tng_idx_dem_id,
             s_dem_id,
             dt, t):
        p, q1, tot_ctcs, j, found_at, found = declare('int', 6)
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 1e-12:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

            # ---- Relative velocity computation (Eq 2.9) ----
            # relative velocity of particle d_idx w.r.t particle s_idx at
            # contact point. The velocity difference provided by PySPH is
            # only between translational velocities, but we need to
            # consider rotational velocities also.
            # Distance till contact point
            a_i = d_rad_s[d_idx] - overlap / 2.
            a_j = s_rad_s[s_idx] - overlap / 2.

            # velocity of particle i at the contact point
            vi_x = d_u[d_idx] + (d_omega_y[d_idx] * nz - d_omega_z[d_idx] * ny) * a_i
            vi_y = d_v[d_idx] + (d_omega_z[d_idx] * nx - d_omega_x[d_idx] * nz) * a_i
            vi_z = d_w[d_idx] + (d_omega_x[d_idx] * ny - d_omega_y[d_idx] * nx) * a_i

            # just flip the normal and compute the angular velocity
            # contribution
            vj_x = s_u[s_idx] + (-s_omega_y[s_idx] * nz + s_omega_z[s_idx] * ny) * a_j
            vj_y = s_v[s_idx] + (-s_omega_z[s_idx] * nx + s_omega_x[s_idx] * nz) * a_j
            vj_z = s_w[s_idx] + (-s_omega_x[s_idx] * ny + s_omega_y[s_idx] * nx) * a_j

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

            # tangential velocity
            vt_x = vij_x - vn_x
            vt_y = vij_y - vn_y
            vt_z = vij_z - vn_z
            # magnitude of the tangential velocity
            # vt_magn = (vt_x * vt_x + vt_y * vt_y + vt_z * vt_z)**0.5

            ############################
            # normal force computation #
            ############################
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
            tmp_1 = 1. / d_m_b[d_idx]
            tmp_2 = 1. / s_m_b[s_idx]
            m_eff = 1. / (tmp_1 + tmp_2)
            eta_n = -2. * (5./6.)**0.5 * beta * (S_n * m_eff)**0.5

            fn = kn * overlap**1.5
            fn_x = fn * nx - eta_n * vn_x
            fn_y = fn * ny - eta_n * vn_y
            fn_z = fn * nz - eta_n * vn_z

            #################################
            # tangential force computation  #
            #################################
            # if the particle is not been tracked then assign an index in
            # tracking history.
            tot_ctcs = d_total_no_tng_contacts[d_idx]

            # d_idx has a range of tracking indices with sources
            # starting index is p
            p = d_idx * d_max_no_tng_contacts_limit[0]
            # ending index is q -1
            q1 = p + tot_ctcs

            # check if the particle is in the tracking list
            # if so, then save the location at found_at
            found = 0
            for j in range(p, q1):
                if s_idx == d_tng_idx[j]:
                    if s_dem_id[s_idx] == d_tng_idx_dem_id[j]:
                        found_at = j
                        found = 1
                        break

            ft_x = 0.
            ft_y = 0.
            ft_z = 0.

            if found == 0:
                found_at = q1
                d_tng_idx[found_at] = s_idx
                d_total_no_tng_contacts[d_idx] += 1
                d_tng_idx_dem_id[found_at] = s_dem_id[s_idx]
                d_tng_ss_x[found_at] = 0.
                d_tng_ss_y[found_at] = 0.
                d_tng_ss_z[found_at] = 0.

            # We are tracking the particle history at found_at
            # tangential velocity
            vij_magn = (vij_x**2. + vij_y**2. + vij_z**2.)**0.5

            if vij_magn < 1e-12:
                d_tng_ss_x[found_at] = 0.
                d_tng_ss_y[found_at] = 0.
                d_tng_ss_z[found_at] = 0.
            else:
                ti_magn = (vt_x**2. + vt_y**2. + vt_z**2.)**0.5

                ti_x = 0.
                ti_y = 0.
                ti_z = 0.

                if ti_magn > 1e-12:
                    ti_x = vt_x / ti_magn
                    ti_y = vt_y / ti_magn
                    ti_z = vt_z / ti_magn

                # project tangential spring on on the current plane normal
                delta_lt_x_star = d_tng_ss_x[found_at] + vij_x * dt
                delta_lt_y_star = d_tng_ss_y[found_at] + vij_y * dt
                delta_lt_z_star = d_tng_ss_z[found_at] + vij_z * dt

                delta_lt_dot_ti = (delta_lt_x_star * ti_x +
                                   delta_lt_y_star * ti_y +
                                   delta_lt_z_star * ti_z)

                d_tng_ss_x[found_at] = delta_lt_dot_ti * ti_x
                d_tng_ss_y[found_at] = delta_lt_dot_ti * ti_y
                d_tng_ss_z[found_at] = delta_lt_dot_ti * ti_z

                # Compute the tangential stiffness
                tmp_1 = (2. - d_nu[d_idx]) / d_G[d_idx]
                tmp_2 = (2. - s_nu[s_idx]) / s_G[s_idx]
                G_eff = 1. / (tmp_1 + tmp_2)
                # Eq 12 [1]
                kt = 8. * G_eff * (R_eff * overlap)**0.5
                S_t = kt
                eta_t = -2 * (5/6)**0.5 * beta * (S_t * m_eff)**0.5

                ft_x_star = -kt * d_tng_ss_x[found_at] - eta_t * vt_x
                ft_y_star = -kt * d_tng_ss_y[found_at] - eta_t * vt_y
                ft_z_star = -kt * d_tng_ss_z[found_at] - eta_t * vt_z

                ft_magn = (ft_x_star**2. + ft_y_star**2. + ft_z_star**2.)**0.5

                ti_x = 0.
                ti_y = 0.
                ti_z = 0.

                if ft_magn > 1e-12:
                    ti_x = ft_x_star / ft_magn
                    ti_y = ft_y_star / ft_magn
                    ti_z = ft_z_star / ft_magn

                fn_magn = (fn_x**2. + fn_y**2. + fn_z**2.)**0.5

                ft_magn_star = min(self.fric_coeff * fn_magn, ft_magn)

                # compute the tangential force, by equation 17 (Lethe)
                ft_x = ft_magn_star * ti_x
                ft_y = ft_magn_star * ti_y
                ft_z = ft_magn_star * ti_z

                # Add damping to the limited force
                ft_x += eta_t * vt_x
                ft_y += eta_t * vt_y
                ft_z += eta_t * vt_z

                # reset the spring length
                d_tng_ss_x[found_at] = -ft_x / kt
                d_tng_ss_y[found_at] = -ft_y / kt
                d_tng_ss_z[found_at] = -ft_z / kt

            d_fx[d_idx] += fn_x + ft_x
            d_fy[d_idx] += fn_y + ft_y
            d_fz[d_idx] += fn_z + ft_z

            # torque = n cross F
            d_torque_x[d_idx] += (ny * ft_z - nz * ft_y) * a_i
            d_torque_y[d_idx] += (nz * ft_x - nx * ft_z) * a_i
            d_torque_z[d_idx] += (nx * ft_y - ny * ft_x) * a_i


class SSDMTContactForce(Equation):
    """
    Hertz

    From TODO
    """
    def __init__(self, dest, sources, gamma):
        self.gamma = gamma
        super(SSDMTContactForce, self).__init__(dest, sources)

    def loop(self, d_idx, d_fx, d_fy,
             d_fz,
             d_rad_s,
             s_idx,
             s_rad_s,
             dt, t, RIJ, XIJ):
        overlap = -1.

        # check the particles are not on top of each other.
        if RIJ > 1e-12:
            overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

        # ---------- force computation starts ------------
        # if particles are overlapping
        if overlap > 0:
            # normal vector (nij) passing from d_idx to s_idx, i.e., i to j
            rinv = 1.0 / RIJ
            # in PySPH: XIJ[0] = d_x[d_idx] - s_x[s_idx]
            nx = XIJ[0] * rinv
            ny = XIJ[1] * rinv
            nz = XIJ[2] * rinv

            ############################
            # DMT force computation    #
            ############################
            tmp_1 = 1. / d_rad_s[d_idx]
            tmp_2 = 1. / s_rad_s[s_idx]
            R_eff = 1. / (tmp_1 + tmp_2)

            f_dmt = - 4. * pi * self.gamma * R_eff

            d_fx[d_idx] += f_dmt * nx
            d_fy[d_idx] += f_dmt * ny
            d_fz[d_idx] += f_dmt * nz


def add_swelling_properties_to_rigid_body(pa, dim=2):
    """Note: Set the diffusion coefficient in the example file
    separately. This function will only add the properties

    """
    add_properties(pa, 'm_dot_lp', 'm_water', 'm_dry',
                   'rho_water_0', 'diffusion_coeff')
    pa.m_dry[:] = pa.m[:]
    pa.rho_water_0[:] = 0.
    pa.diffusion_coeff[:] = 1.

    # volume of water in the particle
    add_properties(pa, 'vol_water')
    pa.vol_water[:] = 0.
    # Initial volume of the particle (dry particle)
    add_properties(pa, 'init_vol')
    pa.init_vol[:] = 4. / 3. * np.pi * pa.rad_s[:]**3.
    # total volume of the particle
    add_properties(pa, 'total_vol')
    pa.total_vol[:] = pa.init_vol[:]

    # total volume of the particle
    add_properties(pa, 'surface_area')
    pa.surface_area[:] = 4. * pi * pa.rad_s[:]**2.

    # concentration of the particle
    add_properties(pa, 'conc_p')
    pa.conc_p[:] = 0.
    # concentration of water
    add_properties(pa, 'conc_w')
    pa.conc_w[:] = 1000.


class DEMScheme(Scheme):
    def __init__(self,
                 dem_particles,
                 boundaries,
                 kn=1e5,
                 en=0.5,
                 fric_coeff=0.0,
                 gamma=0.,
                 dim=2,
                 gx=0.0,
                 gy=0.0,
                 gz=0.0,
                 contact_model="LVCDisplacement"):
        self.dem_particles = dem_particles

        if boundaries is None:
            self.boundaries = []
        else:
            self.boundaries = boundaries

        self.dim = dim

        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.kn = kn
        self.en = en
        self.fric_coeff = fric_coeff
        self.gamma = gamma

        self.contact_model = contact_model
        self.swelling = False

        self.solver = None

    def add_user_options(self, group):
        # add_bool_argument(
        #     group,
        #     'shear-stress-tvf-correction',
        #     dest='shear_stress_tvf_correction',
        #     default=True,
        #     help='Add the extra shear stress rate term arriving due to TVF')

        # add_bool_argument(group,
        #                   'edac',
        #                   dest='edac',
        #                   default=True,
        #                   help='Use pressure evolution equation EDAC')
        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.0,
                           type=float,
                           help="Friction coefficient")

        group.add_argument("--en", action="store",
                           dest="en", default=0.1,
                           type=float,
                           help="Coefficient of restitution")

        group.add_argument("--gamma", action="store",
                           dest="gamma", default=0.0,
                           type=float,
                           help="Surface energy")

        choices = ['LVC']
        group.add_argument("--contact-model",
                           action="store",
                           dest='contact_model',
                           default="LVCDisplacement",
                           choices=choices,
                           help="Specify what contact model to use " % choices)

        add_bool_argument(group,
                          'swelling',
                          dest='swelling',
                          default=False,
                          help='Apply swelling to the particles')

    def consume_user_options(self, options):
        _vars = ['contact_model', 'fric_coeff', 'en', 'gamma', 'swelling']
        data = dict((var, self._smart_getattr(options, var)) for var in _vars)
        self.configure(**data)

    def get_equations(self):
        from pysph.sph.equation import Group, MultiStageEquations
        all = list(set(self.dem_particles + self.boundaries))

        stage1 = []
        g1 = []

        # ------------------------
        # stage 1 equations starts
        # ------------------------
        if self.swelling is True:
            for granules in self.dem_particles:
                g1.append(
                    # see the previous examples and write down the sources
                    ComputeSwelling(
                        dest=granules, sources=None))
            stage1.append(Group(equations=g1, real=False))

        # ------------------------
        # stage 2 equations starts
        # ------------------------
        stage2 = []
        g1 = []
        for granules in self.dem_particles:
            g1.append(
                # see the previous examples and write down the sources
                UpdateTangentialContacts(
                    dest=granules, sources=self.dem_particles))
        stage2.append(Group(equations=g1, real=False))

        g2 = []
        for granules in self.dem_particles:
            g2.append(
                BodyForce(dest=granules,
                          sources=None,
                          gx=self.gx,
                          gy=self.gy,
                          gz=self.gz))

        for granules in self.dem_particles:
            g2.append(SSHertzContactForce(dest=granules,
                                          sources=self.dem_particles,
                                          en=self.en,
                                          fric_coeff=self.fric_coeff))

            g2.append(SSDMTContactForce(dest=granules,
                                        sources=self.dem_particles,
                                        gamma=self.gamma))

            if len(self.boundaries) > 0:
                g2.append(SWHertzContactForce(dest=granules,
                                              sources=self.boundaries,
                                              en=self.en,
                                              fric_coeff=self.fric_coeff))

        stage2.append(Group(equations=g2, real=False))

        return MultiStageEquations([stage1, stage2])

    def configure_solver(self,
                         kernel=None,
                         integrator_cls=None,
                         extra_steppers=None,
                         **kw):
        from pysph.base.kernels import CubicSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from pysph.solver.solver import Solver
        if kernel is None:
            kernel = CubicSpline(dim=self.dim)

        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        for dem_particles in self.dem_particles:
            if dem_particles not in steppers:
                steppers[dem_particles] = DEMStep()

        cls = integrator_cls if integrator_cls is not None else GTVFIntegrator
        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim,
                             integrator=integrator,
                             kernel=kernel,
                             **kw)

    def get_solver(self):
        return self.solver
