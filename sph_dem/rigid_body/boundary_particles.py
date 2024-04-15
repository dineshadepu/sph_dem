from pysph.sph.equation import Equation
from compyle.api import declare
from math import sqrt

from pysph.sph.equation import Group
from pysph.sph.isph.wall_normal import ComputeNormals, SmoothNormals


def add_boundary_identification_properties(pa):
    # for normals
    pa.add_property('normal', stride=3)
    pa.add_property('normal0', stride=3)
    pa.add_property('normal_tmp', stride=3)
    pa.add_property('normal_norm')

    # check for boundary particle
    pa.add_property('is_boundary', type='int')

    pa.add_output_arrays(['is_boundary'])


class IdentifyBoundaryParticleCosAngle(Equation):
    def __init__(self, dest, sources):
        super(IdentifyBoundaryParticleCosAngle, self).__init__(dest, sources)

    def initialize(self, d_idx, d_is_boundary, d_normal_norm, d_normal):
        # set all of them to be boundary
        i, idx3 = declare('int', 2)
        idx3 = 3 * d_idx

        normal_norm = (d_normal[idx3]**2. + d_normal[idx3 + 1]**2. +
                       d_normal[idx3 + 2]**2.)

        d_normal_norm[d_idx] = normal_norm

        # normal norm is always one
        if normal_norm > 1e-6:
            # first set the particle as boundary if its normal exists
            d_is_boundary[d_idx] = 1
        else:

            d_is_boundary[d_idx] = 0

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h, d_is_boundary,
                 d_normal, s_m, s_x, s_y, s_z, s_h, SPH_KERNEL, NBRS, N_NBRS):
        i, idx3, s_idx = declare('int', 3)
        xij = declare('matrix(3)')
        idx3 = 3 * d_idx

        h_i = d_h[d_idx]
        # normal norm is always one
        if d_is_boundary[d_idx] == 1:
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                xij[0] = d_x[d_idx] - s_x[s_idx]
                xij[1] = d_y[d_idx] - s_y[s_idx]
                xij[2] = d_z[d_idx] - s_z[s_idx]
                rij = sqrt(xij[0]**2. + xij[1]**2. + xij[2]**2.)
                if rij > 1e-9 * h_i and rij < 2. * h_i:
                    # dot product between the vector and line joining sidx
                    dot = -(d_normal[idx3] * xij[0] + d_normal[idx3 + 1] *
                            xij[1] + d_normal[idx3 + 2] * xij[2])

                    fac = dot / rij

                    if fac > 0.5:
                        d_is_boundary[d_idx] = 0
                        break


class ComputeNormalsEDAC(Equation):
    """Compute normals using a simple approach

    .. math::

       -\frac{m_j}{\rho_j} DW_{ij}

    First compute the normals, then average them and finally normalize them.

    """
    def initialize(self, d_idx, d_edac_normal_tmp, d_edac_normal):
        idx = declare('int')
        idx = 3 * d_idx
        d_edac_normal_tmp[idx] = 0.0
        d_edac_normal_tmp[idx + 1] = 0.0
        d_edac_normal_tmp[idx + 2] = 0.0
        d_edac_normal[idx] = 0.0
        d_edac_normal[idx + 1] = 0.0
        d_edac_normal[idx + 2] = 0.0

    def loop(self, d_idx, d_edac_normal_tmp, s_idx, s_m, s_rho, DWIJ):
        idx = declare('int')
        idx = 3 * d_idx
        fac = -s_m[s_idx] / s_rho[s_idx]
        d_edac_normal_tmp[idx] += fac * DWIJ[0]
        d_edac_normal_tmp[idx + 1] += fac * DWIJ[1]
        d_edac_normal_tmp[idx + 2] += fac * DWIJ[2]

    def post_loop(self, d_idx, d_edac_normal_tmp, d_h):
        idx = declare('int')
        idx = 3 * d_idx
        mag = sqrt(d_edac_normal_tmp[idx]**2 + d_edac_normal_tmp[idx + 1]**2 +
                   d_edac_normal_tmp[idx + 2]**2)
        if mag > 0.25 / d_h[d_idx]:
            d_edac_normal_tmp[idx] /= mag
            d_edac_normal_tmp[idx + 1] /= mag
            d_edac_normal_tmp[idx + 2] /= mag
        else:
            d_edac_normal_tmp[idx] = 0.0
            d_edac_normal_tmp[idx + 1] = 0.0
            d_edac_normal_tmp[idx + 2] = 0.0


class SmoothNormalsEDAC(Equation):
    def loop(self, d_idx, d_edac_normal, s_edac_normal_tmp, s_idx, s_m, s_rho, WIJ):
        idx = declare('int')
        idx = 3 * d_idx
        fac = s_m[s_idx] / s_rho[s_idx] * WIJ
        d_edac_normal[idx] += fac * s_edac_normal_tmp[3 * s_idx]
        d_edac_normal[idx + 1] += fac * s_edac_normal_tmp[3 * s_idx + 1]
        d_edac_normal[idx + 2] += fac * s_edac_normal_tmp[3 * s_idx + 2]

    def post_loop(self, d_idx, d_edac_normal, d_h):
        idx = declare('int')
        idx = 3 * d_idx
        mag = sqrt(d_edac_normal[idx]**2 + d_edac_normal[idx + 1]**2 +
                   d_edac_normal[idx + 2]**2)
        if mag > 1e-3:
            d_edac_normal[idx] /= mag
            d_edac_normal[idx + 1] /= mag
            d_edac_normal[idx + 2] /= mag
        else:
            d_edac_normal[idx] = 0.0
            d_edac_normal[idx + 1] = 0.0
            d_edac_normal[idx + 2] = 0.0


class IdentifyBoundaryParticleCosAngleEDAC(Equation):
    def __init__(self, dest, sources):
        super(IdentifyBoundaryParticleCosAngleEDAC,
              self).__init__(dest, sources)

    def initialize(self, d_idx, d_edac_is_boundary, d_edac_normal_norm,
                   d_edac_normal):
        # set all of them to be boundary
        i, idx3 = declare('int', 2)
        idx3 = 3 * d_idx

        normal_norm = (d_edac_normal[idx3]**2. + d_edac_normal[idx3 + 1]**2. +
                       d_edac_normal[idx3 + 2]**2.)

        d_edac_normal_norm[d_idx] = normal_norm

        # normal norm is always one
        if normal_norm > 1e-6:
            # first set the particle as boundary if its normal exists
            d_edac_is_boundary[d_idx] = 1
        else:

            d_edac_is_boundary[d_idx] = 0

    def loop_all(self, d_idx, d_x, d_y, d_z, d_rho, d_h, d_edac_is_boundary,
                 d_edac_normal, s_m, s_x, s_y, s_z, s_h, SPH_KERNEL, NBRS,
                 N_NBRS):
        i, idx3, s_idx = declare('int', 3)
        xij = declare('matrix(3)')
        idx3 = 3 * d_idx

        # normal norm is always one
        if d_edac_is_boundary[d_idx] == 1:
            for i in range(N_NBRS):
                s_idx = NBRS[i]
                xij[0] = d_x[d_idx] - s_x[s_idx]
                xij[1] = d_y[d_idx] - s_y[s_idx]
                xij[2] = d_z[d_idx] - s_z[s_idx]
                rij = sqrt(xij[0]**2. + xij[1]**2. + xij[2]**2.)
                if rij > 1e-9 * d_h[d_idx]:
                    # dot product between the vector and line joining sidx
                    dot = -(d_edac_normal[idx3] * xij[0] +
                            d_edac_normal[idx3 + 1] * xij[1] +
                            d_edac_normal[idx3 + 2] * xij[2])

                    fac = dot / rij

                    if fac > 0.5:
                        d_edac_is_boundary[d_idx] = 0
                        break


def get_boundary_identification_etvf_equations(destinations, sources, boundaries=None):
    eqs = []
    g1 = []
    g2 = []
    g3 = []
    all = list(set(destinations + sources))

    for dest in destinations:
        g1.append(ComputeNormals(dest=dest, sources=all))

    for dest in destinations:
        g2.append(SmoothNormals(dest=dest, sources=[dest]))

    for dest in destinations:
        # the sources here will the particle array and the boundaries
        if boundaries == None:
            srcs = [dest]
        else:
            srcs = list(set([dest] + boundaries))

        g3.append(IdentifyBoundaryParticleCosAngle(dest=dest, sources=srcs))

    eqs.append(Group(equations=g1))
    eqs.append(Group(equations=g2))
    eqs.append(Group(equations=g3))

    return eqs
