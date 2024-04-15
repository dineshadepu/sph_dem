from solid_mech import (ComputeAuHatETVF, ComputeAuHatETVFSun2019,
                        SavePositionsIPSTBeforeMoving, AdjustPositionIPST,
                        CheckUniformityIPST, ResetParticlePositionsIPST,
                        EDACEquation, setup_ipst, ComputeAuhatETVFIPSTFluids,
                        SetHIJForInsideParticles)


########################
# IPST equations start #
########################
def setup_ipst(pa, kernel):
    props = 'ipst_x ipst_y ipst_z ipst_dx ipst_dy ipst_dz'.split()

    for prop in props:
        pa.add_property(prop)

    pa.add_constant('ipst_chi0', 0.)
    pa.add_property('ipst_chi')

    equations = [
        Group(
            equations=[
                CheckUniformityIPST(dest=pa.name, sources=[pa.name]),
            ], real=False),
    ]

    sph_eval = SPHEvaluator(arrays=[pa], equations=equations, dim=2,
                            kernel=kernel(dim=2))

    sph_eval.evaluate(dt=0.1)

    pa.ipst_chi0[0] = min(pa.ipst_chi)


class SavePositionsIPSTBeforeMoving(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z):
        d_ipst_x[d_idx] = d_x[d_idx]
        d_ipst_y[d_idx] = d_y[d_idx]
        d_ipst_z[d_idx] = d_z[d_idx]


class AdjustPositionIPST(Equation):
    def __init__(self, dest, sources, u_max):
        self.u_max = u_max
        super(AdjustPositionIPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_dx, d_ipst_dy, d_ipst_dz):
        d_ipst_dx[d_idx] = 0.0
        d_ipst_dy[d_idx] = 0.0
        d_ipst_dz[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_h, d_ipst_dx, d_ipst_dy,
             d_ipst_dz, WIJ, XIJ, RIJ, dt):
        tmp = self.u_max * dt
        Vj = s_m[s_idx] / s_rho[s_idx]  # volume of j

        nij_x = 0.
        nij_y = 0.
        nij_z = 0.
        if RIJ > 1e-12:
            nij_x = XIJ[0] / RIJ
            nij_y = XIJ[1] / RIJ
            nij_z = XIJ[2] / RIJ

        d_ipst_dx[d_idx] += tmp * Vj * nij_x * WIJ
        d_ipst_dy[d_idx] += tmp * Vj * nij_y * WIJ
        d_ipst_dz[d_idx] += tmp * Vj * nij_z * WIJ

    def post_loop(self, d_idx, d_x, d_y, d_z, d_ipst_dx, d_ipst_dy, d_ipst_dz,
                  d_normal):
        idx3 = declare('int')
        idx3 = 3 * d_idx

        # before adding the correction of position, cut off the normal component
        dr_dot_normal = (d_ipst_dx[d_idx] * d_normal[idx3] +
                         d_ipst_dy[d_idx] * d_normal[idx3 + 1] +
                         d_ipst_dz[d_idx] * d_normal[idx3 + 2])

        # if it is going away from the continuum then nullify the
        # normal component.
        if dr_dot_normal > 0.:
            # remove the normal acceleration component
            d_ipst_dx[d_idx] -= dr_dot_normal * d_normal[idx3]
            d_ipst_dy[d_idx] -= dr_dot_normal * d_normal[idx3 + 1]
            d_ipst_dz[d_idx] -= dr_dot_normal * d_normal[idx3 + 2]

        d_x[d_idx] = d_x[d_idx] + d_ipst_dx[d_idx]
        d_y[d_idx] = d_y[d_idx] + d_ipst_dy[d_idx]
        d_z[d_idx] = d_z[d_idx] + d_ipst_dz[d_idx]


class CheckUniformityIPST(Equation):
    """
    For this specific equation one has to update the NNPS

    """
    def __init__(self, dest, sources, tolerance=0.2, debug=False):
        self.inhomogenity = 0.0
        self.debug = debug
        self.tolerance = tolerance
        super(CheckUniformityIPST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_chi):
        d_ipst_chi[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_rho, s_m, d_ipst_chi, d_h, WIJ, XIJ, RIJ,
             dt):
        d_ipst_chi[d_idx] += d_h[d_idx] * d_h[d_idx] * WIJ

    def reduce(self, dst, t, dt):
        chi_max = serial_reduce_array(dst.ipst_chi, 'min')
        self.inhomogenity = fabs(chi_max - dst.ipst_chi0[0])

    def converged(self):
        debug = self.debug
        inhomogenity = self.inhomogenity

        if inhomogenity > self.tolerance:
            if debug:
                print("Not converged:", inhomogenity)
            return -1.0
        else:
            if debug:
                print("Converged:", inhomogenity)
            return 1.0


class ComputeAuhatETVFIPSTFluids(Equation):
    def __init__(self, dest, sources, rho0, dim=2):
        self.rho0 = rho0
        super(ComputeAuhatETVFIPSTFluids, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z,
                   d_auhat, d_avhat, d_awhat, d_rho_splash, dt):
        if d_rho_splash[d_idx] < 0.5 * self.rho0:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.
        else:
            dt_square_inv = 2. / (dt * dt)
            d_auhat[d_idx] = (d_x[d_idx] - d_ipst_x[d_idx]) * dt_square_inv
            d_avhat[d_idx] = (d_y[d_idx] - d_ipst_y[d_idx]) * dt_square_inv
            d_awhat[d_idx] = (d_z[d_idx] - d_ipst_z[d_idx]) * dt_square_inv


class ComputeAuhatETVFIPSTSolids(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z,
                   d_auhat, d_avhat, d_awhat, d_rho_splash, d_rho_ref, dt):
        if d_rho_splash[d_idx] < 0.5 * d_rho_ref[0]:
            d_auhat[d_idx] = 0.
            d_avhat[d_idx] = 0.
            d_awhat[d_idx] = 0.
        else:
            dt_square_inv = 2. / (dt * dt)
            d_auhat[d_idx] = (d_x[d_idx] - d_ipst_x[d_idx]) * dt_square_inv
            d_avhat[d_idx] = (d_y[d_idx] - d_ipst_y[d_idx]) * dt_square_inv
            d_awhat[d_idx] = (d_z[d_idx] - d_ipst_z[d_idx]) * dt_square_inv
    # def post_loop(self, d_idx, d_rho, d_h_b, d_h, d_normal, d_auhat, d_avhat,
    #               d_awhat, d_is_boundary):
    #     idx3 = declare('int')
    #     idx3 = 3 * d_idx

    #     # first put a clearance
    #     magn_auhat = sqrt(d_auhat[d_idx] * d_auhat[d_idx] +
    #                       d_avhat[d_idx] * d_avhat[d_idx] +
    #                       d_awhat[d_idx] * d_awhat[d_idx])

    #     if magn_auhat > 1e-12:
    #         # tmp = min(magn_auhat, self.u_max * 0.5)
    #         tmp = magn_auhat
    #         d_auhat[d_idx] = tmp * d_auhat[d_idx] / magn_auhat
    #         d_avhat[d_idx] = tmp * d_avhat[d_idx] / magn_auhat
    #         d_awhat[d_idx] = tmp * d_awhat[d_idx] / magn_auhat

    #         # Now apply the filter for boundary particles and adjacent particles
    #         if d_h_b[d_idx] < d_h[d_idx]:
    #             if d_is_boundary[d_idx] == 1:
    #                 # since it is boundary make its shifting acceleration zero
    #                 d_auhat[d_idx] = 0.
    #                 d_avhat[d_idx] = 0.
    #                 d_awhat[d_idx] = 0.
    #             else:
    #                 # implies this is a particle adjacent to boundary particle

    #                 # check if the particle is going away from the continuum
    #                 # or into the continuum
    #                 au_dot_normal = (d_auhat[d_idx] * d_normal[idx3] +
    #                                  d_avhat[d_idx] * d_normal[idx3 + 1] +
    #                                  d_awhat[d_idx] * d_normal[idx3 + 2])

    #                 # if it is going away from the continuum then nullify the
    #                 # normal component.
    #                 if au_dot_normal > 0.:
    #                     # remove the normal acceleration component
    #                     d_auhat[d_idx] -= au_dot_normal * d_normal[idx3]
    #                     d_avhat[d_idx] -= au_dot_normal * d_normal[idx3 + 1]
    #                     d_awhat[d_idx] -= au_dot_normal * d_normal[idx3 + 2]


class ResetParticlePositionsIPST(Equation):
    def initialize(self, d_idx, d_ipst_x, d_ipst_y, d_ipst_z, d_x, d_y, d_z):
        d_x[d_idx] = d_ipst_x[d_idx]
        d_y[d_idx] = d_ipst_y[d_idx]
        d_z[d_idx] = d_ipst_z[d_idx]


######################
# IPST equations end #
######################
