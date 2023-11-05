import numpy
import numpy as np

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.base.utils import get_particle_array
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (QuinticSpline)
from pysph.examples.solid_mech.impact import add_properties

from pysph.sph.integrator import Integrator


def get_particle_array_fluid(name, x, y, z=0., m=0., h=0., rho=0., ):
    pa = get_particle_array(name=name,
                            x=x,
                            y=y,
                            z=z,
                            h=h,
                            rho=rho,
                            m=m)
    add_properties(pa, 'arho', 'aconcentration', 'concentration', 'diff_coeff',
                   'is_static', 'ap', 'auhat', 'avhat', 'awhat',
                   'uhat', 'vhat', 'what', 'p0')
    add_properties(pa, 'm_frac')
    pa.add_constant('c0_ref', 0.)
    pa.add_constant('p0_ref', 0.)
    pa.m_frac[:] = 1.
    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    pa.add_output_arrays(['concentration', 'is_static', 'diff_coeff'])

    return pa


def get_particle_array_boundary(constants=None, **props):
    solids_props = [
        'wij', 'm_frac', 'ug', 'vf', 'uf', 'wf', 'vg', 'wg'
    ]

    # set wdeltap to -1. Which defaults to no self correction
    consts = {

    }
    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=solids_props,
                            **props)
    pa.m_frac[:] = 1.

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


class ContinuityEquation(Equation):
    r"""Density rate:

    :math:`\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab}`

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class DeltaSPHCorrection(Equation):
    def __init__(self, dest, sources, c0, delta_fac=0.2):
        self.delta_fac = delta_fac
        self.c0 = c0
        super(DeltaSPHCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_arho, s_idx, s_m, s_rho, d_rho, d_h, DWIJ, VIJ, XIJ,
             R2IJ, EPS):
        tmp = 2 * s_m[s_idx] / s_rho[s_idx] * (d_rho[d_idx] - s_rho[s_idx])
        tmp1 = XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2]
        tmp2 = self.delta_fac * d_h[d_idx] * self.c0

        d_arho[d_idx] += tmp2 * tmp * tmp1 / (R2IJ + EPS)


class EDACEquation(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(EDACEquation, self).__init__(dest, sources)

    def initialize(self, d_ap, d_idx):
        d_ap[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, d_rho, d_c0_ref, d_u, d_v, d_w,
             s_p, s_m, s_rho, d_ap, DWIJ, XIJ,
             s_u, s_v, s_w, R2IJ, VIJ, EPS):
        cs2 = d_c0_ref[0] * d_c0_ref[0]

        rhoj1 = 1.0 / s_rho[s_idx]
        Vj = s_m[s_idx] * rhoj1
        rhoi = d_rho[d_idx]
        pi = d_p[d_idx]
        rhoj = s_rho[s_idx]
        pj = s_p[s_idx]

        vij_dot_dwij = -(VIJ[0] * DWIJ[0] + VIJ[1] * DWIJ[1] +
                         VIJ[2] * DWIJ[2])

        #######################################################
        # first term on the rhs of Eq 23 of the current paper #
        #######################################################
        d_ap[d_idx] += - rhoi * cs2 * Vj * vij_dot_dwij

        #######################################################
        # fourth term on the rhs of Eq 19 of the current paper #
        #######################################################
        rhoij = d_rho[d_idx] + s_rho[s_idx]
        # The viscous damping of pressure.
        xijdotdwij = DWIJ[0] * XIJ[0] + DWIJ[1] * XIJ[1] + DWIJ[2] * XIJ[2]
        tmp = Vj * 4 * xijdotdwij / (rhoij * (R2IJ + EPS))
        d_ap[d_idx] += self.nu * tmp * (d_p[d_idx] - s_p[s_idx])


class StateEquation(Equation):
    def __init__(self, dest, sources, p0, rho0, b=1.0):
        self.b = b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        # d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b) + self.p0
        d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b)


class StateEquationInternalFlow(Equation):
    def __init__(self, dest, sources, p0, rho0, b=1.0):
        self.b = b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquationInternalFlow, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b) + self.p0


class GTVFSetP0(Equation):
    def initialize(self, d_idx, d_rho, d_p0, d_p, d_p0_ref):
        d_p0[d_idx] = min(10. * abs(d_p[d_idx]), d_p0_ref[0])


class DiffusionEquation(Equation):
    """
    Rate of change of concentration of a particle from Fick's law.
    Equation 20 (without advection) of reference [1].

    [1]
    """
    def initialize(self, d_idx, d_aconcentration):
        d_aconcentration[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m,
             s_m,
             d_diff_coeff,
             s_diff_coeff,
             d_concentration,
             s_concentration,
             d_aconcentration,
             R2IJ, EPS, DWIJ, VIJ, XIJ):

        # averaged shear viscosity Eq. (6)
        tmp_i = s_m[s_idx] * d_rho[d_idx] * d_diff_coeff[d_idx]
        tmp_j = d_m[d_idx] * s_rho[s_idx] * s_diff_coeff[s_idx]

        rho_i = d_rho[d_idx]
        rho_j = s_rho[s_idx]
        rho_ij = (rho_i * rho_j)

        conc_ij = d_concentration[d_idx] - s_concentration[s_idx]

        # scalar part of the kernel gradient
        Fij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        # accelerations 3rd term in Eq. (8)
        tmp = (tmp_i + tmp_j) * conc_ij * Fij/(R2IJ + EPS)

        d_aconcentration[d_idx] += 1. / rho_ij * tmp


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au,
             d_av, d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class ComputeAuHatGTVF(Equation):
    def __init__(self, dest, sources):
        super(ComputeAuHatGTVF, self).__init__(dest, sources)

    def initialize(self, d_idx, d_auhat, d_avhat, d_awhat):
        d_auhat[d_idx] = 0.0
        d_avhat[d_idx] = 0.0
        d_awhat[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, d_p0, s_rho, s_m, d_auhat, d_avhat,
             d_awhat, WIJ, SPH_KERNEL, DWIJ, XIJ, RIJ, HIJ):
        dwijhat = declare('matrix(3)')

        rhoa = d_rho[d_idx]

        rhoa21 = 1. / (rhoa * rhoa)

        # add the background pressure acceleration
        tmp = -d_p0[d_idx] * s_m[s_idx] * rhoa21

        SPH_KERNEL.gradient(XIJ, RIJ, 0.5 * HIJ, dwijhat)

        d_auhat[d_idx] += tmp * dwijhat[0]
        d_avhat[d_idx] += tmp * dwijhat[1]
        d_awhat[d_idx] += tmp * dwijhat[2]


class SolidWallNoSlipBC(Equation):
    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(SolidWallNoSlipBC, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au, d_av,
             d_aw, d_u, d_v, d_w, s_ug, s_vg, s_wg, R2IJ, EPS,
             DWIJ, XIJ):
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 4 * (etai * etaj)/(etai + etaj)

        xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        tmp = s_m[s_idx]/(d_rho[d_idx] * s_rho[s_idx])
        fac = tmp * etaij * xdotdij/(R2IJ + EPS)

        d_au[d_idx] += fac * (d_u[d_idx] - s_ug[s_idx])
        d_av[d_idx] += fac * (d_v[d_idx] - s_vg[s_idx])
        d_aw[d_idx] += fac * (d_w[d_idx] - s_wg[s_idx])


class FluidStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_vhat,
               d_what, d_auhat, d_avhat, d_awhat, d_is_static, dt):
        dtb2 = 0.5 * dt
        if d_is_static[d_idx] == 0.:
            d_u[d_idx] += dtb2 * d_au[d_idx]
            d_v[d_idx] += dtb2 * d_av[d_idx]
            d_w[d_idx] += dtb2 * d_aw[d_idx]

            d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
            d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
            d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_rho, d_arho, d_concentration,
               d_aconcentration, d_ap, d_p, d_uhat, d_vhat, d_what, dt):
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_concentration[d_idx] += dt * d_aconcentration[d_idx]
        d_p[d_idx] += dt * d_ap[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_is_static, dt):
        dtb2 = 0.5 * dt
        if d_is_static[d_idx] == 0.:
            d_u[d_idx] += dtb2 * d_au[d_idx]
            d_v[d_idx] += dtb2 * d_av[d_idx]
            d_w[d_idx] += dtb2 * d_aw[d_idx]


class FluidsScheme(Scheme):
    def __init__(self, fluids, boundaries, dim, c0, nu, rho0, pb=0.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.0):
        self.c0 = c0
        self.nu = nu
        self.rho0 = rho0
        self.pb = pb
        print("pb value is", self.pb)
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dim = dim
        self.alpha = alpha
        self.fluids = fluids
        self.boundaries = boundaries
        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=None,
                           help="Alpha for the artificial viscosity.")

    def consume_user_options(self, options):
        vars = [
            'alpha'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = FluidStep
        cls = (integrator_cls
               if integrator_cls is not None else GTVFIntegrator)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)

    def get_equations(self):
        from pysph.sph.wc.gtvf import (MomentumEquationViscosity)
        from pysph.sph.basic_equations import (IsothermalEOS)
        from pysph.sph.wc.edac import (SolidWallPressureBC,
                                       ClampWallPressure,
                                       SourceNumberDensity)
        from pysph.sph.wc.transport_velocity import (SetWallVelocity)

        from pysph.sph.wc.transport_velocity import (
            MomentumEquationArtificialViscosity
        )

        all = self.fluids + self.boundaries
        # =========================#
        # stage 1 equations start
        # =========================#
        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquation(dest=fluid,
                                          sources=all), )

            eqs.append(DiffusionEquation(dest=fluid,
                                         sources=self.fluids), )
        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []

        tmp = []
        for fluid in self.fluids:
            tmp.append(
                StateEquation(dest=fluid,
                              sources=None,
                              rho0=self.rho0,
                              p0=self.pb))

        stage2.append(Group(equations=tmp, real=False))

        if len(self.boundaries) > 0:
            eqs = []
            for boundary in self.boundaries:
                eqs.append(SetWallVelocity(dest=boundary, sources=self.fluids))
            stage2.append(Group(equations=eqs, real=False))

        if len(self.boundaries) > 0:
            eqs = []
            for boundary in self.boundaries:
                eqs.append(
                    SourceNumberDensity(dest=boundary, sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=boundary, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                # eqs.append(
                #     ClampWallPressure(dest=boundary, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            if self.alpha > 0.:
                eqs.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=all, c0=self.c0,
                        alpha=self.alpha
                    )
                )

            if self.nu > 0.0:
                eqs.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=self.fluids, nu=self.nu
                    )
                )
                if len(self.boundaries) > 0:
                    eqs.append(
                        SolidWallNoSlipBC(
                            dest=fluid, sources=self.boundaries, nu=self.nu
                        )
                    )
            eqs.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all,
                                                 gx=self.gx, gy=self.gy,
                                                 gz=self.gz), )

        stage2.append(Group(equations=eqs, real=True))

        return MultiStageEquations([stage1, stage2])
