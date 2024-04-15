"""We follow papers [1] and [2]

[1] A coupled {SPH}–{DEM} model for erosion process of
solid surface by abrasive water-jet impact, doi = {10.1007/s40571-023-00555-4},

[2] Numerical simulation of dike failure using a GPU-based coupled
DEM–SPH model, doi = {10.1016/j.compfluid.2023.106090},

"""
import numpy
import numpy as np
from math import sqrt
from compyle.api import declare

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.base.utils import get_particle_array
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (QuinticSpline)
from pysph.examples.solid_mech.impact import add_properties

from pysph.sph.integrator import Integrator
from sph_dem.rigid_body.compute_rigid_body_properties import add_properties_stride


def get_particle_array_fluid(name, x, y, z=0., m=0., h=0., rho=0., ):
    pa = get_particle_array(name=name,
                            x=x,
                            y=y,
                            z=z,
                            h=h,
                            rho=rho,
                            m=m)
    add_properties(pa, 'rho_bar', 'arho_bar',
                   'auhat', 'avhat', 'awhat',
                   'uhat', 'vhat', 'what', 'p0', 'epsilon_porous',
                   'ap')
    pa.add_constant('c0_ref', 0.)
    pa.add_constant('p0_ref', 0.)
    pa.add_constant('n', 4.)
    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    pa.add_output_arrays(['rho_bar', 'arho_bar'])

    return pa


def get_particle_array_boundary(constants=None, **props):
    solids_props = [
        'wij', 'ug', 'vf', 'uf', 'wf', 'vg', 'wg', 'rho_bar'
    ]

    # set wdeltap to -1. Which defaults to no self correction
    consts = {

    }
    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=solids_props,
                            **props)
    pa.rho_bar[:] = pa.rho[:]

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


def compute_initial_porosity_of_the_fluid_particles(fluid_pa, dem_pa):
    """
    Add porosity equation required properties

    We have real fluid density before porous consideration, which is
    rho. We compute epsilon_porous and compute the rho_bar which is the
    final density we consider to evolve.
    """
    # compute the porosity and set the density values
    from pysph.tools.sph_evaluator import SPHEvaluator

    eqs = []
    g1 = []
    g1.append(PorosityEquation(dest=fluid_pa.name, sources=[dem_pa.name]))
    eqs.append(Group(equations=g1))

    kernel = QuinticSpline(dim=2)
    arrays = [fluid_pa, dem_pa]
    a_eval = SPHEvaluator(arrays=arrays, equations=eqs,
                          dim=2, kernel=kernel)

    # When
    a_eval.evaluate(0., 1e-4)

    fluid_pa.rho_bar[:] = fluid_pa.rho[:] * fluid_pa.epsilon_porous[:]


class ContinuityEquationUnresolvedCoupling(Equation):
    r"""Equation no 8 of [1]

    [1] A coupled SPH–DEM model for erosion process of solid surface
    by abrasive water-jet impact
    """
    """Tested in xxxx example"""
    def initialize(self, d_idx, d_arho_bar):
        d_arho_bar[d_idx] = 0.0

    def loop(self, d_idx, d_arho_bar, d_rho_bar, s_idx, s_m, s_rho_bar, DWIJ,
             VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho_bar[d_idx] += d_rho_bar[d_idx] * s_m[s_idx] / s_rho_bar[s_idx] * vijdotdwij


class PorosityEquation(Equation):
    r"""Equation no 7 of [1]

    [1] A coupled SPH–DEM model for erosion process of solid surface
    by abrasive water-jet impact
    """
    """Tested in xxxx example"""
    def initialize(self, d_idx, d_epsilon_porous):
        d_epsilon_porous[d_idx] = 0.

    def loop(self, d_idx, d_epsilon_porous,
             d_x,
             d_y,
             d_z,
             s_x,
             s_y,
             s_z,
             s_idx, s_h, s_m, s_dem_volume,
             SPH_KERNEL):
        i, = declare('int', 1)
        xij = declare('matrix(3)')
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]

        xij[0] = x - s_x[s_idx]
        xij[1] = y - s_y[s_idx]
        xij[2] = z - s_z[s_idx]
        rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])

        # Use the smoothing length of the bigger particle (DEM particle)
        wij = SPH_KERNEL.kernel(xij, rij, s_h[s_idx])
        d_epsilon_porous[d_idx] -= wij * s_dem_volume[s_idx]

    def post_loop(self, d_idx, d_epsilon_porous):
        d_epsilon_porous[d_idx] = 1 + d_epsilon_porous[d_idx]


class StateEquation(Equation):
    """Tested in xxxx example"""
    def __init__(self, dest, sources, p0, rho0, b=1.0):
        self.b = b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho_bar, d_epsilon_porous):
        gamma = 7
        tmp = d_rho_bar[d_idx] / (d_epsilon_porous[d_idx] * self.rho0)

        d_p[d_idx] = self.p0 / gamma * (tmp**gamma - self.b)


class MomentumEquationPressureGradientUnresolvedCoupling(Equation):
    r"""Equation no 9 of [1]

    [1] A coupled SPH–DEM model for erosion process of solid surface
    by abrasive water-jet impact
    """
    """Tested in xxxx example"""
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradientUnresolvedCoupling, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho_bar, s_rho_bar, d_idx, s_idx, d_p, s_p, s_m, d_au,
             d_av, d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho_bar[d_idx] * d_rho_bar[d_idx]
        rhoj2 = s_rho_bar[s_idx] * s_rho_bar[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class MomentumEquationArtificialViscosityUnresolvedCoupling(Equation):
    r"""Equation no 11 of [1]

    [1] A coupled SPH–DEM model for erosion process of solid surface
    by abrasive water-jet impact
    """
    """Tested in xxxx example"""
    def __init__(self, dest, sources, c0, alpha=0.1):
        self.alpha = alpha
        self.c0 = c0
        super(MomentumEquationArtificialViscosityUnresolvedCoupling, self).__init__(
            dest, sources
        )

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_au, d_av, d_aw,
             d_rho_bar,
             s_rho_bar,
             R2IJ, EPS, DWIJ, VIJ, XIJ, HIJ):

        # v_{ab} \cdot r_{ab}
        vijdotrij = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

        # scalar part of the accelerations Eq. (11)
        piij = 0.0
        if vijdotrij < 0:
            # inverse of sum of i and j rho values
            rho_ij = (d_rho_bar[d_idx] + s_rho_bar[s_idx]) / 2.

            # inverse of sum of i and j speed of sound
            # c_ij = (d_c[d_idx] + s_c[s_idx]) / 2.

            muij = (HIJ * vijdotrij)/(R2IJ + EPS)

            piij = -self.alpha*self.c0*muij
            piij = s_m[s_idx] * piij / rho_ij

        d_au[d_idx] += -piij * DWIJ[0]
        d_av[d_idx] += -piij * DWIJ[1]
        d_aw[d_idx] += -piij * DWIJ[2]


class SolidWallNoSlipBC(Equation):
    """Tested in xxxx example"""
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
               d_what, d_auhat, d_avhat, d_awhat, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]

        d_uhat[d_idx] = d_u[d_idx] + dtb2 * d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2 * d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2 * d_awhat[d_idx]

    def stage2(self, d_idx, d_x, d_y, d_z, d_rho_bar, d_arho_bar,
               d_ap, d_p, d_uhat, d_vhat, d_what, dt):
        d_rho_bar[d_idx] += dt * d_arho_bar[d_idx]
        # d_p[d_idx] += dt * d_ap[d_idx]

        d_x[d_idx] += dt * d_uhat[d_idx]
        d_y[d_idx] += dt * d_vhat[d_idx]
        d_z[d_idx] += dt * d_what[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, dt):
        dtb2 = 0.5 * dt
        d_u[d_idx] += dtb2 * d_au[d_idx]
        d_v[d_idx] += dtb2 * d_av[d_idx]
        d_w[d_idx] += dtb2 * d_aw[d_idx]


class ComputePressureAndDragForceOnSphericalParticles(Equation):
    """Tested in xxxx example"""

    def __init__(self, dest, sources, nu):
        self.nu = nu
        super(SolidWallNoSlipBC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_epsilon_porous, d_shepard_denom,
                   d_scaling_factor, d_u_bar, d_v_bar, d_w_bar):
        # Compute the porosity and effective velocity of the spherical particle
        # extrapolated from the fluid around
        d_epsilon_porous[d_idx] = 0.
        d_shepard_denom[d_idx] = 0.
        d_u_bar[d_idx] = 0.
        d_v_bar[d_idx] = 0.
        d_w_bar[d_idx] = 0.
        d_scaling_factor[d_idx] = 0.

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au, d_av,
             d_aw, d_u, d_v, d_w, s_p, s_rho_bar,
             d_fx, d_fy, d_fz, d_h, d_vol, d_shepard_denom,
             d_u_bar,
             d_v_bar,
             d_w_bar,
             s_u,
             s_v,
             s_w,
             d_epsilon_porous,
             s_epsilon_porous,
             d_scaling_factor,
             R2IJ, EPS,
             DWIJ, WIJ, XIJ):
        # Pressure force
        tmp = s_m[s_idx] * s_p[s_idx] / s_rho_bar[s_idx]
        d_fx[d_idx] += - d_vol[d_idx] * tmp * DWIJ(d_h[d_idx])
        d_fy[d_idx] += - d_vol[d_idx] * tmp * DWIJ(d_h[d_idx])
        d_fz[d_idx] += - d_vol[d_idx] * tmp * DWIJ(d_h[d_idx])

        # Compute the porosity and effective velocity of the spherical particle
        # extrapolated from the fluid around
        vol_j = s_m[s_idx] / s_rho[s_idx]
        d_epsilon_porous[d_idx] += - vol_j * s_epsilon_porous[s_idx] * WIJ(d_h[d_idx])
        d_u_bar[d_idx] += - s_u[s_idx] * vol_j * WIJ(d_h[d_idx])
        d_v_bar[d_idx] += - s_v[s_idx] * vol_j * WIJ(d_h[d_idx])
        d_w_bar[d_idx] += - s_w[s_idx] * vol_j * WIJ(d_h[d_idx])
        d_shepard_denom[d_idx] += vol_j * WIJ(d_h[d_idx])

    def post_loop(self, d_idx, d_rho, d_u, d_v, d_w,
                  d_fx, d_fy, d_fz,
                  d_shepard_denom,
                  d_u_bar,
                  d_v_bar,
                  d_w_bar,
                  d_m,
                  d_epsilon_porous,
                  d_rad_s,
                  d_h, d_vol):
        # divide the porosity by denominator
        d_u_bar[d_idx] /= d_shepard_denom[d_idx]
        d_v_bar[d_idx] /= d_shepard_denom[d_idx]
        d_w_bar[d_idx] /= d_shepard_denom[d_idx]

        d_epsilon_porous[d_idx] /= d_shepard_denom[d_idx]
        d_v_bar[d_idx] /= d_shepard_denom[d_idx]
        d_w_bar[d_idx] /= d_shepard_denom[d_idx]

        # effective velocity of the spherical particle in fluid
        u_eff = d_u_bar[d_idx] - d_u[d_idx]
        v_eff = d_v_bar[d_idx] - d_v[d_idx]
        w_eff = d_w_bar[d_idx] - d_w[d_idx]
        vmag_eff = (u_eff**2. + v_eff**2. + w_eff**2.)**0.5

        # Compute the interphase momentum exchange coefficient beta (eq 43 of Yu)
        eps_i = d_epsilon_porous[d_idx]
        diameter = 2. * d_rad_s[d_idx]
        tmp_1 = (1. - eps_i)**2. / eps_i * self.mu / diameter**2.
        tmp_2 = 1.75 * (1. - eps_i) * self.fluid_rho0 / diameter * vmag_eff
        beta_i = 150. * tmp_1 + tmp_2
        if eps_i > 0.8:
            Re_i = vmag_eff * eps_i * self.fluid_rho0 * diameter / self.mu

            Cd = 0.44
            if Re_i < 1000:
                Cd = 24 / Re_i * (1. + 0.15 * Re_i**0.687)
            beta_i = 0.75 * Cd * eps_i * (1. - eps_i) / diameter * self.fluid_rho0 * vmag_eff * eps_i**(-2.65)

        tmp = beta_i / (1. - eps_i) * d_m[d_idx] / d_rho[d_idx]
        d_fx[d_idx] = tmp * u_eff
        d_fy[d_idx] = tmp * v_eff
        d_fz[d_idx] = tmp * w_eff


class ComputeForceOnFluidDueToSphericalParticle(Equation):
    """Tested in xxxx example"""
    def initialize(self, d_idx, d_fx_coupled, d_fy_coupled, d_fz_coupled):
        d_fx_coupled[d_idx] = 0.
        d_fy_coupled[d_idx] = 0.
        d_fz_coupled[d_idx] = 0.

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m,
             d_u, d_v, d_w, s_p, s_rho_bar,
             d_fx, d_fy, d_fz, d_h, d_vol, d_shepard_denom,
             d_m,
             d_rho_bar,
             d_au,
             d_av,
             d_aw,
             s_fx_coupled,
             s_fy_coupled,
             s_fz_coupled,
             s_h,
             s_shepard_denom,
             d_fx_coupled,
             d_fy_coupled,
             d_fz_coupled,
             WIJ):
        vol_i = d_m[d_idx] / d_rho[d_idx]
        d_fx_coupled[d_idx] -= (vol_i / s_shepard_denom[s_idx] *
                                s_fx_coupled[s_idx] * WIJ(s_h[s_idx]))
        d_fy_coupled[d_idx] -= (vol_i / s_shepard_denom[s_idx] *
                                s_fy_coupled[s_idx] * WIJ(s_h[s_idx]))
        d_fz_coupled[d_idx] -= (vol_i / s_shepard_denom[s_idx] *
                                s_fz_coupled[s_idx] * WIJ(s_h[s_idx]))

    def post_loop(self, d_idx,
                  d_fx_coupled,
                  d_fy_coupled,
                  d_fz_coupled,
                  d_au,
                  d_av,
                  d_aw,
                  d_m):
        d_au[d_idx] += d_fx_coupled[d_idx] / d_m[d_idx]
        d_au[d_idx] += d_fy_coupled[d_idx] / d_m[d_idx]
        d_au[d_idx] += d_fz_coupled[d_idx] / d_m[d_idx]


class UnresolvedCouplingScheme(Scheme):
    def __init__(self, fluids, boundaries, spherical_particles, dim, c0, nu, rho0, pb=0.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.0):
        self.c0 = c0
        self.nu = nu
        self.rho0 = rho0
        self.pb = pb
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dim = dim
        self.alpha = alpha
        self.fluids = fluids
        self.boundaries = boundaries
        self.spherical_particles = spherical_particles
        self.solver = None

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=0.01,
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
        from pysph_dem.dem import DEMStep
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

        bodystep = DEMStep()
        bodies = self.spherical_particles

        for body in bodies:
            if body not in steppers:
                steppers[body] = bodystep

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

        all = self.fluids + self.boundaries
        # =========================#
        # stage 1 equations start
        # =========================#
        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(PorosityEquation(
                dest=fluid, sources=self.spherical_particles), )

        stage1.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquationUnresolvedCoupling(dest=fluid,
                                                            sources=all), )

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
                eqs.append(
                    ClampWallPressure(dest=boundary, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            if self.alpha > 0.:
                eqs.append(
                    MomentumEquationArtificialViscosityUnresolvedCoupling(
                        dest=fluid, sources=all, c0=self.c0,
                        alpha=self.alpha
                    )
                )

            if self.nu > 0.0:
                # eqs.append(
                #     MomentumEquationViscosity(
                #         dest=fluid, sources=self.fluids, nu=self.nu
                #     )
                # )
                if len(self.boundaries) > 0:
                    eqs.append(
                        SolidWallNoSlipBC(
                            dest=fluid, sources=self.boundaries, nu=self.nu
                        )
                    )
            eqs.append(
                MomentumEquationPressureGradientUnresolvedCoupling(
                    dest=fluid, sources=all, gx=self.gx, gy=self.gy,
                    gz=self.gz), )

        stage2.append(Group(equations=eqs, real=True))

        # eqs = []
        # for spherical_particles in self.spherical_particles:
        #     eqs.append(
        #         ComputePressureAndDragForceOnSphericalParticles(
        #             dest=spherical_particles, sources=self.fluids,
        #             nu=self.nu
        #         )
        #     )

        # stage2.append(Group(equations=eqs, real=True))

        # eqs = []
        # for fluid in self.fluids:
        #     eqs.append(
        #         ComputeForceOnFluidDueToSphericalParticle(
        #             dest=fluid, sources=self.spherical_particles,
        #         )
        #     )

        # stage2.append(Group(equations=eqs, real=True))

        return MultiStageEquations([stage1, stage2])
