"""
Investigations on the hydroelastic slamming of deformable wedges by using the smoothed particle element method

https://www.sciencedirect.com/science/article/pii/S0889974622001311#sec2
"""
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

from sph_dem.fluids import (ContinuityEquation, StateEquation,
                            StateEquationInternalFlow, MomentumEquationPressureGradient,
                            FluidStep, SolidWallNoSlipBC, EDACEquation,
                            DeltaSPHCorrection, ComputeAuHatGTVF, GTVFSetP0)
from sph_dem.rigid_body.rigid_body_3d import (UpdateSlaveBodyState,
                                                UpdateTangentialContacts,
                                                BodyForce,
                                                SSHertzContactForce,
                                                SSDMTContactForce,
                                                SWHertzContactForce,
                                                SumUpExternalForces,
                                                GTVFRigidBody3DMasterStep)

from sph_dem.rigid_body.rigid_body_3d import (RBStirrerForce)


def add_rigid_fluid_properties_to_rigid_body(pa):
    add_properties(pa, 'arho')
    add_properties(pa, 'wij')
    add_properties(pa, 'ug', 'vf', 'uf', 'wf', 'vg', 'wg')

    add_properties(pa, 'fx_p', 'fy_p', 'fz_p')
    add_properties(pa, 'fx_v', 'fy_v', 'fz_v')


class RigidFluidPressureForce(Equation):
    """Force on rigid body due to the interaction with fluid.
    The force equation is taken from SPH-DCDEM paper by Canelas

    nu: dynamics viscosity of the fluid

    """
    def __init__(self, dest, sources, nu=0):
        self.nu = nu
        super(RigidFluidPressureForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_fx_p, d_fy_p, d_fz_p):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.

        d_fx_p[d_idx] = 0.
        d_fy_p[d_idx] = 0.
        d_fz_p[d_idx] = 0.

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_m, d_fx, d_fy,
             d_fz, d_fx_p, d_fy_p, d_fz_p,
             DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ, R2IJ, EPS, VIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        # pressure forces
        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -d_m[d_idx] * s_m[s_idx] * pij

        d_fx[d_idx] += tmp * DWIJ[0]
        d_fy[d_idx] += tmp * DWIJ[1]
        d_fz[d_idx] += tmp * DWIJ[2]

        d_fx_p[d_idx] += tmp * DWIJ[0]
        d_fy_p[d_idx] += tmp * DWIJ[1]
        d_fz_p[d_idx] += tmp * DWIJ[2]

        # # viscous forces
        # xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        # tmp_1 = s_m[s_idx] * 4 * self.nu * xdotdij
        # tmp_2 = (d_rho[d_idx] + s_rho[s_idx]) * (R2IJ + EPS)
        # fac = tmp_1 / tmp_2

        # d_fx[d_idx] += d_m[d_idx] * fac * VIJ[0]
        # d_fy[d_idx] += d_m[d_idx] * fac * VIJ[1]
        # d_fz[d_idx] += d_m[d_idx] * fac * VIJ[2]


class RigidFluidViscousForce(Equation):
    """Force on rigid body due to the interaction with fluid.
    The force equation is taken from SPH-DCDEM paper by Canelas

    nu: dynamics viscosity of the fluid

    """
    def __init__(self, dest, sources, nu=0):
        self.nu = nu
        super(RigidFluidViscousForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz,
                   d_fx_v, d_fy_v, d_fz_v):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.
        d_fx_v[d_idx] = 0.
        d_fy_v[d_idx] = 0.
        d_fz_v[d_idx] = 0.

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_m, d_fx, d_fy,
             d_fz,
             d_fx_v,
             d_fy_v,
             d_fz_v,
             DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ, R2IJ, EPS, VIJ):
        # viscous forces
        xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp_1 = s_m[s_idx] * 4 * self.nu * xdotdij
        tmp_2 = (d_rho[d_idx] + s_rho[s_idx]) * (R2IJ + EPS)
        fac = tmp_1 / tmp_2

        d_fx[d_idx] += d_m[d_idx] * fac * VIJ[0]
        d_fy[d_idx] += d_m[d_idx] * fac * VIJ[1]
        d_fz[d_idx] += d_m[d_idx] * fac * VIJ[2]

        d_fx_v[d_idx] += d_m[d_idx] * fac * VIJ[0]
        d_fy_v[d_idx] += d_m[d_idx] * fac * VIJ[1]
        d_fz_v[d_idx] += d_m[d_idx] * fac * VIJ[2]


class MomentumEquationViscosityRigidBody(Equation):
    """BD Rogers FSI paper equation 26"""
    def __init__(self, dest, sources, nu=1e-6):
        self.nu = nu
        super(MomentumEquationViscosityRigidBody, self).__init__(dest, sources)

    def loop(self, d_idx, s_idx, d_rho, s_rho, s_m, d_au,
             d_av, d_aw, VIJ, R2IJ, EPS, DWIJ, XIJ):
        etai = self.nu * d_rho[d_idx]
        etaj = self.nu * s_rho[s_idx]

        etaij = 4 * (etai * etaj)/(etai + etaj)

        xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        tmp = s_m[s_idx]/(d_rho[d_idx] * s_rho[s_idx])
        fac = tmp * etaij * xdotdij/(R2IJ + EPS)

        d_au[d_idx] += fac * VIJ[0]
        d_av[d_idx] += fac * VIJ[1]
        d_aw[d_idx] += fac * VIJ[2]


class ParticlesFluidScheme(Scheme):
    def __init__(self, fluids, boundaries,
                 rigid_bodies_master,
                 rigid_bodies_slave,
                 rigid_bodies_wall,
                 stirrer,
                 dim, c0, nu, rho0, h,
                 pb=0.0, gx=0.0, gy=0.0, gz=0.0,
                 alpha=0.0, en=1.0, fric_coeff=0.5,
                 en_wall=1.0, fric_coeff_wall=0.5,
                 gamma=0.):
        self.c0 = c0
        self.nu = nu
        self.h = h
        self.rho0 = rho0
        self.pb = pb
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dim = dim
        self.alpha = alpha
        self.fluids = fluids
        self.boundaries = boundaries
        self.solver = None

        self.rigid_bodies_master = rigid_bodies_master
        self.rigid_bodies_slave = rigid_bodies_slave

        self.rigid_bodies_wall = rigid_bodies_wall
        if stirrer is None:
            self.stirrer = []
        else:
            self.stirrer = stirrer

        self.en = en
        self.fric_coeff = fric_coeff
        self.en_wall = en_wall
        self.fric_coeff_wall = fric_coeff_wall
        self.gamma = gamma

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=0.1,
                           help="Alpha for the artificial viscosity.")

        # group.add_argument("--kr-stiffness", action="store",
        #                    dest="kr", default=1e5,
        #                    type=float,
        #                    help="Repulsive spring stiffness")

        # group.add_argument("--kf-stiffness", action="store",
        #                    dest="kf", default=1e3,
        #                    type=float,
        #                    help="Tangential spring stiffness")

        group.add_argument("--fric-coeff", action="store",
                           dest="fric_coeff", default=0.5,
                           type=float,
                           help="Friction coefficient")

        group.add_argument("--en", action="store",
                           dest="en", default=0.1,
                           type=float,
                           help="Coefficient of restitution")

        group.add_argument("--fric-coeff-wall", action="store",
                           dest="fric_coeff_wall", default=0.5,
                           type=float,
                           help="Friction coefficient wall")

        group.add_argument("--en-wall", action="store",
                           dest="en_wall", default=0.1,
                           type=float,
                           help="Coefficient of restitution wall")

        group.add_argument("--gamma", action="store",
                           dest="gamma", default=0.0,
                           type=float,
                           help="Surface energy")

        group.add_argument("--nu", action="store",
                           dest="nu", default=0.,
                           type=float,
                           help="Coefficient of viscosity")

        # add_bool_argument(
        #     group,
        #     'two-way-coupling',
        #     dest='two_way_coupling',
        #     default=True,
        #     help='Should we handle the two way coupling')

    def consume_user_options(self, options):
        vars = [
            'alpha',
            'fric_coeff',
            'en',
            'fric_coeff_wall',
            'en_wall',
            'gamma',
            'nu'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        self.edac_alpha = self._get_edac_nu()
        self.edac_nu = self.edac_alpha * self.h * self.c0 / 8.

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

        bodystep = GTVFRigidBody3DMasterStep()
        bodies = self.rigid_bodies_master

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

        from pysph.sph.wc.transport_velocity import (
            MomentumEquationArtificialViscosity
        )

        # set the rigid body particle array which interacts with the fluid
        rb_fluid = self.rigid_bodies_slave

        all = self.fluids + self.boundaries
        # =========================#
        # stage 1 equations start
        # =========================#
        stage1 = []

        if len(self.rigid_bodies_slave) > 0:
            g5 = []
            for name in self.rigid_bodies_master:
                g5.append(
                    UpdateSlaveBodyState(dest=name[:-6:]+"slave",
                                         sources=[name]))

            stage1.append(Group(equations=g5, real=False))

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquation(dest=fluid,
                                          sources=all), )
            eqs.append(ContinuityEquation(dest=fluid,
                                          sources=rb_fluid))

            eqs.append(
                EDACEquation(dest=fluid,
                             sources=all + self.rigid_bodies_slave,
                             nu=self.edac_nu))

        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []

        if len(self.rigid_bodies_slave) > 0:
            g5 = []
            for name in self.rigid_bodies_master:
                g5.append(
                    UpdateSlaveBodyState(dest=name[:-6:]+"slave",
                                         sources=[name]))

            stage2.append(Group(equations=g5, real=False))

        # tmp = []
        # for fluid in self.fluids:
        #     tmp.append(
        #         StateEquation(dest=fluid,
        #                       sources=None,
        #                       rho0=self.rho0,
        #                       p0=self.pb))

        # stage2.append(Group(equations=tmp, real=False))

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

        if len(rb_fluid) > 0:
            eqs = []
            for body in rb_fluid:
                eqs.append(SetWallVelocity(dest=body, sources=self.fluids))
            stage2.append(Group(equations=eqs, real=False))

        if len(rb_fluid) > 0:
            eqs = []
            for body in rb_fluid:
                eqs.append(
                    SourceNumberDensity(dest=body, sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=body, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressure(dest=body, sources=None))

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
                    MomentumEquationViscosityRigidBody(
                        dest=fluid, sources=rb_fluid, nu=self.nu
                    )
                )

            eqs.append(
                MomentumEquationPressureGradient(dest=fluid,
                                                 sources=all+rb_fluid,
                                                 gx=self.gx, gy=self.gy,
                                                 gz=self.gz), )

        stage2.append(Group(equations=eqs, real=True))

        # Compute the force on the rigid bodies due to fluid
        if len(rb_fluid) > 0:
            eqs = []
            for body in rb_fluid:
                eqs.append(
                    RigidFluidPressureForce(dest=body, sources=self.fluids,
                                            nu=self.nu))
                eqs.append(
                    RigidFluidViscousForce(dest=body,
                                           sources=self.fluids,
                                           nu=self.nu))

            stage2.append(Group(equations=eqs, real=True))

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
            # g2.append(SSDMTContactForce(dest=body,
            #                             sources=self.rigid_bodies_master,
            #                             gamma=self.gamma))

            if len(self.rigid_bodies_wall) > 0:
                g2.append(SWHertzContactForce(dest=body,
                                              sources=self.rigid_bodies_wall,
                                              en=self.en_wall,
                                              fric_coeff=self.fric_coeff_wall))

        stage2.append(Group(equations=g2, real=False))

        # Compute force on the rigid body slave due to interaction with the
        # stirrer
        if len(self.stirrer) > 0:
            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies_slave:
                g6.append(RBStirrerForce(dest=name, sources=self.stirrer,
                                         en=self.en,
                                         fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g6, real=False))

        # computation of total force and torque at center of mass
        g6 = []
        for name in self.rigid_bodies_master:
            g6.append(SumUpExternalForces(
                dest=name, sources=[name[:-6:]+"slave"]))

        stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def _get_edac_nu(self):
        if self.alpha > 0:
            nu = self.alpha

            # print(self.alpha)
            # print("using artificial viscosity for edac with nu = %s" % nu)
        else:
            nu = self.nu
            # print("using real viscosity for edac with nu = %s" % self.nu)
        return nu
