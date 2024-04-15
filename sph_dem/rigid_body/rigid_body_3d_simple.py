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
from sph_dem.rigid_body.rigid_body_3d import (RigidBody3DScheme,
                                                UpdateSlaveBodyState)

from sph_dem.dem_simple import (
    BodyForce,
    SSHertzContactForce,
    SWHertzContactForce)

from numpy import sin, cos


class RigidBody3DSimpleScheme(RigidBody3DScheme):
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
            # g1 = []
            # for body in self.rigid_bodies_master:
            #     g1.append(
            #         # see the previous examples and write down the sources
            #         UpdateTangentialContacts(
            #             dest=body, sources=self.rigid_bodies_master))
            # stage2.append(Group(equations=g1, real=False))

            g2 = []
            for body in self.rigid_bodies_master:
                g2.append(
                    BodyForce(dest=name,
                              sources=None,
                              gx=self.gx,
                              gy=self.gy,
                              gz=self.gz))

            for body in self.rigid_bodies_master:
                g2.append(SSHertzContactForce(dest=body,
                                              sources=self.rigid_bodies_master,
                                              en=self.en,
                                              fric_coeff=self.fric_coeff))
                if len(self.boundaries) > 0:
                    g2.append(SWHertzContactForce(dest=body,
                                                  sources=self.boundaries,
                                                  en=self.en,
                                                  fric_coeff=self.fric_coeff))

            stage2.append(Group(equations=g2, real=False))

        return MultiStageEquations([stage1, stage2])
