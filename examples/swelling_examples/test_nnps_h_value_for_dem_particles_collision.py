"""In this example we will test our formulation of handling many particles in
contact. We take 5 particles and allow them to collide.

Run the simulation to understand the example better.
"""
import numpy as np

from pysph.base.kernels import CubicSpline
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.examples.solid_mech.impact import add_properties

from pysph.tools.geometry import get_2d_block
from pysph_dem.dem import DEMScheme, setup_dem_particles, setup_wall_dem


class B1SSCollideNormalElastic(Application):
    def add_user_options(self, group):
        # group.add_argument(
        #     "--re", action="store", type=float, dest="re", default=0.0125,
        #     help="Reynolds number of flow."
        # )

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=10,
                           help="Number of particles in diamter of a rigid cylinder")

        # group.add_argument(
        #     "--remesh", action="store", type=float, dest="remesh", default=0,
        #     help="Remeshing frequency (setting it to zero disables it)."
        # )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        # self.re = self.options.re
        # ======================
        # ======================

        self.sphere_radius = 0.01
        self.sphere_h = 1. * self.sphere_radius * 2.
        self.sphere_rho = 2800
        self.sphere_velocity = 10
        self.en = 1.

        self.h = self.sphere_radius * 1 / 1.25

        # self.solid_rho = 500
        # self.m = 1000 * self.dx * self.dx
        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.dim = 2
        self.tf = 0.1
        self.dt = 1e-4

    def create_particles(self):
        r = self.sphere_radius
        x = np.array([0., r, -r, r + r/100, -r - r / 100.])
        y = np.array([0., 2 * r + r / 100, 2 * r + r / 100, 4 * r + r/100, 4 * r + r/100])
        u = np.array([0., 0., 0., 0., 0.])
        v = np.array([1., 0., 0., 0., 0.])
        m_b = 4/3. * np.pi * self.sphere_radius**3. * self.sphere_rho
        _I = 2. / 5. * m_b * self.sphere_radius**2
        I_inverse = 1. / _I
        E = np.ones_like(x) * 1e5
        nu = np.ones_like(x) * 0.2

        spheres = get_particle_array(name='spheres',
                                     x=x,
                                     y=y,
                                     v=v,
                                     h=self.h,
                                     m_b=m_b,
                                     I_inverse=I_inverse,
                                     rho_b=self.sphere_rho,
                                     rad_s=self.sphere_radius,
                                     E=E,
                                     nu=nu)
        dem_id = np.array([0, 0, 0, 0, 0])
        spheres.add_property('dem_id', type='int', data=dem_id)
        setup_dem_particles(spheres, 3)

        spheres.add_output_arrays(['ss_overlap', 'ss_fn'])
        return [spheres]

    def create_scheme(self):
        dem = DEMScheme(
            dem_particles=['spheres'],
            boundaries=None,
            dim=0,
            en=1.
        )
        s = SchemeChooser(default='dem', dem=dem)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            en=self.en
           )

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=10)
        print("dt = %g"%self.dt)

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output
        import os

        info = self.read_info(fname)
        files = self.output_files
        y_current = []
        v_current = []
        overlap_current = []
        fn_current = []
        u = []
        t_current = []

        for sd, spheres in iter_output(files, 'spheres'):
            t_current.append(sd['t'])
            overlap_current.append(spheres.ss_overlap[1])
            fn_current.append(spheres.ss_fn[1])
        overlap_current = np.asarray(overlap_current) * 1e6
        fn_current = np.asarray(fn_current) / 1e3

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        # data_fn_analytical = np.loadtxt(os.path.join(
        #     directory, 'benchmark_1_ss_colliding_elastic_fn_analytical.csv'), delimiter=',')

        # overlap_fn_analy, fn_analy = data_fn_analytical[:, 0], data_fn_analytical[:, 1]

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 # overlap_fn_analy=overlap_fn_analy,
                 # fn_analy=fn_analy,
                 t_current=t_current,
                 fn_current=fn_current)
        # ========================
        # Variation of y velocity
        # ========================
        plt.clf()
        # plt.plot(overlap_fn_analy, fn_analy, label='Analytical')
        plt.plot(overlap_current, fn_current, "*", label='Current')

        plt.title('Variation in fn force')
        plt.xlabel('overlap')
        plt.ylabel('fn-force')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "fn_vs_overlap.png")
        plt.savefig(fig, dpi=300)
        # ============================
        # Variation of y velocity ends
        # ============================


if __name__ == '__main__':
    app = B1SSCollideNormalElastic()
    app.run()
    app.post_process(app.info_filename)
