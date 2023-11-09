#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import cycle, product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'Helvetica', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


# n_core = 32
# n_thread = 32 * 2
n_core = 6
n_thread = n_core * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class Benchmark4SWCollidingOblique(Problem):
    def get_name(self):
        return 'benchmark_4_sw_colliding_oblique'

    def setup(self):
        get_path = self.input_path

        cmd = 'python examples/benchmark_4_sw_colliding_oblique.py' + backend

        velocity = 3.9
        fric_coeff = 0.092
        dt = 1e-7
        # Base case info
        self.case_info = {
            'angle_2': (dict(
                angle=2.,
                timestep=dt,
                ), 'Angle=2.'),

            'angle_5': (dict(
                angle=5.,
                timestep=dt,
                ), 'Angle=5.'),

            'angle_10': (dict(
                angle=10.,
                timestep=dt,
                ), 'Angle=10.'),

            'angle_15': (dict(
                angle=15.,
                timestep=dt,
                ), 'Angle=15.'),

            'angle_20': (dict(
                angle=20.,
                timestep=dt,
                ), 'Angle=20.'),

            'angle_25': (dict(
                angle=25.,
                timestep=dt,
                ), 'Angle=25.'),

            'angle_30': (dict(
                angle=30.,
                timestep=dt,
                ), 'Angle=30.'),

            'angle_35': (dict(
                angle=35.,
                timestep=dt,
                ), 'Angle=35.'),

            'angle_40': (dict(
                angle=40.,
                timestep=dt,
                ), 'Angle=40.'),

            'angle_50': (dict(
                angle=50.,
                timestep=dt,
                ), 'Angle=50.'),

            'angle_55': (dict(
                angle=55.,
                timestep=dt,
                ), 'Angle=55.'),

            'angle_60': (dict(
                angle=60.,
                timestep=dt,
                ), 'Angle=60.'),

            'angle_65': (dict(
                angle=65.,
                timestep=dt,
                ), 'Angle=65.'),

            'angle_70': (dict(
                angle=70.,
                timestep=dt,
                ), 'Angle=70.'),

            # 'angle_80': (dict(
            #     samples=samples,
            #     velocity=5.,
            #     angle=80.,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=80.'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       velocity=velocity,
                       fric_coeff=fric_coeff,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_theta_vs_omega()

    def plot_theta_vs_omega(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))
            angle_exp = data[name]['angle_exp']
            ang_vel_exp = data[name]['ang_vel_exp']
            angle_lethe = data[name]['angle_lethe']
            ang_vel_lethe = data[name]['ang_vel_lethe']

        angle_current = []
        ang_vel_current = []

        for name in self.case_info:
            angle_current.append(data[name]['angle_current'])
            ang_vel_current.append(data[name]['ang_vel_current'])

        plt.plot(angle_exp, ang_vel_exp, '*', label='Kharaz, Gorham and Salman (Exp)')
        plt.plot(angle_lethe, ang_vel_lethe, 'v-', label='Lethe')
        plt.plot(angle_current, ang_vel_current, '^-', label='Simulated')
        plt.xlabel('Angle')
        plt.ylabel(r'Angular Velocity')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('angle_vs_ang_vel.pdf'))
        plt.clf()
        plt.close()


class RB05SphericalImpactsWallKharazGorhamSalman(Problem):
    def get_name(self):
        return 'rb_05_spherical_impacts_wall_kharaz_gorham_salman'

    def setup(self):
        get_path = self.input_path

        cmd = 'python examples/rigid_body/05_spherical_rigid_body_impact_wall_Kharaz_Gorham_Salman.py' + backend
        velocity = 3.9
        fric_coeff = 0.092
        dt = 1e-7
        # Base case info
        self.case_info = {
            'angle_2': (dict(
                angle=2.,
                timestep=dt,
                ), 'Angle=2.'),

            'angle_5': (dict(
                angle=5.,
                timestep=dt,
                ), 'Angle=5.'),

            'angle_10': (dict(
                angle=10.,
                timestep=dt,
                ), 'Angle=10.'),

            'angle_15': (dict(
                angle=15.,
                timestep=dt,
                ), 'Angle=15.'),

            'angle_20': (dict(
                angle=20.,
                timestep=dt,
                ), 'Angle=20.'),

            'angle_25': (dict(
                angle=25.,
                timestep=dt,
                ), 'Angle=25.'),

            'angle_30': (dict(
                angle=30.,
                timestep=dt,
                ), 'Angle=30.'),

            'angle_35': (dict(
                angle=35.,
                timestep=dt,
                ), 'Angle=35.'),

            'angle_40': (dict(
                angle=40.,
                timestep=dt,
                ), 'Angle=40.'),

            'angle_50': (dict(
                angle=50.,
                timestep=dt,
                ), 'Angle=50.'),

            'angle_55': (dict(
                angle=55.,
                timestep=dt,
                ), 'Angle=55.'),

            'angle_60': (dict(
                angle=60.,
                timestep=dt,
                ), 'Angle=60.'),

            'angle_65': (dict(
                angle=65.,
                timestep=dt,
                ), 'Angle=65.'),

            'angle_70': (dict(
                angle=70.,
                timestep=dt,
                ), 'Angle=70.'),

            # 'angle_80': (dict(
            #     samples=samples,
            #     velocity=5.,
            #     angle=80.,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=80.'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       velocity=velocity,
                       fric_coeff=fric_coeff,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_theta_vs_omega()

    def plot_theta_vs_omega(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))
            angle_exp = data[name]['angle_exp']
            ang_vel_exp = data[name]['ang_vel_exp']
            angle_lethe = data[name]['angle_lethe']
            ang_vel_lethe = data[name]['ang_vel_lethe']

        angle_current = []
        ang_vel_current = []

        for name in self.case_info:
            angle_current.append(data[name]['angle_current'])
            ang_vel_current.append(data[name]['ang_vel_current'])

        plt.plot(angle_exp, ang_vel_exp, '*', label='Kharaz, Gorham and Salman (Exp)')
        plt.plot(angle_lethe, ang_vel_lethe, 'v-', label='Lethe')
        plt.plot(angle_current, ang_vel_current, '^-', label='Simulated')
        plt.xlabel('Angle')
        plt.ylabel(r'Angular Velocity')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('angle_vs_ang_vel.pdf'))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    PROBLEMS = [
        # Discrete element method benchmarks
        RB01SingleBodyTranslatingAndRotating,
        RB02SingleBodyHittingWallDifferentAnglesNoDamping,
        RB03SingleBodyHittingWallWithDamping,
        RB04TwoParticlesColliding,

        # Fluid benchmarks
        HS2DTank,

        # RFC benchmarks
        RFC01SingleParticleEnteringHSTank,
        RFC02TwoParticlesEnteringHSTank,
        RFC03ManyParticlesEnteringHSTank]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
    # Extra notes
    # Peng-Nan Sun, An accurate FSI-SPH

# contact_force_normal_x[0::2], contact_force_normal_y[0::2], contact_force_normal_z[0::2]
# contact_force_normal_x[1::2], contact_force_normal_y[1::2], contact_force_normal_z[1::2]
# au_contact, av_contact, aw_contact

        # vyas_2021_rebound_kinematics_3d_compare_flipped(),  # Done
