#!/usr/bin/python3
"""
Periodic power optimization of flight path of a single 3DOF kite system.
Aircraft model uses dimensions of AP2 with default AWEbox options

:author: Jochem De Schutter
:edited: Thomas Haas
"""

# %%
import awebox as awe
import casadi as ca
import numpy as np
import copy
import matplotlib.pyplot as plt
from awebox.logger.logger import Logger as awelogger
# awelogger.logger.setLevel('DEBUG') # Detailed information, typically of interest only when diagnosing problems.

# single kite with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1: 0}
options['user_options.system_model.kite_dof'] = 3
options['user_options.system_model.kite_type'] = 'soft'

if options['user_options.system_model.kite_type'] == 'rigid':
    options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
    options['solver.initialization.shape'] = 'circular'
    options['model.system_bounds.theta.t_f'] = [5.0, 15.0]
else:
    options['user_options.kite_standard'] = awe.kitepower_data.data_dict()
    options['params.tether.rho'] = 724.0
    options['params.tether.cd'] = 1.1
    options['params.model_bounds.tether_force_limits'] = np.array([1e3, 5e3])
    options['model.model_bounds.acceleration.include'] = False
    options['model.model_bounds.tether_force.include'] = False
    options['params.tether.f_max'] = 1.2  # tether stress safety factor
    # system bounds
    # TODO get rid of redundant option
    options['model.ground_station.ddl_t_max'] = 2.0
    options['model.ground_station.dddl_t_max'] = 50.0
    options['model.system_bounds.x.dl_t'] = [-15.0, 15.0]  # m/s
    options['model.system_bounds.x.pitch'] = [0.0, np.pi/6]
    options['model.system_bounds.u.dpitch'] = [-0.9, 0.9]
    options['model.system_bounds.u.dyaw'] = [-1, 1]
    options['model.system_bounds.x.q'] = [
        np.array([0, -200, 30.0]),  # q_z > 30 m
        np.array([np.inf, 200, 700])]
    
    # initialization
    options['solver.initialization.init_clipping'] = True  # False if you want to get rid of clipping initial guesses -> circumference and et.al that are used for winding period calculation.
    options['solver.initialization.shape'] = 'lemniscate'
    options['solver.initialization.lemniscate.az_width'] = 30.0*np.pi/180.0
    options['solver.initialization.lemniscate.el_width'] = 15.0*np.pi/180.0
    options['solver.initialization.inclination_deg'] = 25.
    options['solver.initialization.groundspeed'] = 35.
    options['solver.initialization.theta.diam_t'] = 5e-3
    options['solver.initialization.l_t'] = 100.0
    options['model.system_bounds.theta.t_f'] = [0, 100]

options['model.tether.control_var'] = 'ddl_t'
options['model.model_bounds.aero_validity.include'] = True
options['model.tether.use_wound_tether'] = True
# it can cause syntax error!!
options['user_options.induction_model'] = 'not_in_use'



# trajectory should be a single pumping cycle
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 3


# wind model
options['params.wind.z_ref'] = 10.0
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 9.

# NLP discretization
options['nlp.n_k'] = 200
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout'
options['nlp.phase_fix_reelout'] = 0.7
options['solver.linear_solver'] = 'ma57'
options['solver.max_iter'] = 5000
options['solver.max_iter_hippo'] = 2000

# initialize and optimize trial
trial = awe.Trial(options, 'single_soft_kite_lift_mode')
trial.build()
trial.optimize()

# extract information from the solution for independent plotting or post-processing
# here: plot relevant system outputs, compare to [Licitra2019, Fig 11].
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3
print('======================================')
print('Average power: {} kW'.format(avg_power))
print('======================================')

# plot reference path (options are: 'states', 'controls', 'constraints', 'quad'
# trial.plot(['isometric', 'states', 'controls'])

# fig = plt.gcf()
# fig.set_size_inches(10, 8)
# ax = fig.get_axes()[0]
# l = ax.get_lines()
# l[0].set_color('b')
# ax.get_legend().remove()
# ax.legend([l[0]], ['ref'], fontsize=14)
# figname = './3dof_trajectory_isometric.png'
# fig.savefig(figname)
# fig.show()

# draw additional plots
# plt.subplots(3, 1, sharex=False)
# plt.subplot(311)
# plt.plot(time, plot_dict['x']['l_t'][0], label='Tether Length')
# plt.ylabel('[m]')
# plt.legend()
# plt.grid(True)

# plt.subplot(312)
# plt.plot(time, plot_dict['x']['dl_t'][0], label='Tether Reel-out Speed')
# plt.ylabel('[m/s]')
# plt.legend()
# plt.hlines([20, -15], time[0], time[-1], linestyle='--', color='black')
# plt.grid(True)

# plt.subplot(313)
# plt.plot(time, outputs['aerodynamics']['airspeed1'][0], label='Airspeed')
# plt.ylabel('[m/s]')
# plt.legend()
# plt.hlines([10, 32], time[0], time[-1], linestyle='--', color='black')
# plt.grid(True)

# %%
