#!/usr/bin/python3
"""
Periodic power optimization of flight path of a single 3DOF kite system.
Aircraft model uses dimensions of AP2 with default AWEbox options

:author: Jochem De Schutter
:edited: Thomas Haas
"""

# imports
import awebox as awe
import casadi as ca
import numpy as np
import copy
import matplotlib.pyplot as plt
from awebox.logger.logger import Logger as awelogger
#awelogger.logger.setLevel('DEBUG') # Detailed information, typically of interest only when diagnosing problems.

# single kite with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options['user_options.kite_standard'] = awe.ampyx_data.data_dict()
options['user_options.system_model.kite_dof'] = 3
options['model.tether.control_var'] = 'ddl_t'

# trajectory should be a single pumping cycle
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1
options['model.system_bounds.theta.t_f'] =  [5.0, 15.0]
    
# wind model
options['params.wind.z_ref'] = 10.0
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 5.

# NLP discretization
options['nlp.n_k'] = 20
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57'
options['solver.mu_hippo'] = 1e-2

# initialize and optimize trial
trial = awe.Trial(options, 'single_kite_lift_mode')
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
trial.plot(['isometric'])
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax = fig.get_axes()[0]
l = ax.get_lines()
l[0].set_color('b')
ax.get_legend().remove()
ax.legend([l[0]], ['ref'], fontsize=14)
figname = './3dof_trajectory_isometric.png'
fig.savefig(figname)