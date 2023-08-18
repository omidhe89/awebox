#!/usr/bin/python3
"""
Circular pumping trajectory for the 6DOF megAWES reference rigid-wing aircraft.
Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

Aircraft dimensions and constraints as in:

"Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems",
Dylan Eijkelhof, Roland Schmehl
Renewable Energy, Vol.196, pp. 137-150, 2022.

:author: Jochem De Schutter
:edited: Thomas Haas
"""

import awebox as awe
from megawes_settings import set_megawes_settings
import matplotlib.pyplot as plt
import numpy as np

# ----------------- user-specific options ----------------- #

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_settings(options)

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # positive (or null) reel-out speed during power generation
options['user_options.trajectory.lift_mode.windings'] = 1 # number of loops
options['model.system_bounds.theta.t_f'] = [1., 1e3] # cycle period [s]

# indicate desired environment
options['user_options.wind.model'] = 'uniform'
options['user_options.wind.u_ref'] = 10.
'''
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 10.
options['params.wind.z_ref'] = 100.0
options['params.wind.log_wind.z0_air'] = 0.0002
'''

# indicate numerical nlp details
options['nlp.n_k'] = 40 # approximately 40 per loop
options['nlp.collocation.u_param'] = 'zoh' # constant control inputs
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'

# specify trial name for outputs
trial_name = 'megawes_uniform_1loop'

# ----------------- solve OCP ----------------- #
# build and optimize the NLP (trial)
trial = awe.Trial(options, trial_name)
trial.build()
trial.optimize()

# save results to csv
trial.write_to_csv(trial_name+'_results')

# plot results
list_of_plots = ['isometric', 'states', 'controls', 'constraints']
trial.plot(list_of_plots)
for i, plot_name in enumerate(list_of_plots, start=1):
    plt.figure(i)
    plt.savefig('./'+trial_name+'_plot_'+plot_name+'.png')
#plt.show

'''
# extract data for post-processing
plot_dict = trial.visualization.plot_dict
outputs = plot_dict['outputs']
time = plot_dict['time_grids']['ip']
avg_power = plot_dict['power_and_performance']['avg_power']/1e3

# plot reference path (options are: 'states', 'controls', 'constraints', 'quad'
trial.plot(['isometric'])
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax = fig.get_axes()[0]
l = ax.get_lines()
l[0].set_color('b')
ax.get_legend().remove()
ax.legend([l[0]], ['ref'], fontsize=14)
figname = './megawes_trajectory_isometric.png'
fig.savefig(figname)
'''

