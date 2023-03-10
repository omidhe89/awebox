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

# indicate desired system architecture
# here: single kite with 6DOF megAWES aircraft
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_settings(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1
#options['model.system_bounds.theta.t_f'] = [10.0, 160] # additional constraints limiting path period 80 -->160

# indicate desired environment
# here: uniform wind velocity profile
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'uniform'
options['user_options.wind.u_ref'] = 10.

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
options['nlp.n_k'] = 40
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'megAWES')
trial.build()
trial.optimize()
#trial.write_to_csv('megAWES_outputs_RENAME')
#trial.plot(['states', 'controls', 'constraints', 'quad'])
#plt.show()

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
figname = './megawes_trajectory_isometric.png'
fig.savefig(figname)

