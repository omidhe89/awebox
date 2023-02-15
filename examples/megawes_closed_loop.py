#!/usr/bin/python3
"""
Closed-loop MPC simulation of circular pumping trajectory for the Ampyx AP2 aircraft.
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

User settings
N_nlp: Number of NLP discretization intervals of the reference flight path  
N_mpc: Number of NLP intervals in the tracking window (MPC horizon)
N_sim: Number of MPC evaluations in closed-loop simulations
N_dt:  Number of built-in integrator steps within one MPC sampling time
t_s:   Sampling time of the MPC controller
"""

import awebox as awe
from megawes_settings import set_megawes_settings
import matplotlib.pyplot as plt
import numpy as np
import copy

# indicate desired system architecture
# here: single kite with 6DOF MegAWES model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_settings(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1
options['model.system_bounds.theta.t_f'] = [10.0, 80.0] # additional constraints limiting path period

# indicate desired environment
# here: wind velocity profile according to power-law
options['params.wind.z_ref'] = 10.0
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 5.

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
N_nlp = 60
options['nlp.n_k'] = N_nlp
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'megAWES')
trial.build()
trial.optimize()

# # draw some of the pre-coded plots for analysis
# trial.plot(['states', 'controls', 'constraints','quad'])
# plt.show()

# set-up closed-loop simulation
N_mpc = 10  # MPC horizon
N_sim = 250 # Closed-loop simulation steps
N_dt = 20   # Built-in integrator steps within one sampling time
t_s = 0.1   # MPC sampling time

# MPC options
options['mpc.scheme'] = 'radau'
options['mpc.d'] = 4
options['mpc.jit'] = False
options['mpc.cost_type'] = 'tracking'
options['mpc.expand'] = True
options['mpc.linear_solver'] = 'ma57'
options['mpc.max_iter'] = 1000
options['mpc.max_cpu_time'] = 2000
options['mpc.N'] = N_mpc
options['mpc.plot_flag'] = False
options['mpc.ref_interpolator'] = 'spline'
options['mpc.u_param'] = 'zoh'
options['mpc.homotopy_warmstart'] = True
options['mpc.terminal_point_constr'] = False

# simulation options
options['sim.number_of_finite_elements'] = N_sim 
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

# reduce average wind speed
options['sim.sys_params']['wind']['u_ref'] = 1.0*options['sim.sys_params']['wind']['u_ref']

# make simulator
closed_loop_sim = awe.sim.Simulation(trial, 'closed_loop', t_s, options)
closed_loop_sim.run(N_sim)

# Plot 3D path
trial.plot(['isometric'])
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax = fig.get_axes()[0]
l = ax.get_lines()
l[0].set_color('b')

# add built-in MPC simulation
x_mpc = np.array(closed_loop_sim.visualization.plot_dict['x']['q10']).T
ax.plot(x_mpc[:,0], x_mpc[:,1], x_mpc[:,2])
l = ax.get_lines()
l[-1].set_color('r')

# layout/save
ax.get_legend().remove()
ax.legend([l[-1], l[0]], ['mpc', 'ref'], fontsize=14)
figname = './megawes_closed_loop_isometric.png'
fig.savefig(figname)
