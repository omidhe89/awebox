"""
Circular pumping trajectory for the 6DOF megAWES reference rigid-wing aircraft.

Aircraft dimensions adapted from:
"Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems",
Dylan Eijkelhof, Roland Schmehl
Renewable Energy, Vol.196, pp. 137-150, 2022.

Aerodynamic model and constraints from BORNE project (Ghent University, UCLouvain, 2024)

:author: Thomas Haas, Ghent University, 2024 (adapted from Jochem De Schutter)
"""
#%%
import awebox as awe
from megawes_settings import set_megawes_path_generation_settings, set_megawes_path_tracking_settings 
import matplotlib.pyplot as plt
import copy
import awebox.pmpc as pmpc
import numpy as np  
plt.ion()
#%%
# ----------------- user-specific options ----------------- #
# indicate aerodynamic model of aircraft
aero_model = 'ALM' # options are 'VLM', 'ALM', and 'CFD'

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_path_generation_settings(aero_model, options)


# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # positive (or null) reel-out speed during power generation
options['user_options.trajectory.lift_mode.windings'] = 1 # number of loops
options['model.system_bounds.theta.t_f'] = [1., 25.] # cycle period [s]

# indicate desired wind environment
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 12.
options['params.wind.z_ref'] = 100.
options['params.wind.log_wind.z0_air'] = 0.0002

# indicate numerical nlp details
options['nlp.n_k'] = 40 # approximately 40 per loop
options['nlp.collocation.u_param'] = 'zoh' # constant control inputs
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'
options['nlp.collocation.ineq_constraints'] = 'shooting_nodes' # default is 'shooting_nodes'

# ----------------- solve OCP ----------------- #

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'MegAWES')
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


trial.plot(['isometric'])
plt.show()


#%% 

# ----------------- create MPC controller with tracking-specific options ----------------- #

#t_end = 1*trial.visualization.plot_dict['theta']['t_f']
# adjust options for path tracking (incl. aero model)
tracking_options = {}
tracking_options = copy.deepcopy(options)
# tracking_options = set_megawes_path_tracking_settings('ALM', traj_options)

tracking_options['user_options.kite_standard'] = awe.megawes_data.data_dict('CFD')
# set MPC options
ts = 0.1 # sampling time (length of one MPC window)
N_mpc = 20 # MPC horizon (number of MPC windows in prediction horizon)
N_sim = 250  # closed-loop simulation steps

# MPC options
tracking_options['mpc.scheme'] = 'radau'
tracking_options['mpc.d'] = 4
tracking_options['mpc.jit'] = False
tracking_options['mpc.cost_type'] = 'tracking'
tracking_options['mpc.expand'] = True
tracking_options['mpc.linear_solver'] = 'ma57'
tracking_options['mpc.max_iter'] = 1000
tracking_options['mpc.max_cpu_time'] = 2000
tracking_options['mpc.N'] = N_mpc
tracking_options['mpc.plot_flag'] = False
tracking_options['mpc.ref_interpolator'] = 'spline'
tracking_options['mpc.u_param'] = 'zoh'
tracking_options['mpc.homotopy_warmstart'] = True
tracking_options['mpc.terminal_point_constr'] = False

# simulation options
N_dt = 20 # integrator steps within one sampling time
tracking_options['sim.number_of_finite_elements'] = N_dt
tracking_options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])



# # simulation options
# options['sim.number_of_finite_elements'] = 20 # integrator steps within one sampling time
# options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

# reduce average wind speed
# options['sim.sys_params']['wind']['u_ref'] = 1.0*options['sim.sys_params']['wind']['u_ref']

# make simulator
closed_loop_sim = awe.sim.Simulation(trial, 'closed_loop', ts, tracking_options)
closed_loop_sim.run(N_sim)
closed_loop_sim.plot(['isometric','states'])
plt.show()
# %%
