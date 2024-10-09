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
from scipy.interpolate import CubicSpline, interp1d, PPoly, PchipInterpolator
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
options['model.system_bounds.theta.t_f'] = [1., 30.] # cycle period [s]

# indicate desired wind environment
options['user_options.wind.model'] = 'uniform'
options['user_options.wind.u_ref'] = 10.
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
ctrl_strategy  = 'closed_loop' #  choose between: 'open_loop'  & 'closed loop'
if ctrl_strategy == 'closed_loop':
    t_end = 1*trial.visualization.plot_dict['theta']['t_f']
else:
    t_end = 1.0*trial.visualization.plot_dict['theta']['t_f']

ctrl_type = 'ndi' # choose between 'ndi' & 'mpc
# adjust options for path tracking (incl. aero model)
tracking_options = {}
tracking_options = copy.deepcopy(options)
tracking_options = set_megawes_path_tracking_settings('ALM', tracking_options)


# set MPC options
N_mpc = 20 # MPC horizon (number of MPC windows in prediction horizon)
N_sim = 40  # closed-loop simulation steps
ts = t_end/N_sim # sampling time (length of one MPC window)

# MPC options
if ctrl_type == 'mpc':
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
else:
    tracking_options['ndi.N'] = N_sim
    tracking_options['ndi.plot_flag'] = False
    tracking_options['ndi.ref_interpolator'] = 'poly'
    tracking_options['ndi.u_param'] = 'zoh'

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

sim = awe.sim.Simulation(trial, ctrl_strategy, ctrl_type ,ts, tracking_options)
if ctrl_strategy == 'open_loop' or (ctrl_strategy == 'closed_loop' and ctrl_type == 'ndi'):
    t_grids_new = np.linspace(0, t_end, N_sim).squeeze()
    u_opt = np.array(trial.solution_dict['V_opt']['u']).squeeze()
    t_grids = np.array(trial.solution_dict['time_grids']['u'].full()).squeeze()
    interp_u = np.zeros((N_sim, trial.model.variables_dict['u'].shape[0])) #
    for i in range(trial.model.variables_dict['u'].shape[0]):  
        i_elements = u_opt[:,i]      
        # Create the periodic interpolation function
        # ***** interp1 *******
        # interp_func = interp1d(t_grids, i_elements, kind='slinear', bounds_error=False, fill_value='extrapolate')
        # interp_u[:,i] = interp_func(t_grids_new)
        # ***** CubicSpline *****
        cs = CubicSpline(t_grids.squeeze(), i_elements.squeeze(), axis=3,  bc_type='periodic', extrapolate='periodic')
        interp_u[:,i] = cs(t_grids_new)

        # interp_u[:,i] = np.interp(t_grids_new, t_grids.squeeze(), i_elements.squeeze(), period=t_end)
    sim.run(N_sim, u_sim = interp_u ) #interp_u
else:
    sim.run(N_sim)

        


sim.plot(['isometric', 'states' ,'aero_coefficients', 'aero_dimensionless'])
fig, axs = plt.subplots(2,1)
axs[0].plot(t_grids_new, interp_u[:,5:8], label='interpolated values',linestyle='--')
axs[0].step(t_grids, u_opt[:,5:8], label='initial values',linestyle='-')

axs[1].plot(t_grids_new, interp_u[:,9], label='interpolated values',linestyle='--')
axs[1].step(t_grids, u_opt[:,9], label='initial values',linestyle='-')
plt.show()

# %%
