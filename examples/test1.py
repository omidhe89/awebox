#!/usr/bin/python3
"""
MPC-based closed loop simulation example for a single 3DOF kite system.
"""

#____________________________________________________________________
# module imports...

# awebox
import awebox as awe
from ampyx_ap2_settings import set_ampyx_ap2_settings


import awebox.pmpc as pmpc
# import awebox.opts.options as opts
# import awebox.mdl.model as model
# import awebox.mdl.architecture as archi
# from awebox.logger.logger import Logger as awelogger

# casadi
import casadi as ca

# python packages
import matplotlib.pyplot as plt
import numpy as np
import copy
# from scipy.interpolate import interp1d
# import pickle as pckl
# import time
# import os

# ap2 module
# from ampyx_ap2_settings import set_ampyx_ap2_settings
# import trial_optimization as opti
# import mpc_generation as mpc_gen
# import data_visuals as visu
# import external_simulator as ext

# ____________________________________________________________________
# User parameters ...
trial_name = 'Ampyx_AP2' # Name of optimization trial
N_mpc = 10 # Number of NLP intervals in the tracking window
t_sam = 0.1 # Sampling time of the MPC controller
N_sim = 10 # integrator steps within one sampling time
N_step = 40 # Number of MPC evaluations in built-in/F_ext simulations

# ____________________________________________________________________
# trajectory optimization...

# single 6DOF AP2 AWES
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_ampyx_ap2_settings(options)

# trajectory should be a single pumping cycle
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1

# wind model
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# NLP discretization
options['nlp.n_k'] = 40
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57'

# trial options
trial_opts = awe.opts.options.Options()
trial_opts.fill_in_seed(options)

# get architecture
archi = awe.mdl.architecture.Architecture(trial_opts['user_options']['system_model']['architecture'])

# initialize, build and optimize trial
trial = awe.Trial(options, trial_name)
trial.build()
trial.optimize()
trial.plot(['isometric', 'states'])
plt.show()

# ____________________________________________________________________
# built-in closed-loop simulation with default options
ts = 0.1
N_sim = 40

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
options['sim.number_of_finite_elements'] = 20 # integrator steps within one sampling time
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

closed_loop_sim = awe.sim.Simulation(trial, 'closed_loop', ts, options)
closed_loop_sim.run(N_sim)


closed_loop_sim.plot('isometric')
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax = fig.get_axes()[0]
l = ax.get_lines()
if len(l) % 2 == 0:
    n = int(len(l)/2)
else:
    n = int(len(l) / 2) + 1
l[0].set_color('r')
l[n].set_color('b')
ax.get_legend().remove()
ax.legend([l[0], l[n]], ['mpc', 'ref'], fontsize=14)
fig.savefig('./test1_mpc.png')

#
# 
# 
# 
# # mpc options
# mpc_opts = awe.opts.options.Options()
# mpc_opts['mpc']['scheme'] = 'radau'
# mpc_opts['mpc']['d'] = 4
# mpc_opts['mpc']['jit'] = False
# mpc_opts['mpc']['cost_type'] = 'tracking'
# mpc_opts['mpc']['expand'] = True
# mpc_opts['mpc']['linear_solver'] = 'ma57'  # 'mumps'
# mpc_opts['mpc']['max_iter'] = 1000
# mpc_opts['mpc']['max_cpu_time'] = 2000
# mpc_opts['mpc']['N'] = N_mpc
# mpc_opts['mpc']['plot_flag'] = False
# mpc_opts['mpc']['ref_interpolator'] = 'spline'
# mpc_opts['mpc']['u_param'] = 'zoh'
# mpc_opts['mpc']['homotopy_warmstart'] = True
# mpc_opts['mpc']['terminal_point_constr'] = False # necessary False option for 6DOF
# 
# # create MPC object
# mpc = pmpc.Pmpc(mpc_opts['mpc'], t_sam, trial)
# mpc.trial.options['sim']['number_of_finite_elements'] = 10 # integrator steps within one sampling time
# mpc.trial.options['sim']['sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# # ====================================================================
# '''
# Step 3: Create integrator
# '''
# 
# # ____________________________________________________________________
# # build trial and system architecture...
# 
# # Turn ON flag for external forces
# options['model.aero.fictitious_embedding'] = 'substitute'
# 
# # Create integrator options
# int_opts = awe.opts.options.Options()
# int_opts.fill_in_seed(options)
# 
# # Rebuild architecture
# architecture = awe.mdl.architecture.Architecture(int_opts['user_options']['system_model']['architecture'])
# int_opts.build(architecture)
# 
# # Rebuild model
# system_model = awe.mdl.model.Model()
# system_model.build(int_opts['model'], architecture)
# 
# # ____________________________________________________________________
# # build parameter structure...
# 
# # Get scaling, DAE and optional parameters
# scaling = system_model.scaling
# dae = system_model.get_dae()
# theta = system_model.variables_dict['theta'](0.0)
# 
# # get and fill in system design parameters
# p0 = dae.p(0.0)
# theta['diam_t'] = 2e-3 / scaling['theta']['diam_t']
# theta['t_f'] = 1.0
# p0['theta'] = theta.cat
# 
# # get and fill in numerical parameters
# params = system_model.parameters(0.0)
# param_num = int_opts['model']['params']
# for param_type in list(param_num.keys()):
#     if isinstance(param_num[param_type], dict):
#         for param in list(param_num[param_type].keys()):
#             if isinstance(param_num[param_type][param], dict):
#                 for subparam in list(param_num[param_type][param].keys()):
#                     params['theta0', param_type, param, subparam] = param_num[param_type][param][subparam]
#             else:
#                 params['theta0', param_type, param] = param_num[param_type][param]
#     else:
#         params['theta0', param_type] = param_num[param_type]
# params['phi', 'gamma'] = 1
# p0['param'] = params
# 
# # ____________________________________________________________________
# # make casadi collocation integrator
# int_options_seed = {}
# int_options_seed['tf'] = t_sam/N_sim
# int_options_seed['number_of_finite_elements'] = N_sim
# int_options_seed['collocation_scheme'] = 'radau'
# int_options_seed['interpolation_order'] = 4
# int_options_seed['rootfinder'] = 'fast_newton'
# integrator = ca.integrator('integrator', 'collocation', dae.dae, int_options_seed)
# 
# # Build root finder of DAE
# dae.build_rootfinder()
# 
# # ____________________________________________________________________
# # initialze states, controls and algebraic variables
# plot_dict = trial.visualization.plot_dict
# 
# # initialization
# x0 = system_model.variables_dict['x'](0.0)  # initialize states
# u0 = system_model.variables_dict['u'](0.0)  # initialize controls
# z0 = dae.z(0.0)  # algebraic variables initial guess
# 
# # Scaled initial states
# x0['q10'] = np.array(plot_dict['x']['q10'])[:, -1] / scaling['x']['q10']
# x0['dq10'] = np.array(plot_dict['x']['dq10'])[:, -1] / scaling['x']['dq10']
# x0['coeff10'] = np.array(plot_dict['x']['coeff10'])[:, -1] / scaling['x']['coeff10']
# x0['l_t'] = np.array(plot_dict['x']['l_t'])[0, -1] / scaling['x']['l_t']
# x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0, -1] / scaling['x']['dl_t']
# 
# # # Scaled initial controls
# # u0['f_fict10'] = np.array(plot_dict['u']['f_fict10'])[:, -1] / scaling['u']['f_fict10']
# # u0['dcoeff10'] = np.array(plot_dict['u']['dcoeff10'])[:, -1] / scaling['u']['dcoeff10']
# # u0['ddl_t'] = np.array(plot_dict['u']['ddl_t'])[:, -1] / scaling['u']['ddl_t']
# 
# # Scaled algebraic vars
# z0['z'] = np.array(plot_dict['z']['lambda10'])[:, -1] / scaling['z']['lambda10']
# 
# # ====================================================================
# '''
# Step 4: Run simulations
# '''
# 
# #____________________________________________________________________
# 
# #____________________________________________________________________
# # closed loop simulation with MPC and external forces
# 
# # Retrieve trial information
# scaling = trial.model.scaling
# 
# # Initialize simulations
# vars0 = trial.model.variables(0.0)
# vars0['theta'] = trial.model.variables_dict['theta'](0.0)
# 
# # Run simulation
# xsim = [x0.cat.full().squeeze()] # init[0] = x0
# usim = [] #[u0.cat.full().squeeze()]
# fsim = []
# stats = []
# 
# # Loop control evaluations
# for k in range(N_step):
# 
#     # Evaluate controls
#     print("Evaluate controls at t = ", "{:.1f}".format(k*t_sam))
#     u0_call = mpc.step(x0, mpc.trial.options['mpc']['plot_flag'])
#     stats.append(mpc.solver.stats())
# 
#     # fill in controls
#     u0['dcoeff10'] = u0_call[3:5] # scaled!
#     u0['ddl_t'] = u0_call[-1] # scaled!
# 
#     # Loop force evaluations
#     for m in range(N_sim):
# 
#         # Force evaluation index
#         n = (k*N_sim) + m
#         print("Evaluate forces at t = ", "{:.2f}".format(n * t_sam/N_sim))
# 
#         # evaluate forces
#         vars0['x'] = x0 # scaled!
#         vars0['u'] = u0 # scaled!
#         z0 = dae.z(dae.rootfinder(z0, x0, p0))
#         vars0['xdot'] = z0['xdot']
#         vars0['z'] = z0['z']
#         outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
#         F_ext = outputs['aerodynamics', 'f_aero_earth1']
# 
#         # fill in forces
#         u0['f_fict10'] = F_ext / scaling['u']['f_fict10'] # external force
#         # u0['m_fict10'] = M_ext / scaling['u']['m_fict10'] # external moment
# 
#         # fill controls into dae parameters
#         p0['u'] = u0
# 
#         # evaluate integrator
#         out = integrator(x0 = x0, p = p0, z0 = z0)
#         z0 = out['zf']
#         x0 = out['xf']
# 
#         # Simulation outputs
#         xsim.append(out['xf'].full().squeeze())
#         usim.append([u0.cat.full().squeeze()][0])
#         fsim.append(F_ext.full().squeeze())
# 
# #____________________________________________________________________
# '''
# Visualizations
# '''
# 
# # Plot 1: States, controls and forces
# fig = plt.figure(figsize=(6., 5.))
# ax = []
# ax.append(fig.add_axes([0.15, 0.7, 0.8, 0.25]))
# ax.append(fig.add_axes([0.15, 0.4, 0.8, 0.25]))
# ax.append(fig.add_axes([0.15, 0.1, 0.8, 0.25]))
# 
# # Plot state q[0]
# ax[0].plot(trial.visualization.plot_dict['time_grids']['ip'], trial.visualization.plot_dict['x']['q10'][0], 'k--')
# ax[0].plot(closed_loop_sim.visualization.plot_dict['time_grids']['ip'], closed_loop_sim.visualization.plot_dict['x']['q10'][0], 'r-')
# ax[0].plot((t_sam/N_sim)*np.arange(n+2), scaling['x']['q']*np.array([x[0] for x in xsim]), 'g-')
# 
# # Plot control u[-1]
# ax[1].step(trial.visualization.plot_dict['time_grids']['ip'], trial.visualization.plot_dict['u']['ddl_t'][0], 'k--', where='post')
# ax[1].step(closed_loop_sim.visualization.plot_dict['time_grids']['ip'], closed_loop_sim.visualization.plot_dict['u']['ddl_t'][0], 'r-', where='post')
# ax[1].step((t_sam/N_sim)*np.arange(n+1), float(scaling['u']['ddl_t'])*np.array([u[-1] for u in usim]), 'g-', where='post')
# 
# # Plot force f[0]
# ax[2].plot(trial.visualization.plot_dict['time_grids']['ip'], trial.visualization.plot_dict['outputs']['aerodynamics']['f_aero_earth1'][0], 'k--')
# ax[2].plot(closed_loop_sim.visualization.plot_dict['time_grids']['ip'], closed_loop_sim.visualization.plot_dict['outputs']['aerodynamics']['f_aero_earth1'][0], 'r-')
# ax[2].plot((t_sam/N_sim)*np.arange(n+1), np.array([f[0] for f in fsim]), 'g-')
# 
# # Layout
# for k in range(3):
#     ax[k].set_xlim([0.0, (n+1)*(t_sam/N_sim)])
# ax[0].set_ylabel("x", fontsize=12)
# ax[1].set_ylabel("u", fontsize=12)
# ax[2].set_ylabel("F", fontsize=12)
# ax[2].set_xlabel("t", fontsize=12)
# 
# # Plot 2: MPC stats
# fig = plt.figure(figsize=(6., 5.))
# ax1 = fig.add_axes([0.12, 0.12, 0.75, 0.75])
# ax2 = ax1.twinx()
# 
# # MPC stats
# eval =  np.arange(1, len(stats) + 1)
# status =  np.array([s['return_status'] for s in stats])
# walltime =  np.array([s['t_wall_total'] for s in stats])
# iterations =  np.array([s['iter_count'] for s in stats])
# 
# # Create masks
# mask1 = status == 'Solve_Succeeded'
# mask2 = status == 'Solved_To_Acceptable_Level'
# mask3 = status == 'Maximum_Iterations_Exceeded'
# mask4 = status == 'Infeasible_Problem_Detected'
# mask5 = status == 'Maximum_CpuTime_Exceeded'
# mask_all = np.array([True] * eval.max())
# mask_list = [mask1, mask2, mask3, mask4, mask5]
# mask_name = ['Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Maximum_Iterations_Exceeded',
#              'Infeasible_Problem_Detected', 'Maximum_CpuTime_Exceeded']
# mask_clr = ['tab:green', 'tab:blue', 'tab:purple', 'tab:red', 'tab:orange']
# # mask_list = [mask1, mask3]
# # mask_name = ['Solve_Succeeded', 'Maximum_Iterations_Exceeded']
# # mask_clr = ['tab:green', 'tab:purple']
# 
# # Plot
# for mask, clr, name in zip(mask_list, mask_clr, mask_name):
#     ax1.bar(eval[mask], iterations[mask], color=clr, label=name)
# ax2.plot(eval, walltime, '-k')  # , markeredgecolor='k', markerfacecolor=clr, label=name)
# 
# # Layout
# ax1.set_title('Performance of MPC evaluations')
# ax1.set_xlabel('Evaluations')
# ax1.set_ylabel('Iterations')
# ax2.set_ylabel('Walltime [s]')
# ax1.set_xlim([1, eval.max()])
# ax1.legend(loc=2)
# ax1.set_ylim([0,1000])
# ax2.set_ylim([0,50])
# plt.show()
