#!/usr/bin/python3
"""
Closed-loop MPC simulation of circular pumping trajectory for the Ampyx AP2 aircraft
using external forces/moments computed according to the aerodynamic model of AWEbox.

Model and constraints as in:

"Performance assessment of a rigid wing Airborne Wind Energy pumping system",
G. Licitra, J. Koenemann, A. Bürger, P. Williams, R. Ruiterkamp, M. Diehl
Energy, Vol.173, pp. 569-585, 2019.

:author: Jochem De Schutter
:edited: Thomas Haas

User settings
N_nlp:  Number of NLP discretization intervals of the reference flight path  
N_mpc:  Number of NLP intervals in the tracking window (MPC horizon)
N_sim:  Number of MPC evaluations in closed-loop simulations
N_dt:   Number of built-in integrator steps within one MPC sampling time
t_s:    Sampling time of the MPC controller
"""

#____________________________________________________________________
# module imports...

# awebox
import awebox as awe
import awebox.tools.integrator_routines as awe_integrators
from ampyx_ap2_settings import set_ampyx_ap2_settings

# casadi
import casadi as ca

# python packages
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import copy
from math import ceil

#____________________________________________________________________
# trajectory optimization...

# indicate desired system architecture
# here: single kite with 6DOF Ampyx AP2 model
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_ampyx_ap2_settings(options)

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = 1
options['model.system_bounds.theta.t_f'] = [5.0, 35.0] # additional constraints limiting path period

# indicate desired environment
# here: wind velocity profile according to power-law
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 10.

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. 
# also, specify a linear solver to perform the Newton-steps within ipopt.
N_nlp = 60
options['nlp.n_k'] = N_nlp
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'

# build and optimize the NLP (trial)
trial = awe.Trial(options, 'Ampyx_AP2')
trial.build()
trial.optimize()
# trial.plot(['isometric']) # options: 'states', 'controls', 'constraints', 'quad'

#____________________________________________________________________
# built-in closed-loop MPC simulation...

# closed-loop MPC simulation options
N_mpc = 10
N_sim = 350
N_dt = 20 
t_s = 0.1

# fill MPC options (default options)
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

# fill simulation options
options['sim.number_of_finite_elements'] = N_dt 
options['sim.sys_params'] = copy.deepcopy(trial.options['solver']['initialization']['sys_params_num'])

# make simulator and run simulation
closed_loop_sim = awe.sim.Simulation(trial, 'closed_loop', t_s, options)
closed_loop_sim.run(N_sim)
# closed_loop_sim.plot(['isometric']) # options: 'states', 'controls', 'constraints', 'quad'

#____________________________________________________________________
# closed-loop MPC simulation with external aero...
# Steps:
# 1. re-build model with modified options (DAE needed for integrator)
# 2. make integrator for external force simulation
# 3. make MPC controller for closed-loop simulation
# 4. retrieve initial states
# 5. run simulation with external aero and MPC controller

# [1] specify modified options (Turn ON flag for external forces)
options['model.aero.fictitious_embedding'] = 'substitute'
mod_opts = awe.opts.options.Options()
mod_opts.fill_in_seed(options)

# re-build architecture
architecture = awe.mdl.architecture.Architecture(mod_opts['user_options']['system_model']['architecture'])
mod_opts.build(architecture)

# re-build model
system_model = awe.mdl.model.Model()
system_model.build(mod_opts['model'], architecture)

# get model scaling and DAE
scaling = system_model.scaling
dae = system_model.get_dae()

# build parameter structure...
theta = system_model.variables_dict['theta'](0.0)
p0 = dae.p(0.0)
theta['diam_t'] = trial.optimization.V_opt['theta', 'diam_t'] #Scaled!
theta['t_f'] = 1.0
p0['theta'] = theta.cat

# get and fill in numerical parameters
params = system_model.parameters(0.0)
param_num = mod_opts['model']['params']
for param_type in list(param_num.keys()):
    if isinstance(param_num[param_type], dict):
        for param in list(param_num[param_type].keys()):
            if isinstance(param_num[param_type][param], dict):
                for subparam in list(param_num[param_type][param].keys()):
                    params['theta0', param_type, param, subparam] = param_num[param_type][param][subparam]
            else:
                params['theta0', param_type, param] = param_num[param_type][param]
    else:
        params['theta0', param_type] = param_num[param_type]
params['phi', 'gamma'] = 1
p0['param'] = params

# [2] make casadi (collocation) integrators from DAE
# 1. Runge-Kutta integrator (works for MegAWES)
dae.build_rootfinder() # Build root finder of DAE
integrator = awe_integrators.rk4root( # Create RK4 integrator
    'F',
    dae.dae,
    dae.rootfinder,
    {'tf': float(t_s/N_dt), 'number_of_finite_elements': N_dt})

# # 2. Collocation integrator (CasADi built-in, works for AP2)
# int_options = {} # integrator options
# int_options['tf'] = float(t_s/N_dt)
# int_options['number_of_finite_elements'] = N_dt
# int_options['collocation_scheme'] = 'radau'
# int_options['interpolation_order'] = 4
# int_options['rootfinder'] = 'fast_newton'
# integrator = ca.integrator('integrator', 'collocation', dae.dae, int_options) # build integrator
# dae.build_rootfinder() # build root finder of DAE

# [3] make MPC object with default options from original trial
mpc_opts = awe.opts.options.Options()
mpc_opts.fill_in_seed(options)
mpc = awe.pmpc.Pmpc(mpc_opts['mpc'], t_s, trial)

# [4] initialize states and algebraic variables
plot_dict = trial.visualization.plot_dict

# initialization
x0 = system_model.variables_dict['x'](0.0)  # initialize states
u0 = system_model.variables_dict['u'](0.0)  # initialize controls
z0 = dae.z(0.0)  # algebraic variables initial guess

# Scaled initial states
x0['q10'] = np.array(plot_dict['x']['q10'])[:, -1] / scaling['x']['q10']
x0['dq10'] = np.array(plot_dict['x']['dq10'])[:, -1] / scaling['x']['dq10']
x0['omega10'] = np.array(plot_dict['x']['omega10'])[:, -1] / scaling['x']['omega10']
x0['r10'] = np.array(plot_dict['x']['r10'])[:, -1] / scaling['x']['r10']
x0['delta10'] = np.array(plot_dict['x']['delta10'])[:, -1] / scaling['x']['delta10']
x0['l_t'] = np.array(plot_dict['x']['l_t'])[0, -1] / scaling['x']['l_t']
x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0, -1] / scaling['x']['dl_t']

# Scaled algebraic vars
z0['z'] = np.array(plot_dict['z']['lambda10'])[:, -1] / scaling['z']['lambda10']

# [5] run simulation with combined MPC/External forces
# Initialize simulations
vars0 = system_model.variables(0.0)
vars0['theta'] = system_model.variables_dict['theta'](0.0)
xsim = [x0.cat.full().squeeze()]
usim = [] #[u0.cat.full().squeeze()]
fsim = []
msim = []
stats = []

# Loop control evaluations
for k in range(N_sim):

    # Print message
    if ((k+1) % 100) == 0:
        print('##### MPC evaluation N = '+str(k+1)+'/'+str(N_sim)+' #####')

    # Evaluate controls
    u0_call = mpc.step(x0, mpc_opts['mpc']['plot_flag'])
    stats.append(mpc.solver.stats())

    # fill in controls
    u0['ddelta10'] = u0_call[6:9] # scaled!
    u0['ddl_t'] = u0_call[-1] # scaled!

    # Loop force evaluations
    for m in range(N_dt):

        # # Force evaluation index
        # n = (k*N_dt) + m
        # print("Evaluate forces at t = ", "{:.3f}".format(n * ts/N_dt))

        # evaluate forces and moments
        vars0['x'] = x0 # scaled!
        vars0['u'] = u0 # scaled!
        z0 = dae.z(dae.rootfinder(z0, x0, p0))
        vars0['xdot'] = z0['xdot']
        vars0['z'] = z0['z']
        outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
        F_ext = outputs['aerodynamics', 'f_aero_earth1']
        M_ext = outputs['aerodynamics', 'm_aero_body1']

        # fill in forces and moments
        u0['f_fict10'] = F_ext / scaling['u']['f_fict10'] # external force
        u0['m_fict10'] = M_ext / scaling['u']['m_fict10'] # external moment

        # fill controls into dae parameters
        p0['u'] = u0

        # evaluate integrator
        out = integrator(x0 = x0, p = p0, z0 = z0)
        z0 = out['zf']
        x0 = out['xf']

        # Simulation outputs
        xsim.append(out['xf'].full().squeeze())
        usim.append([u0.cat.full().squeeze()][0])
        fsim.append(F_ext.full().squeeze())
        msim.append(M_ext.full().squeeze())

# ____________________________________________________________________
# visualize closed-loop simulation results
# plot reference path
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

# add external force simulation
ax.plot([x[0] for x in xsim], [x[1] for x in xsim], [x[2] for x in xsim])
l = ax.get_lines()
l[-1].set_color('g')

# layout/save
ax.get_legend().remove()
ax.legend([l[-1], l[-2], l[0]], ['fext', 'mpc', 'ref'], fontsize=14)
figname = './ampyx_ap2_external_aero_isometric.png'
fig.savefig(figname)

# # ____________________________________________________________________
# # Visualize states, controls and forces
# N_cycles = ceil(T_end/trial.optimization.V_opt['theta', 't_f'])
# fig = plt.figure(figsize=(6., 5.))
# ax = []
# ax.append(fig.add_axes([0.15, 0.7, 0.8, 0.25]))
# ax.append(fig.add_axes([0.15, 0.4, 0.8, 0.25]))
# ax.append(fig.add_axes([0.15, 0.1, 0.8, 0.25]))
#
# # Reference time
# Tp = float(trial.optimization.V_opt['theta', 't_f'])
# t_ref = np.concatenate([k*Tp + trial.visualization.plot_dict['time_grids']['ip'] for k in range(N_cycles)])
# t_sim = dt*np.arange(N_sim*N_dt+1)
#
# # Plot state q[0]
# ax[0].plot(t_ref, np.tile(trial.visualization.plot_dict['x']['q10'][0], N_cycles), 'b-')
# ax[0].plot(t_sim, scaling['x']['q']*np.array([x[0] for x in xsim]), 'g-')
#
# # Plot control u[-1]
# ax[1].step(t_ref, np.tile(trial.visualization.plot_dict['u']['ddl_t'][0], N_cycles), 'b-', where='post')
# ax[1].step(t_sim[:-1], float(scaling['u']['ddl_t'])*np.array([u[-1] for u in usim]), 'g-', where='post')
#
# # Plot force f[0]
# ax[2].plot(t_ref, np.tile(trial.visualization.plot_dict['outputs']['aerodynamics']['f_aero_earth1'][0], N_cycles), 'b-')
# ax[2].plot(t_sim[:-1], np.array([f[0] for f in fsim]), 'g-')
#
# # Layout
# for k in range(3):
#     ax[k].set_xlim([0, T_end])
# ax[0].set_ylabel(r"$q_x$", fontsize=14)
# ax[1].set_ylabel(r"$\ddot{\ell}_T$", fontsize=14)
# ax[2].set_ylabel(r"$F_x$", fontsize=14)
# ax[2].set_xlabel(r"$t$", fontsize=14)
# ax[0].legend(['ref path', 'mpc/ext F'], loc=2, fontsize=14)
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_tracking_plot.png'
# fig.savefig(figname)
#
# # ____________________________________________________________________
# # Visualize MPC stats
# fig = plt.figure(figsize=(10., 5.))
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
# ax1.set_title('Performance of MPC evaluations', fontsize=14)
# ax1.set_xlabel('Evaluations', fontsize=14)
# ax1.set_ylabel('Iterations', fontsize=14)
# ax2.set_ylabel('Walltime [s]', fontsize=14)
# ax1.set_xlim([1, eval.max()])
# ax1.legend(loc=2)
# ax1.set_ylim([0,50])
# ax2.set_ylim([0,1])
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_perf_plot.png'
# fig.savefig(figname)
#
# # ____________________________________________________________________
# # Visualize all components of position and speed, controls and forces
#
# # Reference time
# Tp = float(trial.optimization.V_opt['theta', 't_f'])
# t_ref = np.concatenate([k*Tp + trial.visualization.plot_dict['time_grids']['ip'] for k in range(N_cycles)])
# t_sim = dt*np.arange(N_sim*N_dt+1)
#
# #---------------------#
# # Plot position
# fig = plt.figure(figsize=(10., 4.))
# ax = []
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.22]))
#
# # Add data
# for k in range(3):
#     ax[k].plot(t_ref, np.tile(trial.visualization.plot_dict['x']['q10'][k], N_cycles), 'b-')
#     ax[k].plot(t_sim, scaling['x']['q']*np.array([x[k] for x in xsim]), 'g-')
#
# # Layout
# for k, var in zip(range(3), ['x', 'y', 'z']):
#     ax[k].set_xlim([0, T_end])
#     ax[k].set_ylabel(r"$q_"+var+"$", fontsize=12)
# ax[2].set_xlabel(r"$t$", fontsize=12)
# ax[0].legend(['ref path', 'mpc/ext F'], loc=1, fontsize=12)
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_position_plot.png'
# fig.savefig(figname)
#
# #---------------------#
# # Plot speed
# fig = plt.figure(figsize=(10., 4.))
# ax = []
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.22]))
#
# # Add data
# for k in range(3):
#     ax[k].plot(t_ref, np.tile(trial.visualization.plot_dict['x']['dq10'][k], N_cycles), 'b-')
#     ax[k].plot(t_sim, scaling['x']['q']*np.array([x[3+k] for x in xsim]), 'g-')
#
# # Layout
# for k, var in zip(range(3), ['x', 'y', 'z']):
#     ax[k].set_xlim([0, T_end])
#     ax[k].set_ylabel(r"$\dot{q}_"+var+"$", fontsize=12)
# ax[2].set_xlabel(r"$t$", fontsize=12)
# ax[0].legend(['ref path', 'mpc/ext F'], loc=1, fontsize=12)
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_speed_plot.png'
# fig.savefig(figname)
#
# #---------------------#
# # Plot controls
# fig = plt.figure(figsize=(10., 4.))
# ax = []
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.22]))
#
# # Add data
# for k in range(2):
#     ax[k].step(t_ref, np.tile(trial.visualization.plot_dict['u']['dcoeff10'][k], N_cycles), 'b-', where='post')
#     ax[k].step(t_sim[:-1], float(scaling['u']['dcoeff10'])*np.array([u[3+k] for u in usim]), 'g-')
# ax[2].step(t_ref, np.tile(trial.visualization.plot_dict['u']['ddl_t'][0], N_cycles), 'b-', where='post')
# ax[2].step(t_sim[:-1], float(scaling['u']['ddl_t'])*np.array([u[-1] for u in usim]), 'g-', where='post')
#
# # Layout
# for k in  range(3):
#     ax[k].set_xlim([0, T_end])
# ax[0].set_ylabel(r"$\dot{C_L}$", fontsize=12)
# ax[1].set_ylabel(r"$\dot{\psi}$", fontsize=12)
# ax[2].set_ylabel(r"$\ddot{\ell}_T$", fontsize=12)
# ax[2].set_xlabel(r"$t$", fontsize=12)
# ax[0].legend(['ref path', 'mpc/ext F'], loc=1, fontsize=12)
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_controls_plot.png'
# fig.savefig(figname)
#
# #---------------------#
# # Plot forces
# fig = plt.figure(figsize=(10., 4.))
# ax = []
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.22]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.22]))
#
# # Add data
# for k in range(3):
#     ax[k].plot(t_ref, np.tile(trial.visualization.plot_dict['outputs']['aerodynamics']['f_aero_earth1'][k], N_cycles), 'b-')
#     ax[k].plot(t_sim[:-1], np.array([f[k] for f in fsim]), 'g-')
#
# # Layout
# for k, var in zip(range(3), ['x', 'y', 'z']):
#     ax[k].set_xlim([0, T_end])
#     ax[k].set_ylabel(r"$F_"+var+"$", fontsize=12)
# ax[2].set_xlabel(r"$t$", fontsize=12)
# ax[0].legend(['ref path', 'mpc/ext F'], loc=1, fontsize=12)
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_forces_plot.png'
# fig.savefig(figname)
#
