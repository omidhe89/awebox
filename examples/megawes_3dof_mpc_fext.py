#!/usr/bin/python3
"""
MPC-based closed loop simulation example using external forces for a single 3DOF kite system (based on MegAWES aircraft).
"""

#____________________________________________________________________
# module imports...

# awebox
import awebox as awe

# casadi
import casadi as ca

# python packages
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import time
# ____________________________________________________________________
# User parameters ...
trial_name = 'megawes_3dof_trajectory' # Name of optimization trial
N_loops = 1 # Number of loops per power cycle
N_nlp = 40 # Number of NLP discretization intervals
ts = 0.1 # MPC sample time (simulation window of each MPC evaluation)
N_sim = 100 # Number of MPC evaluations

# ____________________________________________________________________
# trajectory options...

# single MegAWES with point-mass model
options = {}
options['user_options.system_model.architecture'] = {1: 0}
options['user_options.kite_standard'] = awe.megawes_data.data_dict()
options['user_options.system_model.kite_dof'] = 3
options['model.tether.control_var'] = 'ddl_t'

# trajectory should be a single pumping cycle
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
options['user_options.trajectory.lift_mode.windings'] = N_loops
options['model.system_bounds.theta.t_f'] = [1.0, 20.0]

# wind model
options['params.wind.z_ref'] = 10.0
options['user_options.wind.model'] = 'log_wind'
options['user_options.wind.u_ref'] = 5.

# ____________________________________________________________________
# discretization options...

# NLP discretization
options['nlp.n_k'] = N_nlp
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['solver.linear_solver'] = 'ma57'
options['solver.mu_hippo'] = 1e-2

# ____________________________________________________________________
# initialize, build and optimize trial
trial = awe.Trial(options, trial_name)
trial.build()
trial.optimize()

# ____________________________________________________________________
# Built MPC object with default options from trial
mpc = awe.pmpc.Pmpc(trial.options['mpc'], ts, trial)

# ____________________________________________________________________
# Re-build model with modified options (DAE needed for integrator)

# Turn ON flag for external forces and create integrator options
options['model.aero.fictitious_embedding'] = 'substitute'
mod_opts = awe.opts.options.Options()
mod_opts.fill_in_seed(options)

# Re-build architecture
architecture = awe.mdl.architecture.Architecture(mod_opts['user_options']['system_model']['architecture'])
mod_opts.build(architecture)

# Re-build model
system_model = awe.mdl.model.Model()
system_model.build(mod_opts['model'], architecture)

# Get scaling and DAE
scaling = system_model.scaling
dae = system_model.get_dae()

# build parameter structure...
theta = system_model.variables_dict['theta'](0.0)
p0 = dae.p(0.0)
theta['diam_t'] = trial.optimization.V_opt['theta', 'diam_t']
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

# ____________________________________________________________________
# make casadi collocation integrator from DAE

# Simulation time step
N_dt = trial.options['sim']['number_of_finite_elements']
dt = float(ts/N_dt)

# Create integrator options
int_options = {}
int_options['tf'] = dt
int_options['number_of_finite_elements'] = N_dt
int_options['collocation_scheme'] = 'radau'
int_options['interpolation_order'] = 4
int_options['rootfinder'] = 'fast_newton'
integrator = ca.integrator('integrator', 'collocation', dae.dae, int_options)

# Build root finder of DAE
dae.build_rootfinder()

# ____________________________________________________________________
# initialize states and algebraic variables
plot_dict = trial.visualization.plot_dict

# initialization
x0 = system_model.variables_dict['x'](0.0)  # initialize states
u0 = system_model.variables_dict['u'](0.0)  # initialize controls
z0 = dae.z(0.0)  # algebraic variables initial guess

# Scaled initial states
x0['q10'] = np.array(plot_dict['x']['q10'])[:, -1] / scaling['x']['q10']
x0['dq10'] = np.array(plot_dict['x']['dq10'])[:, -1] / scaling['x']['dq10']
x0['coeff10'] = np.array(plot_dict['x']['coeff10'])[:, -1] / scaling['x']['coeff10']
x0['l_t'] = np.array(plot_dict['x']['l_t'])[0, -1] / scaling['x']['l_t']
x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0, -1] / scaling['x']['dl_t']

# Scaled algebraic vars
z0['z'] = np.array(plot_dict['z']['lambda10'])[:, -1] / scaling['z']['lambda10']

# ====================================================================
'''
Step 4: Run simulations
'''

# Retrieve trial information
scaling = trial.model.scaling

# Initialize simulations
vars0 = trial.model.variables(0.0)
vars0['theta'] = trial.model.variables_dict['theta'](0.0)

# Run simulation
xsim = [x0.cat.full().squeeze()] # init[0] = x0
usim = [] #[u0.cat.full().squeeze()]
fsim = []
stats = []

# Loop control evaluations
for k in range(N_sim):

    # Print message
    if ((k+1) % 100) == 0:
        print('##### MPC evaluation N = '+str(k+1)+'/'+str(N_sim)+' #####')

    # Evaluate controls
    # print("Evaluate controls at t = ", "{:.1f}".format(k*ts))
    u0_call = mpc.step(x0, mpc.trial.options['mpc']['plot_flag'])
    stats.append(mpc.solver.stats())

    # fill in controls
    u0['dcoeff10'] = u0_call[3:5] # scaled!
    u0['ddl_t'] = u0_call[-1] # scaled!

    # Loop force evaluations
    for m in range(N_dt):

        # Force evaluation index
        n = (k*N_dt) + m
        print("Evaluate forces at t = ", "{:.3f}".format(n * ts/N_dt))

        # evaluate forces
        vars0['x'] = x0 # scaled!
        vars0['u'] = u0 # scaled!
        z0 = dae.z(dae.rootfinder(z0, x0, p0))
        vars0['xdot'] = z0['xdot']
        vars0['z'] = z0['z']
        outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
        F_ext = outputs['aerodynamics', 'f_aero_earth1']

        # fill in forces
        u0['f_fict10'] = F_ext / scaling['u']['f_fict10'] # external force
        # u0['m_fict10'] = M_ext / scaling['u']['m_fict10'] # external moment

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

#____________________________________________________________________
'''
Visualizations
'''

# ____________________________________________________________________
# Visualize closed-loop simulation results
figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_isometric_plot.png'
trial.plot('isometric')
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax = fig.get_axes()[0]
ax.plot([x[0] for x in xsim], [x[1] for x in xsim], [x[2] for x in xsim], color='g')
l = ax.get_lines()
n = int((len(l)-1)/2)
l[0].set_color('b')
ax.get_legend().remove()
ax.legend([l[-1], l[0]], ['fext', 'ref'], fontsize=14)
fig.savefig(figname)

# # ____________________________________________________________________
# # Visualize states, controls and forces
# fig = plt.figure(figsize=(6., 5.))
# ax = []
# ax.append(fig.add_axes([0.15, 0.7, 0.8, 0.25]))
# ax.append(fig.add_axes([0.15, 0.4, 0.8, 0.25]))
# ax.append(fig.add_axes([0.15, 0.1, 0.8, 0.25]))
#
# # Reference time
# N_cycles = 3
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
#     ax[k].set_xlim([1*Tp, 3*Tp])
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
# N_cycles = 3
# Tp = float(trial.optimization.V_opt['theta', 't_f'])
# t_ref = np.concatenate([k*Tp + trial.visualization.plot_dict['time_grids']['ip'] for k in range(N_cycles)])
# t_sim = dt*np.arange(N_sim*N_dt+1)
#
# #---------------------#
# # Plot position
# fig = plt.figure(figsize=(10., 4.))
# ax = []
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.25]))
#
# # Add data
# for k in range(3):
#     ax[k].plot(t_ref, np.tile(trial.visualization.plot_dict['x']['q10'][k], N_cycles), 'b-')
#     ax[k].plot(t_sim, scaling['x']['q']*np.array([x[k] for x in xsim]), 'g-')
#
# # Layout
# for k, var in zip(range(3), ['x', 'y', 'z']):
#     ax[k].set_xlim([0, 3*Tp])
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
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.25]))
#
# # Add data
# for k in range(3):
#     ax[k].plot(t_ref, np.tile(trial.visualization.plot_dict['x']['dq10'][k], N_cycles), 'b-')
#     ax[k].plot(t_sim, scaling['x']['q']*np.array([x[3+k] for x in xsim]), 'g-')
#
# # Layout
# for k, var in zip(range(3), ['x', 'y', 'z']):
#     ax[k].set_xlim([0, 3*Tp])
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
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.25]))
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
#     ax[k].set_xlim([0, 3*Tp])
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
# ax.append(fig.add_axes([0.1, 0.7, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.41, 0.85, 0.25]))
# ax.append(fig.add_axes([0.1, 0.12, 0.85, 0.25]))
#
# # Add data
# for k in range(3):
#     ax[k].plot(t_ref, np.tile(trial.visualization.plot_dict['outputs']['aerodynamics']['f_aero_earth1'][k], N_cycles), 'b-')
#     ax[k].plot(t_sim[:-1], np.array([f[k] for f in fsim]), 'g-')
#
# # Layout
# for k, var in zip(range(3), ['x', 'y', 'z']):
#     ax[k].set_xlim([0, 3*Tp])
#     ax[k].set_ylabel(r"$F_"+var+"$", fontsize=12)
# ax[2].set_xlabel(r"$t$", fontsize=12)
# ax[0].legend(['ref path', 'mpc/ext F'], loc=1, fontsize=12)
# figname = 'megawes_3dof_n' + '{:d}'.format(N_loops) + '_fext_forces_plot.png'
# fig.savefig(figname)
#
