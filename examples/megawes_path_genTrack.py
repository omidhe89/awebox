#!/usr/bin/python3
"""
Generation of optimal trajectory using MegAWES aircraft
:authors: Jochem De Schutter, Thomas Haas
:date: 26/10/2023
"""

import awebox as awe
import awebox.tools.integrator_routines as awe_integrators
import awebox.pmpc as pmpc
from megawes_settings_modified import set_megawes_settings, set_path_generation_options
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import os
import copy
import csv
import collections
import pickle
# import logging
# import time
# from py4awe.data_func import csv2dict

# ----------------- case initialization ----------------- #

# case
trial_name="megawes_uniform_1loop"

# user settings
N_nlp = 60

N_sim = 10000
N_mpc = 10
N_dt = 20
mpc_sampling_time = 0.1
time_step = mpc_sampling_time/N_dt
N_max_fail = 20

# ================= PART 1: Generation ================= #

# ----------------- default options ----------------- #

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_settings(options)

# ----------------- user-specific options ----------------- #

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
# options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # TH: This option doesn't work in the simulator
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options['user_options.trajectory.lift_mode.windings'] = 1
options['model.system_bounds.theta.t_f'] = [1., 15.]

# indicate desired environment
options['user_options.wind.model'] = 'uniform'
options['user_options.wind.u_ref'] = 10.
options['params.wind.z_ref'] = 100.
options['params.wind.power_wind.exp_ref'] = 0.1

# indicate numerical nlp details
options['nlp.n_k'] = N_nlp
options['nlp.collocation.u_param'] = 'zoh' 
options['solver.linear_solver'] = 'ma57'

# ----------------- Generation and tracking specific options ----------------- #
options_tracking = copy.deepcopy(options)
options_generation = copy.deepcopy(options)
options_generation = set_path_generation_options(options_generation, gamma=0.8)

# ----------------- build and optimize OCP trial ----------------- #

# get architecture
arch = awe.mdl.architecture.Architecture(options_generation['user_options.system_model.architecture'])

# initialize trial
trial = awe.Trial(options_generation, trial_name)

# build trial
trial.build()

# optimize trial
trial.optimize(options_seed=options_generation)

# ----------------- build mpc object ----------------- #

# adjust trial's options_seed
print("Trajectory generation with ddl = "+"{}".format(options_generation["model.ground_station.ddl_t_max"])+" m/s^2")
print("Trajectory tracking with ddl = "+"{}".format(options_tracking["model.ground_station.ddl_t_max"])+" m/s^2")
print("In 'trial.options_seed', ddl = "+"{}".format(trial.options_seed["model.ground_station.ddl_t_max"])+" m/s^2")
trial.options_seed = options_tracking
print("After modification, in 'trial.options_seed' ddl = "+"{}".format(trial.options_seed["model.ground_station.ddl_t_max"])+" m/s^2")

# create MPC options
mpc_opts = awe.Options()
mpc_opts['mpc']['N'] = N_mpc
mpc_opts['mpc']['terminal_point_constr'] = False
mpc_opts['mpc']['homotopy_warmstart'] = True
mpc_opts['mpc']['max_iter'] = 200
mpc_opts['mpc']['max_cpu_time'] = 5.  # seconds

# create PMPC object
mpc = pmpc.Pmpc(mpc_opts['mpc'], mpc_sampling_time, trial)

# ----------------- build integrator ----------------- #

# specify modified options (Turn ON flag for external forces)
int_options = copy.deepcopy(options_tracking)
int_options['model.aero.fictitious_embedding'] = 'substitute'
modified_options = awe.opts.options.Options()
modified_options.fill_in_seed(int_options)

# re-build architecture
architecture = awe.mdl.architecture.Architecture(modified_options['user_options']['system_model']['architecture'])
modified_options.build(architecture)

# re-build model
system_model = awe.mdl.model.Model()
system_model.build(modified_options['model'], architecture)

# get model scaling (alternative: = trial.model.scaling)
scaling = system_model.scaling

# get model DAE
dae = system_model.get_dae()

# make casadi (collocation) integrators from DAE
dae.build_rootfinder() # Build root finder of DAE

# get optimized parameters (theta.keys() = ['diam_t', 't_f'])
theta = system_model.variables_dict['theta'](0.0)
theta['diam_t'] = trial.optimization.V_opt['theta', 'diam_t'] #Scaled!
theta['t_f'] = 1.0

# get numerical parameters (params.keys() = ['theta0', 'phi'])
params = system_model.parameters(0.0)
param_num = modified_options['model']['params']
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

# build parameter structure from DAE (p0.keys() = ['u', 'theta', 'param'])
p0 = dae.p(0.0)
p0['theta'] = theta.cat
p0['param'] = params

# create integrator
integrator = awe_integrators.rk4root('F', dae.dae, dae.rootfinder, {'tf': float(mpc_sampling_time/N_dt), 'number_of_finite_elements': N_dt})

# create symbolic structures for integrators and aerodynamics model
x0_init = ca.MX.sym('x', 23)
u0_init = ca.MX.sym('u', 10)
z0_init = dae.z(0.0)
p0_init = ca.MX.sym('p', 150)

# ----------------- create integrator function ----------------- #
# F_int: Returns new states for specified old states and parameters

# outputs
outputs_integrator = integrator(x0=x0_init, z0=z0_init, p=p0_init)

# evaluate integrator
z0_out = outputs_integrator['zf']
x0_out = outputs_integrator['xf']
q0_out = outputs_integrator['qf']

# Create function
F_int = ca.Function('F_int', [x0_init, p0_init], [x0_out, z0_out, q0_out], ['x0', 'p0'], ['xf', 'zf', 'qf'])

# ----------------- create external aero function ----------------- #
# F_aero: Returns f_earth and m_body for specified states and controls

# solve for z0 with DAE rootfinder (p0 already created)
z0_rf = dae.z(dae.rootfinder(z0_init, x0_init, p0))

vars0_init = ca.vertcat(
    x0_init,
    z0_rf['xdot'],
    u0_init,
    z0_rf['z'],
    system_model.variables_dict['theta'](0.0)
)

# outputs
outputs = system_model.outputs(system_model.outputs_fun(vars0_init, p0['param']))

# extract forces and moments from outputs
F_ext_evaluated = outputs['aerodynamics', 'f_aero_earth1']
M_ext_evaluated = outputs['aerodynamics', 'm_aero_body1']

# Create function
F_aero = ca.Function('F_aero', [x0_init, u0_init], [F_ext_evaluated, M_ext_evaluated], ['x0', 'u0'], ['F_ext', 'M_ext'])

# ----------------- build time functions for MPC ----------------- #
# F_tgrids: Returns time grid of simulation horizon from collocation grid

# time grid in symbolic form
t0 = ca.SX.sym('t0')

# reference interpolation time grid in symbolic form
t_grid = ca.MX.sym('t_grid', mpc.t_grid_coll.shape[0])
t_grid_x = ca.MX.sym('t_grid_x', mpc.t_grid_x_coll.shape[0])
t_grid_u = ca.MX.sym('t_grid_u', mpc.t_grid_u.shape[0])

# time function
F_tgrids = ca.Function('F_tgrids',[t0], [t0 + mpc.t_grid_coll, t0 + mpc.t_grid_x_coll, t0 + mpc.t_grid_u],
                        ['t0'],['tgrid','tgrid_x','tgrid_u'])

# ----------------- build reference function for MPC ----------------- #
# F_ref: Returns tracked reference on specified time grid

# reference function
ref = mpc.get_reference(t_grid, t_grid_x, t_grid_u)

# reference function
F_ref = ca.Function('F_ref', [t_grid, t_grid_x, t_grid_u], [ref], ['tgrid', 'tgrid_x', 'tgrid_u'],['ref'])

# ----------------- build helper functions for MPC ----------------- #
# helper_functions: Return initial guess and controls

# shift solution
V = mpc.trial.nlp.V
V_init = [V['theta'], V['phi'], V['xi']]
for k in range(N_mpc-1):
    V_init.append(V['x',k+1])
    if mpc_opts['mpc']['u_param'] == 'zoh':
        V_init.append(V['u', k+1])
        V_init.append(V['xdot', k+1])
        V_init.append(V['z', k+1])
    for j in range(mpc_opts['mpc']['d']):
        V_init.append(V['coll_var', k+1, j, 'x'])
        V_init.append(V['coll_var', k+1, j, 'z'])
        if mpc_opts['mpc']['u_param'] == 'poly':
            V_init.append(V['coll_var', k+1, j, 'u'])

# copy final interval
V_init.append(V['x', N_mpc-1])
if mpc_opts['mpc']['u_param'] == 'zoh':
    V_init.append(V['u', N_mpc-1])
    V_init.append(V['xdot', N_mpc-1])
    V_init.append(V['z', N_mpc-1])
for j in range(mpc_opts['mpc']['d']):
    V_init.append(V['coll_var', N_mpc-1, j, 'x'])
    V_init.append(V['coll_var', N_mpc-1, j, 'z'])
    if mpc_opts['mpc']['u_param'] == 'poly':
        V_init.append(V['coll_var', N_mpc-1, j, 'u'])
V_init.append(V['x',N_mpc])

# shifted solution
V_shifted = ca.vertcat(*V_init)

# first control
if mpc_opts['mpc']['u_param'] == 'poly':
    u0_shifted = ca.mtimes(mpc.trial.nlp.Collocation.quad_weights[np.newaxis,:], ca.horzcat(*V['coll_var',0,:,'u']).T).T
elif mpc_opts['mpc']['u_param'] == 'zoh':
    u0_shifted = V['u',0]
u0_shifted = mpc.trial.model.variables_dict['u'](u0_shifted)

# controls
u_si = []
for name in list(mpc.trial.model.variables_dict['u'].keys()):
    u_si.append(u0_shifted[name]*scaling['u'][name])
u_si = ca.vertcat(*u_si)

# helper function
helper_functions = ca.Function('helper_functions',[V], [V_shifted, u_si], ['V'], ['V_shifted', 'u0'])

# ================= PART 2: Tracking ================= #
"""
NMPC tracking of optimal path using MegAWES aircraft
:authors: Jochem De Schutter, Thomas Haas
:date: 26/10/2023
"""

# ----------------- evaluate MPC performance ----------------- #
def visualize_mpc_perf(stats):
    # Visualize MPC stats
    fig = plt.figure(figsize=(10., 5.))
    ax1 = fig.add_axes([0.12, 0.12, 0.75, 0.75])
    ax2 = ax1.twinx()

    # MPC stats
    eval =  np.arange(1, len(stats) + 1)
    status =  np.array([s['return_status'] for s in stats])
    walltime =  np.array([s['t_proc_total'] for s in stats])
    iterations =  np.array([s['iter_count'] for s in stats])

    # Create masks
    mask1 = status == 'Solve_Succeeded'
    mask2 = status == 'Solved_To_Acceptable_Level'
    mask3 = status == 'Maximum_Iterations_Exceeded'
    mask4 = status == 'Infeasible_Problem_Detected'
    mask5 = status == 'Maximum_CpuTime_Exceeded'
    mask_all = np.array([True] * eval.max())
    mask_list = [mask1, mask2, mask3, mask4, mask5]
    mask_name = ['Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Maximum_Iterations_Exceeded',
                 'Infeasible_Problem_Detected', 'Maximum_CpuTime_Exceeded']
    mask_clr = ['tab:green', 'tab:blue', 'tab:purple', 'tab:red', 'tab:orange']
    # mask_list = [mask1, mask3]
    # mask_name = ['Solve_Succeeded', 'Maximum_Iterations_Exceeded']
    # mask_clr = ['tab:green', 'tab:purple']

    # Plot
    for mask, clr, name in zip(mask_list, mask_clr, mask_name):
        ax1.bar(eval[mask], iterations[mask], color=clr, label=name)
    ax2.plot(eval, walltime, '-k')  # , markeredgecolor='k', markerfacecolor=clr, label=name)

    # Layout
    ax1.set_title('Performance of MPC evaluations', fontsize=14)
    ax1.set_xlabel('Evaluations', fontsize=14)
    ax1.set_ylabel('Iterations', fontsize=14)
    ax2.set_ylabel('Walltime [s]', fontsize=14)
    ax1.set_xlim([1, eval.max()])
    ax1.legend(loc=2)
    ax1.set_ylim([0,250])
    ax2.set_ylim([0,6])

    return

# ----------------- load optimization data ----------------- #

# create CasADi symbolic variables and dicts
x0 = system_model.variables_dict['x'](0.0)  # initialize states
u0 = system_model.variables_dict['u'](0.0)  # initialize controls
z0 = dae.z(0.0)  # algebraic variables initial guess
w0 = mpc.w0
vars0 = system_model.variables(0.0)
vars0['theta'] = system_model.variables_dict['theta'](0.0)
bounds = mpc.solver_bounds
#    # save solver bounds
#    for var in list(mpc.solver_bounds.keys()):
#        filename = foldername + var + '_bounds.pckl'
#        with open(filename, 'wb') as handle:
#            pickle.dump(mpc.solver_bounds[var], handle, protocol=pickle.HIGHEST_PROTOCOL)
# # load solver bounds
# bounds = {}
# for var in ['lbw', 'ubw', 'lbg', 'ubg']:
#     filename = foldername + var + '_bounds.pckl'
#     with open(filename, 'rb') as handle:
#         bounds[var] = pickle.load(handle)
#
# # load trial results
# filename = foldername + trial_name+'_results.csv'
# awes = csv2dict(filename)
#
# # load trial parameters
# filename = foldername + trial_name+'_theta.csv'
# with open(filename, 'r') as file:
#     csv_reader = csv.reader(file, delimiter=' ')
#     lines = [line for line in csv_reader]
#     theta = [float(value) for value in lines[1]]
t_f = trial.optimization.V_final['theta','t_f'].full().squeeze()

# ----------------- initialize states/alg. vars ----------------- #

# Scaled initial states
plot_dict = trial.visualization.plot_dict
x0['q10'] = np.array(plot_dict['x']['q10'])[:, -1] / scaling['x']['q10']
x0['dq10'] = np.array(plot_dict['x']['dq10'])[:, -1] / scaling['x']['dq10']
x0['omega10'] = np.array(plot_dict['x']['omega10'])[:, -1] / scaling['x']['omega10']
x0['r10'] = np.array(plot_dict['x']['r10'])[:, -1] / scaling['x']['r10']
x0['delta10'] = np.array(plot_dict['x']['delta10'])[:, -1] / scaling['x']['delta10']
x0['l_t'] = np.array(plot_dict['x']['l_t'])[0, -1] / scaling['x']['l_t']
x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0, -1] / scaling['x']['dl_t']

# Scaled algebraic vars
z0['z'] = np.array(plot_dict['z']['lambda10'])[:, -1] / scaling['z']['lambda10']

# # Scaled initial states
# x0['q10'] = np.array([awes['x_q10_'+str(i)][-1] for i in range(3)]) / scaling['x']['q10']
# x0['dq10'] = np.array([awes['x_dq10_'+str(i)][-1] for i in range(3)]) / scaling['x']['dq10']
# x0['omega10'] = np.array([awes['x_omega10_'+str(i)][-1] for i in range(3)]) / scaling['x']['omega10']
# x0['r10'] = np.array([awes['x_r10_'+str(i)][-1] for i in range(9)]) / scaling['x']['r10']
# x0['delta10'] = np.array([awes['x_delta10_'+str(i)][-1] for i in range(3)]) / scaling['x']['delta10']
# x0['l_t'] = np.array(awes['x_l_t_0'][-1]) / scaling['x']['l_t']
# x0['dl_t'] = np.array(awes['x_dl_t_0'][-1]) / scaling['x']['dl_t']
# 
# # Scaled algebraic vars
# z0['z'] = np.array(awes['z_lambda10_0'][-1]) / scaling['z']['lambda10']

# ----------------- initialize MPC controller ----------------- #

# MPC weights
nx = 23
nu = 10
Q = np.ones((nx, 1))
R = np.ones((nu, 1))
P = np.ones((nx, 1))

# # Load function objects and solver
# F_tgrids = ca.external('F_tgrids', foldername + 'F_tgrids.so')
# F_ref = ca.external('F_ref', foldername + 'F_ref.so')
# F_aero = ca.external('F_aero', foldername + 'F_aero.so')
# F_int = ca.external('F_int', foldername + 'F_int.so')
# helpers = ca.external('helper_functions', foldername + 'helper_functions.so')
# solver = ca.nlpsol('solver', 'ipopt', foldername + 'mpc_solver.so', opts)
#
# # Load evaluation functions g_fun and P_fun
# filename = foldername + "F_gfun.pckl"
# with open(filename, 'rb') as handle:
#     F_gfun = pickle.load(handle)
#
# filename = foldername + "F_pfun.pckl"
# with open(filename, 'rb') as handle:
#     F_pfun = pickle.load(handle)

# ----------------- run simulation ----------------- #

# initialize simulation
xsim = [x0.cat.full().squeeze()]
usim = []
fsim = []
msim = []
stats = []
N_mpc_fail = 0

# Loop through time steps
for k in range(N_sim):

    # current time
    current_time = k * time_step

    # evaluate MPC
    if (k % N_dt) < 1e-6:

        # ----------------- evaluate mpc step ----------------- #

        # initial guess
        if k == 0:
            w0 = w0.cat.full().squeeze().tolist()

        # get reference time
        tgrids = F_tgrids(t0 = current_time)
        for grid in list(tgrids.keys()):
            tgrids[grid] = ca.vertcat(*list(map(lambda x: x % t_f, tgrids[grid].full()))).full().squeeze()

        # get reference
        ref = F_ref(tgrid = tgrids['tgrid'], tgrid_x = tgrids['tgrid_x'])['ref']

        # solve MPC problem
        u_ref = 10.
        sol = mpc.solver(x0=w0, lbx=bounds['lbw'], ubx=bounds['ubw'], lbg=bounds['lbg'], ubg=bounds['ubg'],
                     p=ca.vertcat(x0, ref, u_ref, Q, R, P))

        # MPC stats
        stats.append(mpc.solver.stats())
        if stats[-1]["success"]==False:
            N_mpc_fail += 1
            #error = F_gfun(sol['x'] - F_pfun(ca.vertcat(x0, ref, u_ref, Q, R, P))).full()

        # MPC outputs
        out = helper_functions(V=sol['x'])

        # write shifted initial guess
        V_shifted = out['V_shifted']
        w0 = V_shifted.full().squeeze().tolist()

        # retrieve new controls
        u0_call = out['u0']

        # fill in controls
        u0['ddelta10'] = u0_call[6:9] / scaling['u']['ddelta10'] # scaled!
        u0['ddl_t'] = u0_call[-1] / scaling['u']['ddl_t'] # scaled!

        # message
        print("iteration=" + "{:3d}".format(k + 1) + "/" + str(N_sim) + ", t=" + "{:.2f}".format(current_time) + " > compute MPC step")

    else:
        # message
        print("iteration=" + "{:3d}".format(k + 1) + "/" + str(N_sim) + ", t=" + "{:.2f}".format(current_time))

    # ----------------- evaluate aerodynamics ----------------- #

    # evaluate forces and moments
    aero_out = F_aero(x0=x0, u0=u0)

    # fill in forces and moments
    u0['f_fict10'] = aero_out['F_ext'] / scaling['u']['f_fict10']  # external force in inertial frame
    u0['m_fict10'] = aero_out['M_ext'] / scaling['u']['m_fict10']  # external moment in body-fixed frame

    # fill controls and aerodynamics into dae parameters
    p0['u'] = u0

    # ----------------- evaluate system dynamics ----------------- #

    # evaluate dynamics with integrator
    out = F_int(x0=x0, p0=p0)
    z0 = out['zf']
    x0 = out['xf']
    qf = out['qf']

    # Simulation outputs
    xsim.append(out['xf'].full().squeeze())
    usim.append([u0.cat.full().squeeze()][0])
    fsim.append(aero_out['F_ext'].full().squeeze())
    msim.append(aero_out['M_ext'].full().squeeze())

    if N_mpc_fail == N_max_fail:
        print(str(N_max_fail)+" failed MPC evaluations: Interrupt loop")
        break

# ----------------- visualize results ----------------- #

# plot reference path
fig = plt.figure(figsize=(8., 8.))
ax = fig.add_subplot(1, 1, 1, projection='3d')
N = 20
time = trial.visualization.plot_dict['time_grids']['ip']
for m in range(N):
    k = m*int(len(time)/N)
    ax.plot([0.0, np.array(plot_dict['x']['q10'])[0,k]],
            [0.0, np.array(plot_dict['x']['q10'])[1,k]],
            [0.0, np.array(plot_dict['x']['q10'])[2,k]], color='lightgray', linewidth=0.8)
ax.plot(np.array(plot_dict['x']['q10'])[0,:], np.array(plot_dict['x']['q10'])[1,:], np.array(plot_dict['x']['q10'])[2,:])

# adjust plot
fig = plt.gcf()
ax = fig.get_axes()[0]
l = ax.get_lines()
l[-1].set_color('b')

# add external force simulation
ax.plot([x[0] for x in xsim], [x[1] for x in xsim], [x[2] for x in xsim])
l = ax.get_lines()
l[-1].set_color('r')
ax.legend([l[-1], l[-2]], ['fext', 'ref'], fontsize=14)

# ----------------- MPC performance ----------------- #

N_eval = len(stats)
for i in range(N_eval):
    message = "MPC eval "+str(i+1)+"/"+str(N_eval)+": Success = "+str(stats[i]["success"])\
              + ", status = "+stats[i]["return_status"]+", iterations = "+"{:04d}".format(stats[i]["iter_count"])\
              + ", walltime = "+"{:.3f}".format(stats[i]["t_proc_total"])
    print(message)

[stats[-1]["t_proc_total"] for i in range(N_eval)]

visualize_mpc_perf(stats)
plt.show()
print("All done")
# ----------------- end ----------------- #
