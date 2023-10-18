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
import awebox.tools.integrator_routines as awe_integrators
import awebox.pmpc as pmpc
import awebox.sim as sim
from megawes_settings_modified import set_megawes_settings, set_path_generation_options, set_path_tracking_trial_options, solve_mpc_step
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import logging
import os
import time
import copy
import csv
import collections


"""
===========================
= Part 1. Path generation =
===========================
"""

# ----------------- initialization ----------------- #

# case
# TH: Specify output folder
foldername="/cfdfile2/data/fm/thomash/Devs/PycharmProjects/awebox/awebox_compiled_solvers/output_files/"
# foldername="/cfdfile2/data/fm/thomash/Tools/awebox/experimental/output_files/"
trial_name="megawes_uniform_1loop"
if not os.path.isdir(foldername):
    os.system('mkdir '+foldername)

# user settings
N_nlp = 40
N_sim = 1000
N_mpc = 10
N_dt = 20
mpc_sampling_time = 0.1
time_step = mpc_sampling_time/N_dt

# compiler flags
# TH: Turn on flags to True
autosave = False
compile_mpc_solver = False
compile_x0_function = False
compile_tgrids_function = False
compile_ref_function = False
compile_feedback_function = False
compile_simulator = False
compile_integrator = False # Keep as False, not yet implemented

# ----------------- default options ----------------- #

# indicate desired system architecture
options = {}
options['user_options.system_model.architecture'] = {1:0}
options = set_megawes_settings(options)

# ----------------- user-specific options ----------------- #

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
# options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' 
# TH: This option doesn't work in the simulator
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

# ----------------- adapt options ----------------- #

# case specific options dicts
mpc_options = copy.deepcopy(options)
ocp_options = copy.deepcopy(options)
int_options = copy.deepcopy(options)

# limit bounds/constraints for path generation
gamma = 1.0
ocp_options = set_path_generation_options(ocp_options, gamma = gamma)

# ----------------- build and optimize OCP trial ----------------- #

# get architecture
arch = awe.mdl.architecture.Architecture(ocp_options['user_options.system_model.architecture'])

# initialize trial
trial = awe.Trial(ocp_options, trial_name)

# build trial
trial.build()

# ----------------- optimize trial ----------------- #
trial.optimize(options_seed=ocp_options)

# ----------------- optimization results ----------------- #
if autosave:

   # path outputs
   filename = foldername + trial_name+'_results'
   trial.write_to_csv(filename)

   # path parameters
   filename = foldername + trial_name+'_theta.csv'
   with open(filename, 'w') as parameters:
       theta = {}
       for name in list(trial.model.variables_dict['theta'].keys()):
           if name != 't_f':
               theta[name] = trial.optimization.V_final['theta',name]
           else:
               theta[name] =  trial.visualization.plot_dict['time_grids']['ip'][-1]
       pcdw = csv.DictWriter(parameters, delimiter=' ', fieldnames=theta)
       pcdw.writeheader()
       ordered_dict = collections.OrderedDict(sorted(list(theta.items())))
       pcdw.writerow(ordered_dict)

   # plots
   list_of_plots = ['isometric'] #['isometric', 'states', 'controls', 'constraints']
   trial.plot(list_of_plots)
   for i, plot_name in enumerate(list_of_plots, start=1):
       plt.figure(i)
       filename = foldername + trial_name + '_plot_' + plot_name + '.png'
       plt.savefig(filename)

# ----------------- build mpc object ----------------- #

# create MPC options
mpc_opts = awe.Options()
mpc_opts['mpc']['N'] = N_mpc
mpc_opts['mpc']['terminal_point_constr'] = False
mpc_opts['mpc']['homotopy_warmstart'] = True

# overwrite trial options to use MPC options:
trial = set_path_tracking_trial_options(trial, gamma = gamma)

# create PMPC object
mpc = pmpc.Pmpc(mpc_opts['mpc'], mpc_sampling_time, trial)

# ----------------- build integrator ----------------- #

# specify modified options (Turn ON flag for external forces)
int_options = copy.deepcopy(options)
int_options['model.aero.fictitious_embedding'] = 'substitute'
modified_options = awe.opts.options.Options()
modified_options.fill_in_seed(int_options)

# re-build architecture
architecture = awe.mdl.architecture.Architecture(modified_options['user_options']['system_model']['architecture'])
modified_options.build(architecture)

# re-build model
system_model = awe.mdl.model.Model()
system_model.build(modified_options['model'], architecture)

# get model scaling (alternative:  = trial.model.scaling)
scaling = system_model.scaling

# get model DAE
dae = system_model.get_dae()

# make casadi (collocation) integrators from DAE
dae.build_rootfinder() # Build root finder of DAE
integrator = awe_integrators.rk4root('F', dae.dae, dae.rootfinder, {'tf': float(mpc_sampling_time/N_dt), 'number_of_finite_elements': N_dt})

# ----------------- build parameter structure ----------------- #
''''
out = integrator(x0 = x0, z0 = z0, p = p0).
Integrator call requires x0, z0, p0. Create entries 'theta' and 'param' of p0.
(The entry 'u' of p0 is created along x0, z0 during the simulation.)
'''

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

# ----------------- create CasADi structures for simulation ----------------- #

# create structures and set to zero
x0 = system_model.variables_dict['x'](0.0)  # initialize states
u0 = system_model.variables_dict['u'](0.0)  # initialize controls
z0 = dae.z(0.0)  # algebraic variables initial guess
vars0 = system_model.variables(0.0)
vars0['theta'] = system_model.variables_dict['theta'](0.0)

# ----------------- TODO: create external aero function ----------------- #

# # evaluate forces and moments
# z0 = dae.z(dae.rootfinder(z0, x0, p0))
#
# vars0['x'] = x0  # scaled!
# vars0['u'] = u0  # scaled!
# vars0['xdot'] = z0['xdot']  # scaled!
# vars0['z'] = z0['z']  # scaled!
#
# outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
# F_ext = outputs['aerodynamics', 'f_aero_earth1']
# M_ext = outputs['aerodynamics', 'm_aero_body1']
#
# F_aero = ca.Function('F_aero', [x0, u0], [F_ext, M_ext], ['x0', 'u0'], ['F_ext', 'M_ext'])

# # ----------------- TODO: create integrator function ----------------- #
#
# out = integrator(x0=x0, z0=z0, p=p0)
# xf = out['xf']
# zf = out['zf']
# qf = out['qf']
# F_int = ca.Function('F_int', [x0, z0, p0], [xf, zf, qf], ['x0', 'z0', 'p'], ['xf', 'zf', 'qf'])

# ----------------- initialize simulation ----------------- #

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

# ----------------- run simulation with build-in features ----------------- #

# initialize simulation
xsim = [x0.cat.full().squeeze()]
usim = []  # [u0.cat.full().squeeze()]
fsim = []
msim = []
stats = []

# Loop time step (TH: different from control intervals)
for k in range(N_sim+1):

    # current time
    current_time = k * time_step
    # print("iteration=" + str(k) + "/" + str(N_sim) + ", t=" + "{:.3f}".format(current_time))
    # print(k, current_time, math.fmod(current_time, mpc_sampling_time))

    # if k == 30:

    if (k % N_dt) < 1e-6:
    # if (current_time+1.0e-12 % mpc_sampling_time) < 1e-6:

        # print message
        print("iteration=" + str(k) + "/" + str(N_sim) + ", t=" + "{:.3f}".format(current_time) + " > compute MPC step")

        # evaluate controls
        u0_call = mpc.step(x0, mpc_opts['mpc']['plot_flag'])
        stats.append(mpc.solver.stats())

        # fill in controls
        u0['ddelta10'] = u0_call[6:9]  # scaled!
        u0['ddl_t'] = u0_call[-1]  # scaled!

    elif (current_time % mpc_sampling_time) > 1e-6:

        # print evaluation
        print("iteration=" + str(k) + "/" + str(N_sim) + ", t=" + "{:.3f}".format(current_time))

    # evaluate forces and moments
    z0 = dae.z(dae.rootfinder(z0, x0, p0))

    vars0['x'] = x0             # scaled!
    vars0['u'] = u0             # scaled!
    vars0['xdot'] = z0['xdot']  # scaled!
    vars0['z'] = z0['z']        # scaled!

    outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
    F_ext = outputs['aerodynamics', 'f_aero_earth1']
    M_ext = outputs['aerodynamics', 'm_aero_body1']

    # # TODO: evaluate force and moments with Function objects
    # out = F_aero(x0=x0, u0=u0)
    # F_ext = out['F_ext']
    # M_ext = out['M_ext']

    # fill in forces and moments
    u0['f_fict10'] = F_ext / scaling['u']['f_fict10']  # external force
    u0['m_fict10'] = M_ext / scaling['u']['m_fict10']  # external moment

    # fill controls into dae parameters
    p0['u'] = u0

    # evaluate integrator
    out = integrator(x0=x0, z0=z0, p=p0)
    z0 = out['zf']
    x0 = out['xf']
    qf = out['qf']

    # # TODO: evaluate integrator with Function objects
    # out = F_int(x0=x0, z0=z0, p=p0)
    # z0 = out['zf']
    # x0 = out['xf']
    # qf = out['qf']

    # Simulation outputs
    xsim.append(out['xf'].full().squeeze())
    usim.append([u0.cat.full().squeeze()][0])
    fsim.append(F_ext.full().squeeze())
    msim.append(M_ext.full().squeeze())

# ----------------- visualize results ----------------- #

# plot reference path
trial.plot(['isometric'])
fig = plt.gcf()
fig.set_size_inches(10, 8)
ax = fig.get_axes()[0]
l = ax.get_lines()
l[0].set_color('b')

# add external force simulation
ax.plot([x[0] for x in xsim], [x[1] for x in xsim], [x[2] for x in xsim])
l = ax.get_lines()
l[-1].set_color('r')
ax.get_legend().remove()
ax.legend([l[-1], l[0]], ['fext', 'ref'], fontsize=14)
plt.show()

# =========================================================== #

# # ----------------- build functions for MPC ----------------- #
#
# # solver bounds
# solver_bounds = mpc.solver_bounds
#
# # scaled states
# x0_si = trial.model.variables_dict['x']
# x0_scaled = []
# for name in list(trial.model.variables_dict['x'].keys()):
#     x0_scaled.append(x0_si[name]/scaling['x'][name])
# x0_scaled = ca.vertcat(*x0_scaled)
#
# # time grids
# t0 = ca.SX.sym('t0')
#
# # reference interpolation
# t_grid = ca.MX.sym('t_grid', mpc.t_grid_coll.shape[0])
# t_grid_x = ca.MX.sym('t_grid_x', mpc.t_grid_x_coll.shape[0])
# t_grid_u = ca.MX.sym('t_grid_u', mpc.t_grid_u.shape[0])
#
# # reference function
# ref = mpc.get_reference(t_grid, t_grid_x, t_grid_u)
#
# # shift solution
# V = mpc.trial.nlp.V
# V_init = [V['theta'], V['phi'], V['xi']]
# for k in range(N_mpc-1):
#     V_init.append(V['x',k+1])
#     if mpc_opts['mpc']['u_param'] == 'zoh':
#         V_init.append(V['u', k+1])
#         V_init.append(V['xdot', k+1])
#         V_init.append(V['z', k+1])
#     for j in range(mpc_opts['mpc']['d']):
#         V_init.append(V['coll_var', k+1, j, 'x'])
#         V_init.append(V['coll_var', k+1, j, 'z'])
#         if mpc_opts['mpc']['u_param'] == 'poly':
#             V_init.append(V['coll_var', k+1, j, 'u'])
#
# # copy final interval
# V_init.append(V['x', N_mpc-1])
# if mpc_opts['mpc']['u_param'] == 'zoh':
#     V_init.append(V['u', N_mpc-1])
#     V_init.append(V['xdot', N_mpc-1])
#     V_init.append(V['z', N_mpc-1])
# for j in range(mpc_opts['mpc']['d']):
#     V_init.append(V['coll_var', N_mpc-1, j, 'x'])
#     V_init.append(V['coll_var', N_mpc-1, j, 'z'])
#     if mpc_opts['mpc']['u_param'] == 'poly':
#         V_init.append(V['coll_var', N_mpc-1, j, 'u'])
# V_init.append(V['x',N_mpc])
#
# # shifted solution
# V_shifted = ca.vertcat(*V_init)
#
# # first control
# if mpc_opts['mpc']['u_param'] == 'poly':
#     u0 = ca.mtimes(mpc.trial.nlp.Collocation.quad_weights[np.newaxis,:], ca.horzcat(*V['coll_var',0,:,'u']).T).T
# elif mpc_opts['mpc']['u_param'] == 'zoh':
#     u0 = V['u',0]
# u0 = mpc.trial.model.variables_dict['u'](u0)
#
# # controls
# u_si = []
# for name in list(mpc.trial.model.variables_dict['u'].keys()):
#     u_si.append(u0[name]*scaling['u'][name])
# u_si = ca.vertcat(*u_si)
#
# # ----------------- build functions for integrator ----------------- #
#
# # scale initial state
# scaling = trial.model.scaling
# x0_si = trial.model.variables_dict['x'](ca.MX.sym('x0_si', trial.model.variables_dict['x'].shape[0]))
# x0_scaled = []
# for name in list(mpc.trial.model.variables_dict['x'].keys()):
#     x0_scaled.append(x0_si[name] / scaling['x'][name])
# x0_scaled = ca.vertcat(*x0_scaled)
#
# # scale initial control
# u0_si = trial.model.variables_dict['u'](ca.MX.sym('u0_si', trial.model.variables_dict['u'].shape[0]))
# u_scaled = []
# for name in list(mpc.trial.model.variables_dict['u'].keys()):
#     u_scaled.append(u0_si[name] / scaling['u'][name])
# u_scaled = ca.vertcat(*u_scaled)
#
# z0_scaled = []
# for name in list(mpc.trial.model.variables_dict['x'].keys()):
#     z0_scaled.append(x0_si[name] / scaling['z'][name])
# x0_scaled = ca.vertcat(*x0_scaled)
#
# z0_scaled = ...
#
# # un-scale simulated state
# xf_scaled = trial.model.variables_dict['x'](integrator.F(x0=x0_scaled, z0=z0_scaled, p=p0)['xf']) # Replace simulator by integrator
# xf_si = []
# for name in list(mpc.trial.model.variables_dict['x'].keys()):
#     xf_si.append(xf_scaled[name] * scaling['x'][name])
# xf_si = ca.vertcat(*xf_si)
#
# qf = trial.model.variables_dict['x'](integrator.F(x0=x0_scaled, z0=z0_scaled, p=p0)['qf']) # Replace simulator by integrator
#
# # retrieve algebraic variables and states derivatives from DAE
# dae = trial.model.get_dae()
# zf_scaled = dae.z(integrator.F(x0=x0_scaled, z0=z0_scaled, p=p0)['zf']) # Here the same
#
# # un-scale simulated algebraic variable
# lambda_scaled = trial.model.variables_dict['z'](zf_scaled['z'])
# zf_si = []
# for name in list(mpc.trial.model.variables_dict['z'].keys()):
#     zf_si.append(lambda_scaled[name] * scaling['z'][name])
# zf_si = ca.vertcat(*zf_si)
# #
# # # un-scale simulated states derivatives
# # xdot_scaled = trial.model.variables_dict['xdot'](zf_scaled['xdot'])
# # xdot_si = []
# # for name in list(mpc.trial.model.variables_dict['xdot'].keys()):
# #     xdot_si.append(xdot_scaled[name] * scaling['xdot'][name])
# # xdot_si = ca.vertcat(*xdot_si)
#
# # evaluate forces and moments
# x0 = system_model.variables_dict['x'](0.0)  # initialize states
# u0 = system_model.variables_dict['u'](0.0)  # initialize controls
# z0 = dae.z(0.0)  # algebraic variables initial guess
#
# # scaled initial states
# plot_dict = trial.visualization.plot_dict
# x0['q10'] = np.array(trial.visualization.plot_dict['x']['q10'])[:, -1] / scaling['x']['q10']
# x0['dq10'] = np.array(trial.visualization.plot_dict['x']['dq10'])[:, -1] / scaling['x']['dq10']
# x0['omega10'] = np.array(trial.visualization.plot_dict['x']['omega10'])[:, -1] / scaling['x']['omega10']
# x0['r10'] = np.array(trial.visualization.plot_dict['x']['r10'])[:, -1] / scaling['x']['r10']
# x0['delta10'] = np.array(trial.visualization.plot_dict['x']['delta10'])[:, -1] / scaling['x']['delta10']
# x0['l_t'] = np.array(trial.visualization.plot_dict['x']['l_t'])[0, -1] / scaling['x']['l_t']
# x0['dl_t'] = np.array(trial.visualization.plot_dict['x']['dl_t'])[0, -1] / scaling['x']['dl_t']
#
# # scaled algebraic vars
# z0['z'] = np.array(trial.visualization.plot_dict['z']['lambda10'])[:, -1] / scaling['z']['lambda10']
#
# # get parameters
# vars0 = system_model.variables(0.0) # Also pickle / pickle everything
# vars0['theta'] = system_model.variables_dict['theta'](0.0)
# vars0['x'] = x0  # scaled!
# vars0['u'] = u0  # scaled!
# z0 = dae.z(dae.rootfinder(z0, x0, p0))
# vars0['xdot'] = z0['xdot']
# vars0['z'] = z0['z']
# outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
# F_ext = outputs['aerodynamics', 'f_aero_earth1']
# M_ext = outputs['aerodynamics', 'm_aero_body1']
#
#
# # ----------------- build code-generated CasADi functions ----------------- #
# # Function objects: Function(name, input expression, output expression, input name, output name)
# # Example: f = Function('f',[x,y],[x,sin(y)*x],['x','y'],['r','q'])
#
# # states function
# F_x0 = ca.Function('F_x0', [x0_si], [x0_scaled, solver_bounds['lbw'],solver_bounds['ubw'], solver_bounds['lbg'],solver_bounds['ubg']],  \
#                            ['x0'], ['x0_scaled', 'lbw', 'ubw', 'lbg','ubg'])
#
# # time function
# F_tgrids = ca.Function('F_tgrids',[t0], [t0 + mpc.t_grid_coll, t0 + mpc.t_grid_x_coll, t0 + mpc.t_grid_u],
#                         ['t0'],['tgrid','tgrid_x','tgrid_u'])
#
# # reference function
# F_ref = ca.Function('F_ref', [t_grid, t_grid_x, t_grid_u], [ref], ['tgrid', 'tgrid_x', 'tgrid_u'],['ref'])
#
# # helper function
# helper_functions = ca.Function('helper_functions',[V], [V_shifted, u_si], ['V'], ['V_shifted', 'u0'])
#
# # integrator function (Inspired from F_sim = ca.Function('F_sim', [x0_si, u0_si], [xf_si, xa_si, xdot_si], ['x0', 'u0'], ['xf', 'xa', 'xdot']))
# F_int = ca.Function('F_int', [x0, z0, p0], [xf_si, zf_si, qf], ['x0', 'p', 'u0'], ['xf', 'xa', 'xdot'])
#
# # built-in aerodynamic forces
# F_aero = ca.Function('F_aero', [x0, u0], [F_ext, M_ext], ['x0', 'u0'], ['F_ext', 'M_ext'])
#
# ## ----------------- compile code-generated functions ----------------- #
# #
# # # compile MPC solver
# # if compile_mpc_solver:
# #
# #    # Solver code/shared library file names
# #    filename1 = foldername + 'mpc_solver.c'
# #    filename2 = foldername + 'mpc_solver.so'
# #
# #    # generate solver dependencies
# #    ts = time.time()
# #    mpc.solver.generate_dependencies('mpc_solver.c')
# #
# #    # compile dependencies
# #    os.system("mv ./mpc_solver.c" + " " + filename1)
# #    os.system("gcc -fPIC -shared -O3 " + filename1 + " -o " + filename2)
# #    tc = time.time() - ts
# #    print("INFO:   MPC solver compilation time: {} s".format(np.round(tc, 3)))
# #
# ## compile states function
# #if compile_x0_function:
# #
# #    # Solver code/shared library file names
# #    filename1 = foldername + 'F_x0.c'
# #    filename2 = foldername + 'F_x0.so'
# #
# #    # generate solver dependencies
# #    ts = time.time()
# #    F_x0.generate('F_x0.c')
# #
# #    # Compile dependencies
# #    os.system("mv F_x0.c"+" "+ filename1)
# #    os.system("gcc -fPIC -shared -O3 "+ filename1+" -o "+filename2)
# #    tc = time.time() - ts
# #    print("INFO:   Initialization functions compilation time: {} s".format(np.round(tc, 3)))
# #
# ## compile time reference function
# #if compile_tgrids_function:
# #
# #    # Solver code/shared library file names
# #    filename1 = foldername + 'F_tgrids.c'
# #    filename2 = foldername + 'F_tgrids.so'
# #
# #    # generate solver dependencies
# #    ts = time.time()
# #    F_tgrids.generate('F_tgrids.c')
# #
# #    # Compile dependencies
# #    os.system("mv F_tgrids.c"+" "+filename1)
# #    os.system("gcc -fPIC -shared -O3 "+filename1+" -o "+filename2)
# #    tc = time.time() - ts
# #    print("INFO:   Time grids functions compilation time: {} s".format(np.round(tc, 3)))
# #
# ## compile tracking reference
# #if compile_ref_function:
# #
# #    # Solver code/shared library file names
# #    filename1 = foldername + 'F_ref.c'
# #    filename2 = foldername + 'F_ref.so'
# #
# #    # generate solver dependencies
# #    ts = time.time()
# #    F_ref.generate('F_ref.c')
# #
# #    # Compile dependencies
# #    os.system("mv F_ref.c"+" "+filename1)
# #    os.system("gcc -fPIC -shared -O3 "+filename1+" -o "+filename2)
# #    tc = time.time() - ts
# #    print("INFO:   Reference functions compilation time: {} s".format(np.round(tc, 3)))
# #
# ## compile feedback function
# #if compile_feedback_function:
# #
# #    # Solver code/shared library file names
# #    filename1 = foldername + 'helper_functions.c'
# #    filename2 = foldername + 'helper_functions.so'
# #
# #    # generate solver dependencies
# #    ts = time.time()
# #    helper_functions.generate('helper_functions.c')
# #
# #    # Compile dependencies
# #    os.system("mv helper_functions.c"+" "+filename1)
# #    os.system("gcc -fPIC -shared -O3 "+filename1+" -o "+filename2)
# #    tc = time.time() - ts
# #    print("INFO:   Helper functions compilation time: {} s".format(np.round(tc, 3)))
# #
#
# # ----------------- compile integrator functions ----------------- #
#
# # compile integrator function
# if compile_integrator:
#
#     # Solver code/shared library file names
#     filename1 = foldername + 'F_int.c'
#     filename2 = foldername + 'F_int.so'
#
#     # generate solver dependencies
#     ts = time.time()
#     F_int.generate('F_int.c')
#
#     # Compile dependencies
#     os.system("mv F_int.c" + " " + filename1)
#     os.system("gcc -fPIC -shared -O3 " + filename1 + " -o " + filename2)
#     tc = time.time() - ts
#
# # ----------------- write files ----------------- #
#
# # initial states
# with open(foldername + 'x001_init.csv', 'w') as initial_state:
#
#     x0 = trial.model.variables_dict['x'](trial.optimization.V_final['x', 0])
#     field_names = collections.OrderedDict()
#     field_names['t'] = 0.0
#     for variable in list(trial.model.variables_dict['x'].keys()):
#         for dim in range(trial.model.variables_dict['x'][variable].shape[0]):
#             field_names['x_{}_{}'.format(variable, dim)] = x0[variable, dim].full()[0][0]
#     pcdw = csv.DictWriter(initial_state, delimiter=' ', fieldnames=field_names)
#     pcdw.writeheader()
#     ordered_dict = collections.OrderedDict(sorted(list(field_names.items())))
#     pcdw.writerow(ordered_dict)
#
# # initial guess
# with open(foldername + 'w001_init.csv', 'w') as initial_guess:
#
#     csvw = csv.writer(initial_guess, delimiter=' ')
#     csvw.writerow(mpc.w0.cat.full().squeeze().tolist())
#
# # initial controls
# with open(foldername + 'u001_init.csv', 'w') as feedback_law:
#
#     # u0 = trial.model.variables_dict['u'](trial.optimization.V_final['coll_var',0,0,'u'])
#     u0 = trial.model.variables_dict['u'](trial.optimization.V_final['u', 0])
#     field_names = collections.OrderedDict()
#     field_names['t'] = []
#     for variable in list(trial.model.variables_dict['u'].keys()):
#         for dim in range(trial.model.variables_dict['u'][variable].shape[0]):
#             field_names['u_{}_{}'.format(variable, dim)] = u0[variable, dim].full()[0][0]
#     pcdw = csv.DictWriter(feedback_law, delimiter=' ', fieldnames=field_names)
#     pcdw.writeheader()
#
# # write initial logging to csv
# with open(foldername + 'log001_init.csv', 'w') as mpc_logger:
#
#     field_names = collections.OrderedDict()
#     field_names['return_status'] = []
#     field_names['success'] = []
#     field_names['iter_count'] = []
#     field_names['t_wall_total'] = []
#     pcdw = csv.DictWriter(mpc_logger, delimiter=' ', fieldnames=field_names)
#     pcdw.writeheader()
#
# # initialise files for z
# with open(foldername + 'xa001_init.csv', 'w') as xa_out:
#
#     # xa = trial.model.variables_dict['z']
#     field_names = collections.OrderedDict()
#     field_names['t'] = []
#     for variable in list(trial.model.variables_dict['z'].keys()):
#         for dim in range(trial.model.variables_dict['z'][variable].shape[0]):
#             field_names['z_{}_{}'.format(variable, dim)] = []
#     pcdw = csv.DictWriter(xa_out, delimiter=' ', fieldnames=field_names)
#     pcdw.writeheader()
#
# # initialise files for xdot
# with open(foldername + 'dx001_init.csv', 'w') as dx_out:
#
#     dx = trial.model.variables_dict['xdot']
#     field_names = collections.OrderedDict()
#     field_names['t'] = []
#     for variable in list(trial.model.variables_dict['xdot'].keys()):
#         for dim in range(trial.model.variables_dict['xdot'][variable].shape[0]):
#             field_names['xdot_{}_{}'.format(variable, dim)] = []
#     pcdw = csv.DictWriter(dx_out, delimiter=' ', fieldnames=field_names)
#     pcdw.writeheader()
#
# # copy files
# os.system('./mpc_reset.sh '+foldername+' 1')
#
# """
# =========================
# = Part 2. Path tracking =
# =========================
# """
#
# # ----------------- initialize simulation with build-in features ----------------- #
# # TH: Can I create casadi objects such as x0, u0, z0 from code-generated functios ?
#
# # initialization
# #pickle save: system_model.variables_dict['x'] and pcikle load
# x0 = system_model.variables_dict['x'](0.0)  # initialize states
# u0 = system_model.variables_dict['u'](0.0)  # initialize controls
# z0 = dae.z(0.0)  # algebraic variables initial guess
#
#
# # open a file, where you ant to store the data
# import pickle
# with open(foldername+'pcikle_test.pckl', 'wb') as file:
#     pickle.dump(system_model.variables_dict, file)
#
# with open(foldername+'pcikle_test.pckl', 'rb') as file:
#     tmp = pickle.load(file)
#
# x0 = tmp['x'](0.0)
# u0 = tmp['u'](0.0)
# z01 = tmp['z'](0.0)
# z02 = dae.z(0.0)
#
# # scaled initial states
# plot_dict = trial.visualization.plot_dict
# x0['q10'] = np.array(plot_dict['x']['q10'])[:, -1] / scaling['x']['q10']
# x0['dq10'] = np.array(plot_dict['x']['dq10'])[:, -1] / scaling['x']['dq10']
# x0['omega10'] = np.array(plot_dict['x']['omega10'])[:, -1] / scaling['x']['omega10']
# x0['r10'] = np.array(plot_dict['x']['r10'])[:, -1] / scaling['x']['r10']
# x0['delta10'] = np.array(plot_dict['x']['delta10'])[:, -1] / scaling['x']['delta10']
# x0['l_t'] = np.array(plot_dict['x']['l_t'])[0, -1] / scaling['x']['l_t']
# x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0, -1] / scaling['x']['dl_t']
#
# # scaled algebraic vars
# z0['z'] = np.array(plot_dict['z']['lambda10'])[:, -1] / scaling['z']['lambda10']
#
# # get parameters
# vars0 = system_model.variables(0.0) # Also pickle / pickle everything
# vars0['theta'] = system_model.variables_dict['theta'](0.0)
#
# # ----------------- load initial states from file ----------------- #
#
# # read from file
# filename = foldername + 'x{:03d}.csv'.format(1)
# with open(filename, 'r') as initial_state:
#     reader = csv.reader(initial_state, delimiter=' ')
#     line = list(reader)[1]
#     x0_input = ca.vertcat(*list(map(lambda x: float(x), line[1:])))
#     t_input = float(line[0])
#
# # scale initial state
# F_x0 = ca.external('F_x0', foldername + 'F_x0.so')
# out_input = F_x0(x0=x0_input)
# x0_input = out_input['x0_scaled']
#
# # ----------------- load tracking reference from file ----------------- #
#
# # get cycle period
# filename = foldername + trial_name + '_theta.csv'
# with open(filename, 'r') as parameters:
#    reader = csv.DictReader(parameters, delimiter=' ')
#    for row in reader:
#        t_ref  = float(row['t_f'])
#
# # get reference time
# F_tgrids = ca.external('F_tgrids', foldername+'F_tgrids.so')
# tgrids = F_tgrids(t0 = t_input)
# for grid in list(tgrids.keys()):
#    tgrids[grid] = ca.vertcat(*list(map(lambda x: x % t_ref, tgrids[grid].full()))).full().squeeze()
#
# # get reference
# F_ref = ca.external('F_ref', foldername+'F_ref.so')
# tracking_reference = F_ref(tgrid = tgrids['tgrid'], tgrid_x = tgrids['tgrid_x'])['ref']
#
# # ----------------- run simulation with build-in features ----------------- #
#
# # initialize simulation
# xsim = [x0.cat.full().squeeze()]
# usim = [] #[u0.cat.full().squeeze()]
# fsim = []
# msim = []
# stats = []
#
# # Loop time step (TH: different from control intervals)
# for k in range(N_sim):
#
#     # current time
#     current_time = k*time_step
#     if (current_time % mpc_sampling_time) <= 1e-6:
#
#         # print message
#         print("iteration="+str(k)+"/"+str(N_sim)+", t="+"{:.3f}".format(current_time)+" > compute MPC step")
#
#         # evaluate controls
#         u0_call = mpc.step(x0, mpc_opts['mpc']['plot_flag'])
#         stats.append(mpc.solver.stats())
#
#         # fill in controls
#         u0['ddelta10'] = u0_call[6:9] # scaled!
#         u0['ddl_t'] = u0_call[-1] # scaled!
#
#         # TODO: replace MPC call using code-generated solvers
#         # TH: First call at t=0 doesn't return same result as build-in call?
#         # TH: Do I need to re-evaluate "tracking_reference" before each MPC call?
#         # TH: Do I need to re-evaluate "out_input" before each MPC call?
#         u0_call_input, stats_input = solve_mpc_step(foldername, x0_input, out_input, tracking_reference)
#
#     elif (current_time % mpc_sampling_time) > 1e-6:
#
#         # print evaluation
#         print("iteration="+str(k)+"/"+str(N_sim)+", t=" + "{:.3f}".format(current_time))
#
#     # evaluate forces and moments
#     vars0['x'] = x0 # scaled!
#     vars0['u'] = u0 # scaled!
#     z0 = dae.z(dae.rootfinder(z0, x0, p0))
#     vars0['xdot'] = z0['xdot']
#     vars0['z'] = z0['z']
#     outputs = system_model.outputs(system_model.outputs_fun(vars0, p0['param']))
#     F_ext = outputs['aerodynamics', 'f_aero_earth1']
#     M_ext = outputs['aerodynamics', 'm_aero_body1']
#
#
#     Aero_ext = ca.Function('F_aero_ext', [x0, u0], [F_ext, M_ext], ['x0', 'p', 'u0'], ['xf', 'xa', 'xdot'])
#     # Scaled x0, u0
#     # if compile_...
#     #    compile Functions...
#
#     # TODO: replace force calculation using code-generated solvers
#     # F_ext_input = None #...
#     # M_ext_input = None #...
#
#     # fill in forces and moments
#     u0['f_fict10'] = F_ext / scaling['u']['f_fict10'] # external force
#     u0['m_fict10'] = M_ext / scaling['u']['m_fict10'] # external moment
#
#     # fill controls into dae parameters
#     p0['u'] = u0
#
#     # evaluate integrator
#     out = integrator(x0 = x0, z0 = z0, p = p0)
#     z0 = out['zf']
#     x0 = out['xf']
#
#     # TODO: replace integrator call using code-generated solvers
#     # out_input = F_integrator(...)
#
#     # TODO: update x0 with results from integrator call
#     # x0_input = x0 # = out_input['xf']
#
#     # Simulation outputs
#     xsim.append(out['xf'].full().squeeze())
#     usim.append([u0.cat.full().squeeze()][0])
#     fsim.append(F_ext.full().squeeze())
#     msim.append(M_ext.full().squeeze())
#
# # ----------------- visualize results ----------------- #
#
# # plot reference path
# trial.plot(['isometric'])
# fig = plt.gcf()
# fig.set_size_inches(10, 8)
# ax = fig.get_axes()[0]
# l = ax.get_lines()
# l[0].set_color('b')
#
# # add external force simulation
# ax.plot([x[0] for x in xsim], [x[1] for x in xsim], [x[2] for x in xsim])
# l = ax.get_lines()
# l[-1].set_color('g')
#
# # layout/save
# ax.get_legend().remove()
# ax.legend([l[-1], l[-2], l[0]], ['fext', 'mpc', 'ref'], fontsize=14)
# figname = foldername + 'simulation_isometric.png'
# fig.savefig(figname)

