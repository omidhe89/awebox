#!/usr/bin/python3
"""
Generation of optimal trajectory using MegAWES aircraft
:authors: Jochem De Schutter, Thomas Haas
:date: 26/10/2023
"""

import awebox as awe
import awebox.tools.integrator_routines as awe_integrators
import awebox.pmpc as pmpc
from megawes_settings_modified import set_megawes_settings
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

# ----------------- initialization ----------------- #

# case settings
foldername="/cfdfile2/data/fm/thomash/Devs/PycharmProjects/awebox/awebox_compiled_solvers/output_files/"
# foldername="/cfdfile2/data/fm/thomash/Tools/awebox/experimental/output_files/"
trial_name="megawes_uniform_1loop"
if not os.path.isdir(foldername):
    os.system('mkdir '+foldername)

# user settings
N_nlp = 60
N_sim = 200
N_mpc = 10
N_dt = 20
mpc_sampling_time = 0.1
time_step = mpc_sampling_time/N_dt

# outputs/compilation flags
autosave = False
COMPILATION_FLAG = False
compile_x0_function = COMPILATION_FLAG
compile_tgrids_function = COMPILATION_FLAG
compile_ref_function = COMPILATION_FLAG
compile_feedback_function = COMPILATION_FLAG
compile_aero_function = COMPILATION_FLAG
compile_integrator_function = COMPILATION_FLAG
compile_mpc_solver = False # independent flag due to large compilation time
filesave = False

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

# ----------------- build and optimize OCP trial ----------------- #

# get architecture
arch = awe.mdl.architecture.Architecture(options['user_options.system_model.architecture'])

# initialize trial
trial = awe.Trial(options, trial_name)

# build trial
trial.build()

# optimize trial
trial.optimize(options_seed=options)

# ----------------- save optimization results ----------------- #
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

# create PMPC object
mpc = pmpc.Pmpc(mpc_opts['mpc'], mpc_sampling_time, trial)

# save solver bounds
for var in list(mpc.solver_bounds.keys()):
    filename = foldername + var + '_bounds.pckl'
    with open(filename, 'wb') as handle:
        pickle.dump(mpc.solver_bounds[var], handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------- compile mpc solver ----------------- #
if compile_mpc_solver:
   src_filename = foldername + 'mpc_solver.c'
   lib_filename = foldername + 'mpc_solver.so'
   mpc.solver.generate_dependencies('mpc_solver.c')
   os.system("mv ./mpc_solver.c" + " " + src_filename)
   os.system("gcc -fPIC -shared -O3 " + src_filename + " -o " + lib_filename)

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

# Compile dependencies
if compile_integrator_function:
    src_filename = foldername + 'F_int.c'
    lib_filename = foldername + 'F_int.so'
    F_int.generate('F_int.c')
    os.system("mv F_int.c "+ src_filename)
    os.system("gcc -fPIC -shared -O3 "+ src_filename + " -o " + lib_filename)

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

# Compile dependencies
if compile_aero_function:
    src_filename = foldername + 'F_aero.c'
    lib_filename = foldername + 'F_aero.so'
    F_aero.generate('F_aero.c')
    os.system("mv F_aero.c "+ src_filename)
    os.system("gcc -fPIC -shared -O3 "+ src_filename + " -o " + lib_filename)

# ----------------- build states function for MPC ----------------- #
# F_x0: Returns states x0 in SI units from scaled states x0_scaled and solver bounds solve_bounds

# states in SI-units in symbolic form
x0_si = trial.model.variables_dict['x']

# scaled states in symbolic form
x0_scaled = []
for name in list(trial.model.variables_dict['x'].keys()):
    x0_scaled.append(x0_si[name]/scaling['x'][name])
x0_scaled = ca.vertcat(*x0_scaled)

# solver bounds
solver_bounds = mpc.solver_bounds

# states function
F_x0 = ca.Function('F_x0', [x0_si], [x0_scaled, solver_bounds['lbw'],solver_bounds['ubw'], solver_bounds['lbg'],solver_bounds['ubg']],  \
                           ['x0'], ['x0_scaled', 'lbw', 'ubw', 'lbg','ubg'])

# compile states function
if compile_x0_function:
   src_filename = foldername + 'F_x0.c'
   lib_filename = foldername + 'F_x0.so'
   F_x0.generate('F_x0.c')
   os.system("mv F_x0.c"+" "+ src_filename)
   os.system("gcc -fPIC -shared -O3 "+ src_filename+" -o "+lib_filename)

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

# compile time reference function
if compile_tgrids_function:
   src_filename = foldername + 'F_tgrids.c'
   lib_filename = foldername + 'F_tgrids.so'
   F_tgrids.generate('F_tgrids.c')
   os.system("mv F_tgrids.c"+" "+src_filename)
   os.system("gcc -fPIC -shared -O3 "+src_filename+" -o "+lib_filename)

# ----------------- build reference function for MPC ----------------- #
# F_ref: Returns tracked reference on specified time grid

# reference function
ref = mpc.get_reference(t_grid, t_grid_x, t_grid_u)

# reference function
F_ref = ca.Function('F_ref', [t_grid, t_grid_x, t_grid_u], [ref], ['tgrid', 'tgrid_x', 'tgrid_u'],['ref'])

# compile tracking reference
if compile_ref_function:
   src_filename = foldername + 'F_ref.c'
   lib_filename = foldername + 'F_ref.so'
   F_ref.generate('F_ref.c')
   os.system("mv F_ref.c"+" "+src_filename)
   os.system("gcc -fPIC -shared -O3 "+src_filename+" -o "+lib_filename)

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

# compile feedback function
if compile_feedback_function:
   src_filename = foldername + 'helper_functions.c'
   lib_filename = foldername + 'helper_functions.so'
   helper_functions.generate('helper_functions.c')
   os.system("mv helper_functions.c"+" "+src_filename)
   os.system("gcc -fPIC -shared -O3 "+src_filename+" -o "+lib_filename)

# ----------------- create CasADi structures to initialize simulation ----------------- #

# create CasADi symbolic variables and dicts
x0 = system_model.variables_dict['x'](0.0)  # initialize states
u0 = system_model.variables_dict['u'](0.0)  # initialize controls
z0 = dae.z(0.0)  # algebraic variables initial guess
w0 = mpc.w0
vars0 = system_model.variables(0.0)
vars0['theta'] = system_model.variables_dict['theta'](0.0)

# gather into dict
simulation_variables = {'x0':x0, 'u0':u0, 'z0':z0, 'p0':p0, 'w0':w0,
                        'vars0':vars0, 'scaling':scaling}

filename = foldername + 'simulation_variables.pckl'
with open(filename, 'wb') as handle:
    pickle.dump(simulation_variables, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------- write files ----------------- #

if filesave == True: 

    # initial states
    with open(foldername + 'x001_init.csv', 'w') as initial_state:
    
        x0_out = trial.model.variables_dict['x'](trial.optimization.V_final['x', 0])
        field_names = collections.OrderedDict()
        field_names['t'] = 0.0
        for variable in list(trial.model.variables_dict['x'].keys()):
            for dim in range(trial.model.variables_dict['x'][variable].shape[0]):
                field_names['x_{}_{}'.format(variable, dim)] = x0_out[variable, dim].full()[0][0]
        pcdw = csv.DictWriter(initial_state, delimiter=' ', fieldnames=field_names)
        pcdw.writeheader()
        ordered_dict = collections.OrderedDict(sorted(list(field_names.items())))
        pcdw.writerow(ordered_dict)
    
    # initial guess
    with open(foldername + 'w001_init.csv', 'w') as initial_guess:
    
        csvw = csv.writer(initial_guess, delimiter=' ')
        csvw.writerow(mpc.w0.cat.full().squeeze().tolist())
    
    # initial controls
    with open(foldername + 'u001_init.csv', 'w') as feedback_law:
    
        u0_out = trial.model.variables_dict['u'](trial.optimization.V_final['u', 0])
        field_names = collections.OrderedDict()
        field_names['t'] = []
        for variable in list(trial.model.variables_dict['u'].keys()):
            for dim in range(trial.model.variables_dict['u'][variable].shape[0]):
                field_names['u_{}_{}'.format(variable, dim)] = u0_out[variable, dim].full()[0][0]
        pcdw = csv.DictWriter(feedback_law, delimiter=' ', fieldnames=field_names)
        pcdw.writeheader()
    
    # write initial logging to csv
    with open(foldername + 'log001_init.csv', 'w') as mpc_logger:
    
        field_names = collections.OrderedDict()
        field_names['return_status'] = []
        field_names['success'] = []
        field_names['iter_count'] = []
        field_names['t_wall_total'] = []
        pcdw = csv.DictWriter(mpc_logger, delimiter=' ', fieldnames=field_names)
        pcdw.writeheader()
    
    # initialise files for z
    with open(foldername + 'xa001_init.csv', 'w') as algebraic_vars:
    
        # xa = trial.model.variables_dict['z']
        field_names = collections.OrderedDict()
        field_names['t'] = []
        for variable in list(trial.model.variables_dict['z'].keys()):
            for dim in range(trial.model.variables_dict['z'][variable].shape[0]):
                field_names['z_{}_{}'.format(variable, dim)] = []
        pcdw = csv.DictWriter(algebraic_vars, delimiter=' ', fieldnames=field_names)
        pcdw.writeheader()
    
    # initialise files for xdot
    with open(foldername + 'dx001_init.csv', 'w') as states_deriv:
    
        field_names = collections.OrderedDict()
        field_names['t'] = []
        for variable in list(trial.model.variables_dict['xdot'].keys()):
            for dim in range(trial.model.variables_dict['xdot'][variable].shape[0]):
                field_names['xdot_{}_{}'.format(variable, dim)] = []
        pcdw = csv.DictWriter(states_deriv, delimiter=' ', fieldnames=field_names)
        pcdw.writeheader()

# ----------------- end ----------------- #

