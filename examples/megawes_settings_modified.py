#!usr/bin/python3
import copy

import awebox as awe
import numpy as np
import casadi as ca
import csv

def set_megawes_settings(options):

    # --------------------------- Default aircraft/tether settings --------------------------- #
    # 6DOF megAWES model
    options['user_options.system_model.kite_dof'] = 6
    options['user_options.kite_standard'] = awe.megawes_data.data_dict()

    # tether parameters (incl. tether drag model)
    options['params.tether.cd'] = 1.2
    diam_t = 0.0297
    options['params.tether.rho'] = 0.6729*4/(np.pi*diam_t**2)
    options['user_options.trajectory.fixed_params'] = {'diam_t': diam_t}
    options['model.tether.use_wound_tether'] = False # don't model generator inertia
    options['model.tether.control_var'] = 'ddl_t' # tether acceleration control
    options['user_options.tether_drag_model'] = 'multi' 
    options['model.tether.aero_elements'] = 5

    # --------------------------- State bounds --------------------------- #
    # state variables bounds
    b = round(options['user_options.kite_standard']['geometry']['b_ref'], 1)
    omega_bound = 50*np.pi/180.0 
    delta_bound = 20*np.pi/180.0 
    options['model.system_bounds.x.q'] =  [np.array([0.0, -(300-b/2), 2*b]), np.array([1000.0, 300-b/2, 600.0])] # Spatial footprint [m]
    options['model.system_bounds.x.omega'] = [np.array(3*[-omega_bound]), np.array(3*[omega_bound])] # Angular rates [deg/s]
    options['user_options.kite_standard.geometry.delta_max'] = np.array(3*[delta_bound]) # Surface deflections [deg]
    options['model.system_bounds.x.l_t'] =  [10.0, 1e3] # Tether length [m]
    options['model.system_bounds.x.dl_t'] =   [-30.0, 30.0] # Tether speed [m/s]

    # --------------------------- Control bounds --------------------------- #
    # control variable bounds
    ddelta_bound = 5*np.pi/180.0 
    options['user_options.kite_standard.geometry.ddelta_max'] = np.array(3*[ddelta_bound]) # Deflection rates [deg/s]
    options['model.ground_station.ddl_t_max'] = 10. # Tether acceleration [m/s^2]

    # --------------------------- Operational constraints --------------------------- #
    # validitiy of aerodynamic model
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] = 10.0
    options['user_options.kite_standard.aero_validity.beta_min_deg'] = -10.0
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] = 4.2
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] = -14.5

    # airspeed limitation
    options['model.model_bounds.airspeed.include'] = True
    options['params.model_bounds.airspeed_limits'] = np.array([10., 120.]) 

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = True
    options['params.model_bounds.tether_force_limits'] = np.array([50, 1.7e6]) #[Eijkelhof2022]

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = True 
    options['model.model_bounds.acceleration.acc_max'] = 3. #[g]

    # constrained roll, pitch, yaw angles 
    options['model.model_bounds.rotation.include'] = True
    options['model.model_bounds.rotation.type'] = 'yaw'
    options['params.model_bounds.rot_angles'] = np.array([80.0*np.pi/180., 80.0*np.pi/180., 70.0*np.pi/180.0]) # default: 80, 80, 160

    # generator is not modelled
    options['model.model_bounds.wound_tether_length.include'] = False # default: True

    # --------------------------- Initialization --------------------------- #
    # initialization
    options['solver.initialization.groundspeed'] = 80. 
    options['solver.initialization.inclination_deg'] = 45. 
    options['solver.initialization.cone_deg'] = 25. 
    options['solver.initialization.l_t'] = 600.

    return options

def set_path_generation_options(options, gamma=0.75):

    # --------------------------- Adjust state bounds --------------------------- #
    # state variables bounds
    b = round(options['user_options.kite_standard']['geometry']['b_ref'], 1)
    options['model.system_bounds.x.q'] = [element + b/2 if i == 0 else element - b/2 for i, element in enumerate(options['model.system_bounds.x.q'])]
    options['user_options.kite_standard.geometry.delta_max'] *= gamma
    options['model.system_bounds.x.omega'] = [gamma * element for element in options['model.system_bounds.x.omega']]
    options['model.system_bounds.x.l_t'][1] *= gamma
    options['model.system_bounds.x.dl_t'] = [gamma * element for element in options['model.system_bounds.x.dl_t']]

    # --------------------------- Adjust control bounds --------------------------- #
    # control variable bounds
    options['user_options.kite_standard.geometry.ddelta_max'] *= gamma
    options['model.ground_station.ddl_t_max'] *= gamma

    # --------------------------- Adjust operational constraints --------------------------- #
    # validitiy of aerodynamic model
    options['model.model_bounds.aero_validity.include'] = True
    options['user_options.kite_standard.aero_validity.beta_max_deg'] *= gamma
    options['user_options.kite_standard.aero_validity.beta_min_deg'] *= gamma
    options['user_options.kite_standard.aero_validity.alpha_max_deg'] *= gamma
    options['user_options.kite_standard.aero_validity.alpha_min_deg'] *= gamma

    # airspeed limitation
    options['model.model_bounds.airspeed.include'] = True
    options['params.model_bounds.airspeed_limits'][1] *= gamma

    # tether force limit
    options['model.model_bounds.tether_stress.include'] = False
    options['model.model_bounds.tether_force.include'] = True
    options['params.model_bounds.tether_force_limits'][1] *= gamma

    # acceleration constraint
    options['model.model_bounds.acceleration.include'] = True
    options['model.model_bounds.acceleration.acc_max'] *= gamma

    # constrained roll, pitch, yaw angles
    options['model.model_bounds.rotation.include'] = True
    options['model.model_bounds.rotation.type'] = 'yaw'
    options['params.model_bounds.rot_angles'] *= gamma

    return options

def solve_mpc_step(foldername, x0, out, ref):

    # read initial guess
    fname = foldername + 'w{:03d}.csv'.format(1)
    with open(fname, 'r') as initial_guess:
       reader = csv.reader(initial_guess, delimiter = ' ', quoting = csv.QUOTE_NONNUMERIC)
       w0 = ca.vertcat(*list(reader)[0])

    # initialize mpc controller
    opts = {}
    opts['ipopt.linear_solver'] = 'ma57'
    opts['ipopt.max_iter'] = 2000
    opts['ipopt.max_cpu_time'] = 1e4 # seconds
    opts['ipopt.print_level'] = 0
    opts['ipopt.sb'] = "yes"
    opts['print_time'] = 0
    opts['record_time'] = 1
    solver = ca.nlpsol('solver', 'ipopt', foldername + 'mpc_solver.so', opts)

    # MPC weights
    nx = 23
    nu = 10
    Q = np.ones((nx,1))
    R = np.ones((nu,1))
    P = np.ones((nx,1))

    # solve MPC problem
    u_ref = 10.
    sol = solver(x0 = w0, lbx = out['lbw'], ubx = out['ubw'], lbg = out['lbg'], ubg = out['ubg'],
                 p = ca.vertcat(x0, ref, u_ref, Q, R, P))
    print('Compute MPC step')

    # Log MPC
    stats = solver.stats()
    filename = foldername + 'log{:03d}.csv'.format(1)
    with open(filename, 'r') as mpc_log:
       reader = csv.reader(mpc_log, delimiter = ' ')
       log_header = list(reader)[0]

    line = []
    for stat in log_header:
       line.append(stats[stat])

    with open(filename, 'a') as mpc_log:
       csvw = csv.writer(mpc_log, delimiter = ' ')
       csvw.writerow(line)

    # MPC outputs
    helpers = ca.external('helper_functions', foldername + 'helper_functions.so')
    out = helpers(V=sol['x'])

    # write shifted initial guess
    V_shifted = out['V_shifted']
    filename = foldername + 'w{:03d}.csv'.format(1)
    with open(filename, 'w') as initial_guess:
        csvw = csv.writer(initial_guess, delimiter=' ')
        csvw.writerow(V_shifted.full().squeeze().tolist())

    # write new controls
    u0 = out['u0']
    filename = foldername + 'u{:03d}.csv'.format(1)
    with open(filename, 'r') as controls:
        reader = csv.reader(controls, delimiter=' ')
        header = list(reader)[0]
    lines = [header, u0.full().squeeze().tolist()]
    with open(filename, 'w') as feedback_law:
        csvw = csv.writer(feedback_law, delimiter=' ')
        csvw.writerows(lines)
    # print(fname, ' written...')

    return u0, stats

