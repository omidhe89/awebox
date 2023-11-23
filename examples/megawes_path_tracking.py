#!/usr/bin/python3
"""
NMPC tracking of optimal path using MegAWES aircraft
:authors: Jochem De Schutter, Thomas Haas
:date: 26/10/2023
"""

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import os
import csv
import pickle
# import awebox as awe
# import awebox.tools.integrator_routines as awe_integrators
# import awebox.pmpc as pmpc
# import awebox.sim as sim
# from megawes_settings_modified import set_megawes_settings, set_path_generation_options, set_path_tracking_trial_options, solve_mpc_step
# import logging
# import time
# import copy
# import collections

# ----------------- import results from AWEbox ----------------- #
def csv2dict(fname):
    '''
    Import CSV outputs from awebox to Python dictionary
    '''

    # read csv file
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)

        # get fieldnames from DictReader object and store in list
        headers = reader.fieldnames

        # store data in columns
        columns = {}
        for row in reader:
            for fieldname in headers:
                val = row.get(fieldname).strip('[]')
                if val == '':
                    val = '0.0'
                columns.setdefault(fieldname, []).append(float(val))

    # add periodicity
    for fieldname in headers:
        columns.setdefault(fieldname, []).insert(0, columns[fieldname][-1])
    columns['time'][0] = 0.0

    return columns

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

# ----------------- case initialization ----------------- #

# case
# TH: Specify output folder
foldername="/cfdfile2/data/fm/thomash/Devs/PycharmProjects/awebox/awebox_compiled_solvers/output_files/"
# foldername="/cfdfile2/data/fm/thomash/Tools/awebox/experimental/output_files/"
trial_name="megawes_uniform_1loop"
if not os.path.isdir(foldername):
    os.system('mkdir '+foldername)

# user settings
N_sim = 6000
N_dt = 20
mpc_sampling_time = 0.1
time_step = mpc_sampling_time/N_dt
N_max_fail = 20

# MPC options
opts = {}
opts['ipopt.linear_solver'] = 'ma57'
opts['ipopt.max_iter'] = 200
opts['ipopt.max_cpu_time'] = 5.  # seconds
opts['ipopt.print_level'] = 0
opts['ipopt.sb'] = "yes"
opts['print_time'] = 0
opts['record_time'] = 1

# ----------------- load optimization data ----------------- #

# load parameters and CasADi symbolic variables
filename = foldername + 'simulation_variables.pckl'
with open(filename, 'rb') as handle:
    struct = pickle.load(handle)
x0 = struct['x0']
u0 = struct['u0']
z0 = struct['z0']
p0 = struct['p0']
w0 = struct['w0']
vars0 = struct['vars0']
scaling = struct['scaling']

# load solver bounds
bounds = {}
for var in ['lbw', 'ubw', 'lbg', 'ubg']:
    filename = foldername + var + '_bounds.pckl'
    with open(filename, 'rb') as handle:
        bounds[var] = pickle.load(handle)

# load trial results
filename = foldername + trial_name+'_results.csv'
awes = csv2dict(filename)

# load trial parameters
filename = foldername + trial_name+'_theta.csv'
with open(filename, 'r') as file:
    csv_reader = csv.reader(file, delimiter=' ')
    lines = [line for line in csv_reader]
    theta = [float(value) for value in lines[1]]
t_f = theta[1]

# ----------------- initialize states/alg. vars ----------------- #

# Scaled initial states
x0['q10'] = np.array([awes['x_q10_'+str(i)][-1] for i in range(3)]) / scaling['x']['q10']
x0['dq10'] = np.array([awes['x_dq10_'+str(i)][-1] for i in range(3)]) / scaling['x']['dq10']
x0['omega10'] = np.array([awes['x_omega10_'+str(i)][-1] for i in range(3)]) / scaling['x']['omega10']
x0['r10'] = np.array([awes['x_r10_'+str(i)][-1] for i in range(9)]) / scaling['x']['r10']
x0['delta10'] = np.array([awes['x_delta10_'+str(i)][-1] for i in range(3)]) / scaling['x']['delta10']
x0['l_t'] = np.array(awes['x_l_t_0'][-1]) / scaling['x']['l_t']
x0['dl_t'] = np.array(awes['x_dl_t_0'][-1]) / scaling['x']['dl_t']

# Scaled algebraic vars
z0['z'] = np.array(awes['z_lambda10_0'][-1]) / scaling['z']['lambda10']

# ----------------- initialize MPC controller ----------------- #

# MPC weights
nx = 23
nu = 10
Q = np.ones((nx, 1))
R = np.ones((nu, 1))
P = np.ones((nx, 1))

# Load function objects and solver
F_tgrids = ca.external('F_tgrids', foldername + 'F_tgrids.so')
F_ref = ca.external('F_ref', foldername + 'F_ref.so')
F_aero = ca.external('F_aero', foldername + 'F_aero.so')
F_int = ca.external('F_int', foldername + 'F_int.so')
helpers = ca.external('helper_functions', foldername + 'helper_functions.so')
solver = ca.nlpsol('solver', 'ipopt', foldername + 'mpc_solver.so', opts)

# Load evaluation functions g_fun and P_fun
filename = foldername + "F_gfun.pckl"
with open(filename, 'rb') as handle:
    F_gfun = pickle.load(handle)

filename = foldername + "F_pfun.pckl"
with open(filename, 'rb') as handle:
    F_pfun = pickle.load(handle)

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
        sol = solver(x0=w0, lbx=bounds['lbw'], ubx=bounds['ubw'], lbg=bounds['lbg'], ubg=bounds['ubg'],
                     p=ca.vertcat(x0, ref, u_ref, Q, R, P))

        # MPC stats
        stats.append(solver.stats())
        if stats[-1]["success"]==False:
            N_mpc_fail += 1
            #error = F_gfun(sol['x'] - F_pfun(ca.vertcat(x0, ref, u_ref, Q, R, P))).full()

        # MPC outputs
        out = helpers(V=sol['x'])

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
for m in range(N):
    k = m*int(len(awes['time'])/N)
    ax.plot([0.0, awes['x_q10_0'][k]], [0.0, awes['x_q10_1'][k]], [0.0, awes['x_q10_2'][k]], color='lightgray', linewidth=0.8)
ax.plot(awes['x_q10_0'], awes['x_q10_1'], awes['x_q10_2'])

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

# ----------------- end ----------------- #
