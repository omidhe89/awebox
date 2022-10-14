
import awebox as awe
import awebox.mdl.model as model
import awebox.mdl.architecture as archi
import casadi as ca
from ampyx_ap2_settings import set_ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
import awebox.opts.options as opts
from scipy.interpolate import interp1d

EXTERNAL_FORCES = True

# indicate desired system architecture
# here: single kite with 3DOF Ampyx AP2 model
options_seed = {}
options_seed['user_options.system_model.architecture'] = {1:0}
#options_seed = set_ampyx_ap2_settings(options_seed)
options_seed['user_options.kite_standard'] = awe.ampyx_data.data_dict()
options_seed['user_options.system_model.kite_dof'] = 3
options_seed['model.tether.control_var'] = 'ddl_t'

# indicate desired operation mode
# here: lift-mode system with pumping-cycle operation, with a one winding trajectory
options_seed['user_options.trajectory.system_type'] = 'lift_mode'
options_seed['user_options.trajectory.type'] = 'power_cycle'
options_seed['user_options.trajectory.lift_mode.windings'] = 1
options_seed['model.system_bounds.theta.t_f'] =  [5.0, 15.0]

# indicate desired environment
# here: wind velocity profile according to power-law
#options_seed['params.wind.z_ref'] = 100.0
#options_seed['params.wind.power_wind.exp_ref'] = 0.15
#options_seed['user_options.wind.model'] = 'power'
#options_seed['user_options.wind.u_ref'] = 10.
options_seed['params.wind.z_ref'] = 10.0
options_seed['user_options.wind.model'] = 'log_wind'
options_seed['user_options.wind.u_ref'] = 5.

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps within ipopt.
options_seed['nlp.n_k'] = 40
options_seed['nlp.collocation.u_param'] = 'zoh'
options_seed['user_options.trajectory.lift_mode.phase_fix'] = 'simple'
options_seed['solver.linear_solver'] = 'ma57' # if HSL is installed, otherwise 'mumps'
options_seed['solver.mu_hippo'] = 1e-2

# build and optimize the NLP (trial)
trial = awe.Trial(options_seed, 'Ampyx_AP2')
trial.build()
trial.optimize()
trial.plot(['isometric'])
plt.show()

# Outputs
plot_dict = trial.visualization.plot_dict

# flag for external forces
if EXTERNAL_FORCES:
    options_seed['model.aero.fictitious_embedding'] = 'substitute'

options = opts.Options()
options.fill_in_seed(options_seed)
architecture = archi.Architecture(options['user_options']['system_model']['architecture'])
options.build(architecture)

system_model = model.Model()
system_model.build(options['model'], architecture)
scaling = system_model.scaling
dae = system_model.get_dae()

# fill in system design parameters
p0 = dae.p(0.0)

# fill in tether diameter (and possible other parameters)
theta = system_model.variables_dict['theta'](0.0)
theta['diam_t'] = 2e-3 / scaling['theta']['diam_t']
theta['t_f'] = 1.0
p0['theta'] = theta.cat

# get numerical parameters
params = system_model.parameters(0.0)

#fill in numerical parameters
param_num = options['model']['params']
for param_type in list(param_num.keys()):
    if isinstance(param_num[param_type],dict):
        for param in list(param_num[param_type].keys()):
            if isinstance(param_num[param_type][param],dict):
                for subparam in list(param_num[param_type][param].keys()):
                    params['theta0',param_type,param,subparam] = param_num[param_type][param][subparam]

            else:
                params['theta0',param_type,param] = param_num[param_type][param]
    else:
        params['theta0',param_type] = param_num[param_type]

if EXTERNAL_FORCES:
    params['phi', 'gamma'] = 1

p0['param'] = params

# make casadi collocation integrator
ts = 0.05 # sampling time
int_opts = {}
int_opts['tf'] = ts
int_opts['number_of_finite_elements'] = 40
int_opts['collocation_scheme'] = 'radau'
int_opts['interpolation_order'] = 4
int_opts['rootfinder'] = 'fast_newton'
integrator = ca.integrator('integrator', 'collocation', dae.dae, int_opts)

# simulate system (TBD)
u0 = system_model.variables_dict['u'](0.0) # initialize controls
x0 = system_model.variables_dict['x'](0.0) # initialize controls
# x0['q10'] = np.array([50, 0, 50]) / scaling['x']['q10']
# x0['r10'] = ca.reshape(np.eye(3), 9, 1)
# x0['l_t'] = np.sqrt(50**2 + 50**2) / scaling['x']['l_t']
x0['q10'] = np.array(plot_dict['x']['q10'])[:,-1] / scaling['x']['q10']
x0['dq10'] = np.array(plot_dict['x']['dq10'])[:,-1] / scaling['x']['dq10']
x0['coeff10'] = np.array(plot_dict['x']['coeff10'])[:,-1] / scaling['x']['coeff10']
#x0['r10'] = np.array(plot_dict['x']['r10'])[:,-1] / scaling['x']['r10'] #x0['r10'] = ca.reshape(np.eye(3), 9, 1)
x0['l_t'] = np.array(plot_dict['x']['l_t'])[0,-1] / scaling['x']['l_t']
x0['dl_t'] = np.array(plot_dict['x']['dl_t'])[0,-1] / scaling['x']['dl_t']
#x0['delta10'] = np.array(plot_dict['x']['delta10'])[:,-1] / scaling['x']['delta10']
z0 = dae.z(0.0) # algebraic variables initial guess

# Reference forces and moments
N = 100
f0 = np.expand_dims(np.array(plot_dict['outputs']['aerodynamics']['f_aero_earth1'])[:,-1], axis=1)
#m0 = np.expand_dims(np.array(plot_dict['outputs']['aerodynamics']['m_aero_body1'])[:,-1], axis=1)
t_out = np.insert(np.expand_dims(np.array(plot_dict['time_grids']['ip']), axis=1), 0, 0.0)
f_out = np.concatenate((f0, np.array(plot_dict['outputs']['aerodynamics']['f_aero_earth1'])), axis=1)
#m_out = np.concatenate((m0, np.array(plot_dict['outputs']['aerodynamics']['m_aero_body1'])), axis=1)
t_ref = np.arange(0,N)*ts
f_ref = np.array([np.interp(t_ref, t_out, f_out[i,:]) for i in range(3)])
#m_ref = np.array([np.interp(t_ref, t_out, m_out[i,:]) for i in range(3)])

#
xsim = [x0.cat.full().squeeze()]
for k in range(N):

    # fill in controls
    u0['f_fict10'] = f_ref[:,k] / scaling['u']['f_fict10'] # external force
    # u0['m_fict10'] = m_ref[:,k] / scaling['u']['m_fict10'] # external moment
    # u0['ddl_t'] = ... / scaling['u']['ddl_t'] # tether acceleration control (scaled)
    # u0['ddelta10'] = 0.0 / scaling['u']['ddelta10'] # not relevant in case of external forces (scaled)
    
    # fill controls into dae parameters
    p0['u'] = u0

    # if desired, change model parameter (e.g. wind speed, relevant for tether drag)
    params['theta0', 'wind', 'u_ref'] = 5.0
    p0['param'] = params

    # evaluate integrator
    out = integrator(x0 = x0, p = p0, z0 = z0)
    z0 = out['zf']
    x0 = out['xf']
    xsim.append(out['xf'].full().squeeze())


trial.plot(['isometric'])
fig = plt.gcf()
ax = fig.get_axes()[0]
ax.plot([x[0] for x in xsim], [x[1] for x in xsim], [x[2] for x in xsim], 'g')
l = ax.get_lines()
l[0].set_color('b')
ax.get_legend().remove()
ax.legend([l[0], l[-1]], ['ref', 'ext. F'])
plt.show()
