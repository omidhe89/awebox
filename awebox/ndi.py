import awebox as awe
import awebox.viz.tools as viz_tools
import casadi as cas
from awebox.logger.logger import Logger as awelogger
import matplotlib.pyplot as plt
import numpy as np
import copy
import awebox.mdl.architecture as archi
        
class Ndi():

    def __init__(self, sim_options, ts, trial):
        """ Constructor.
        """
        self.__pocp_trial = trial
        self.__ts = ts
        self.__N = sim_options['N']
        self.__ndi_options = sim_options
        # store model data
        self.__var_list = ['x', 'z', 'u']
        self.__nx = trial.model.variables['x'].shape[0]
        self.__nu = trial.model.variables['u'].shape[0]
        self.__nz = trial.model.variables['z'].shape[0]

        options = copy.deepcopy(trial.options_seed)
        options['visualization.cosmetics.plot_ref'] = True
        fixed_params = {}
        for name in list(self.__pocp_trial.model.variables_dict['theta'].keys()):
            if name != 't_f':
                fixed_params[name] = self.__pocp_trial.optimization.V_final['theta',name].full()
        fixed_params['t_f'] = self.__N*self.__ts
        options['user_options.trajectory.fixed_params'] = fixed_params

        self.G = []
        self.F = []


        self.__trial = awe.Trial(seed = options)
        architecture = archi.Architecture(self.__trial.options['user_options']['system_model']['architecture'])
        self.__build_controller(architecture)
        

 
        self.u_ndi = []
        self.A_des = cas.SX_eye(3)
    
    
    
    def __build_controller(self, architecture):

        """ Build options, model, and necessary controller elements """

        awelogger.logger.info("Building NDI trial...")

        # build
        self.__trial.options.build(architecture)
        self.__trial.model.build(self.__trial.options['model'], architecture)
        self.F, self.G = self.__extract_aerodynamic(architecture)
        return None
    
    def __extract_aerodynamic(self, architecture):
        kite_nodes = architecture.kite_nodes
        parent_map = architecture.parent_map

        model_dynamics = self._Ndi__trial.model.outputs['model_for_control']
        # kite_dynamics = []
        # for kite in kite_nodes:
        #     kite_dynamics[kite] = cas.horzsplit(model_dynamics,12)
        f_rot, g_rot_c1, g_rot_c2, g_rot_c3 = cas.vertsplit(model_dynamics,3)
        g_rot = cas.horzcat(g_rot_c1, g_rot_c2, g_rot_c3)
        return f_rot, g_rot
    
    def get_reference(self, t_grid, t_grid_x, t_grid_u):
        """ Interpolate reference on NLP time grids.
        """

        ip_dict = {}
        V_ref = self.__trial.nlp.V(0.0)
        for var_type in self.__var_list:
            ip_dict[var_type] = []
            for name in list(self.__trial.model.variables_dict[var_type].keys()):
                for dim in range(self.__trial.model.variables_dict[var_type][name].shape[0]):
                    if var_type == 'x':
                        ip_dict[var_type].append(self.__interpolator(t_grid_x, name, dim,var_type))
                    elif (var_type == 'u') and self.__mpc_options['u_param']:
                        ip_dict[var_type].append(self.__interpolator(t_grid_u, name, dim,var_type))
                    else:
                        ip_dict[var_type].append(self.__interpolator(t_grid, name, dim,var_type))
            if self.__mpc_options['ref_interpolator'] == 'poly':
                ip_dict[var_type] = cas.horzcat(*ip_dict[var_type]).T
            elif self.__mpc_options['ref_interpolator'] == 'spline':
                ip_dict[var_type] = cas.vertcat(*ip_dict[var_type])

        counter = 0
        counter_x = 0
        counter_u = 0
        V_list = []

        for name in self.__trial.model.variables_dict['theta'].keys():
            if name != 't_f':
                V_list.append(self.__pocp_trial.optimization.V_opt['theta',name])
            else:
                V_list.append(self.__N*self.__ts)
        V_list.append(np.zeros(V_ref['phi'].shape))
        V_list.append(np.zeros(V_ref['xi'].shape))

        for k in range(self.__N):
            for j in range(self.__trial.nlp.d+1):
                if j == 0:
                    V_list.append(ip_dict['x'][:,counter_x])
                    counter_x += 1

                    if self.__mpc_options['u_param'] == 'zoh':
                        V_list.append(ip_dict['u'][:, counter_u])
                        V_list.append(np.zeros((self.__nx, 1)))
                        V_list.append(np.zeros((self.__nz, 1)))
                        counter_u += 1
                else:
                    for var_type in self.__var_list:
                        if var_type == 'x':
                            V_list.append(ip_dict[var_type][:,counter_x])
                            counter_x += 1
                        elif var_type == 'z' or (var_type == 'u' and self.__mpc_options['u_param']=='poly'):
                            V_list.append(ip_dict[var_type][:,counter])
                    counter += 1

        V_list.append(ip_dict['x'][:,counter_x])

        V_ref = V_ref(cas.vertcat(*V_list))

        return V_ref
    """
    def build(self, options, architecture):
        
        super().build(options, architecture)  # Call the Model's build method
        self.__parameters = self.parameters
        self.__variables = self.variables
        self.__outputs = self.outputs
        self.F, self.G = self.extract_aerodynamic(self, self.__variables, self.__outputs['aerodynamics']['m_aero_body_1'] , self.__parameters, self.__outputs, architecture) #model.f_nodes
        self.u_ndi = self.rotation_ndi_controller(self, self.__variables, architecture)  
        return None

    def extract_aerodynamic(self, variables, f_nodes, parameters, outputs, architecture):
        kite_nodes = architecture.kite_nodes
        parent_map = architecture.parent_map

        j_inertia = parameters['theta0', 'geometry', 'j']
        F = self.F
        G = self.G
        x = variables['SI']['x']
        
        for kite in kite_nodes:
            parent = parent_map[kite]
            moment = f_nodes['m' + str(kite) + str(parent)]
            omega = x['omega' + str(kite) + str(parent)]
            delta = variables['SI']['x']['delta' + str(kite) + str(parent)]
            tether_moment = outputs['tether_moments']['n{}{}'.format(kite, parent)]

            moment_aero_star = cas.substitute(moment,delta, cas.SX.zeros(3,1))
            tmp = cas.simplify(moment - moment_aero_star)
            moment_aero_delta = cas.jacobian(tmp, delta)

            G['kite' + str(kite) + str(parent) + '_1'] = cas.mtimes(cas.inv(j_inertia), moment_aero_delta)
            omega_cross_J_omega = vect_op.cross(omega, cas.mtimes(j_inertia, omega))
            F['kite' + str(kite) + str(parent) + '_1'] = cas.mtimes(cas.inv(j_inertia), moment_aero_star - (omega_cross_J_omega + tether_moment))
        
        return F, G

    def rotation_ndi_controller(self, variables, architecture):

        u_ndi = self.u_ndi
        A_des = self.A_des

        kite_nodes = architecture.kite_nodes
        parent_map = architecture.parent_map
        x = variables['SI']['x']
        
        for kite in kite_nodes:
            parent = parent_map[kite]
            omega = x['omega' + str(kite) + str(parent)]
            u_ndi = [u_ndi, cas.mtimes(-cas.inv(self.G['kite' + str(kite) + str(parent) + '_1']),self.F['kite' + str(kite) + str(parent)+ '_1'] + cas.mtimes(A_des, omega))]    


        return u_ndi
    """