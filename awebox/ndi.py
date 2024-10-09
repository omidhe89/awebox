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
        options['nlp.discretization'] = 'direct_collocation'
        options['nlp.n_k'] = self.__N
        options['nlp.collocation.u_param'] = 'zoh'
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
        
        self.__create_reference_interpolator()
        # periodic indexing
        self.__index = 0

       
        
        self.u_ndi = []
        self.A_des = cas.diag(sim_options['ctrl_params'])
     
    def __build_controller(self, architecture):

        """ Build options, model, and necessary controller elements """

        awelogger.logger.info("Building NDI trial...")

        # build
        self.__trial.options.build(architecture)
        self.__trial.model.build(self.__trial.options['model'], architecture)
        self.F, self.G = self.__extract_aerodynamic(architecture)
        self.__trial.formulation.build(self.__trial.options['formulation'], self.__trial.model)
        self.__trial.nlp.build(self.__trial.options['nlp'], self.__trial.model, self.__trial.formulation)
        return None
    
    def __extract_aerodynamic(self, architecture):
        kite_nodes = architecture.kite_nodes
        parent_map = architecture.parent_map
        
        model_dynamics = self.__trial.model.rot_dyn_dict
        f_rot_funcs = []
        g_rot_funcs = []
        for kite in kite_nodes:
            f_rot_funcs = [f_rot_funcs, model_dynamics['F' + str(kite)]]
            g_rot_funcs = [g_rot_funcs, model_dynamics['G' + str(kite)]]
        
        '''
        # dynamics in MX format!
        model_dynamics = self._Ndi__trial.model.outputs['model_for_control']
        f_rot, g_rot_c1, g_rot_c2, g_rot_c3 = cas.vertsplit(model_dynamics,3)
        g_rot = cas.horzcat(g_rot_c1, g_rot_c2, g_rot_c3)
        '''

        return f_rot_funcs, g_rot_funcs

    def rotation_ndi_controller(self, x0_scaled, nu, parameters, architecture):
        u_ndi = []
        for kite in architecture.kite_nodes:
            F = self.F[kite](x0_scaled, parameters)
            G = self.G[kite](x0_scaled, parameters)
            u_ndi = cas.inv(G) @ (nu - F)
        return u_ndi
    def step(self, x0, plot_flag = False, u_ref = None):

        """ Compute NDI feedback control for given initial condition.
        """

        awelogger.logger.info("Compute NDI feedback...")


        # interpolator = self.__trial.nlp.Collocation.build_interpolator(
        # self.__trial.options['nlp'], self.__trial.optimization.V_opt)
        # T_ref = self.__trial.visualization.plot_dict['time_grids']['ip'][-1]
        # t_grid = np.linspace(0, self.__N*self.__ts, self.__N)
        # self.__t_grid = cas.vertcat(*list(map(lambda x: x % T_ref, t_grid))).full().squeeze()
        

        # update reference
        ref = self.get_reference(*self.__compute_time_grids(self.__index))
        nu = self.A_des @ (ref['x'][self.__index][6:9] - x0[6:9])
        dx_delta = self.rotation_ndi_controller(x0, nu, self.__pocp_trial.optimization.p_fix_num['theta0'], self.__trial.model.architecture)
        self.__index += 1
        # create a casadi function including ndi parameters -> then evaluate the function here numerically
        return cas.vertcat(cas.DM.zeros(18,1), dx_delta, cas.DM.zeros(3,1))

    def __compute_time_grids(self, index):
        """ Compute NLP time grids based in periodic index
        """

        Tref = self.__ref_dict['time_grids']['ip'][-1]
        t_grid = self.__t_grid_coll + index*self.__ts
        t_grid = cas.vertcat(*list(map(lambda x: x % Tref, t_grid))).full().squeeze()

        t_grid_x = self.__t_grid_x_coll + index*self.__ts
        t_grid_x = cas.vertcat(*list(map(lambda x: x % Tref, t_grid_x))).full().squeeze()

        t_grid_u = self.__t_grid_u + index*self.__ts
        t_grid_u = cas.vertcat(*list(map(lambda x: x % Tref, t_grid_u))).full().squeeze()

        return t_grid, t_grid_x, t_grid_u
    
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
                    elif (var_type == 'u') and self.__ndi_options['u_param'] == 'zoh':
                        ip_dict[var_type].append(self.__interpolator(t_grid_u, name, dim,var_type))
                    else:
                        ip_dict[var_type].append(self.__interpolator(t_grid, name, dim,var_type))
            if self.__ndi_options['ref_interpolator'] == 'poly':
                ip_dict[var_type] = cas.horzcat(*ip_dict[var_type]).T
            elif self.__ndi_options['ref_interpolator'] == 'spline':
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

                    if self.__ndi_options['u_param'] == 'zoh':
                        V_list.append(ip_dict['u'][:, counter_u])
                        V_list.append(np.zeros((self.__nx, 1)))
                        V_list.append(np.zeros((self.__nz, 1)))
                        counter_u += 1
                else:
                    for var_type in self.__var_list:
                        if var_type == 'x':
                            V_list.append(ip_dict[var_type][:,counter_x])
                            counter_x += 1
                        elif var_type == 'z' or (var_type == 'u' and self.__ndi_options['u_param']=='poly'):
                            V_list.append(ip_dict[var_type][:,counter])
                    counter += 1

        V_list.append(ip_dict['x'][:,counter_x])

        V_ref = V_ref(cas.vertcat(*V_list))

        return V_ref
    
    def __create_reference_interpolator(self):
        """ Create time-varying reference generator for tracking MPC based on interpolation of
            optimal periodic steady state.
        """

        # MPC time grid
        self.__t_grid_coll = self.__trial.nlp.time_grids['coll'](self.__N*self.__ts)
        self.__t_grid_coll = cas.reshape(self.__t_grid_coll.T, self.__t_grid_coll.numel(),1).full()
        self.__t_grid_x_coll = self.__trial.nlp.time_grids['x_coll'](self.__N*self.__ts)
        self.__t_grid_x_coll = cas.reshape(self.__t_grid_x_coll.T, self.__t_grid_x_coll.numel(),1).full()
        self.__t_grid_u = self.__trial.nlp.time_grids['u'](self.__N*self.__ts)
        self.__t_grid_u = cas.reshape(self.__t_grid_u.T, self.__t_grid_u.numel(),1).full()

        # interpolate steady state solution
        self.__ref_dict = self.__pocp_trial.visualization.plot_dict
        nlp_options = self.__pocp_trial.options['nlp']
        V_opt = self.__pocp_trial.optimization.V_opt
        if self.__ndi_options['ref_interpolator'] == 'poly':
            self.__interpolator = self.__pocp_trial.nlp.Collocation.build_interpolator(nlp_options, V_opt)
        elif self.__ndi_options['ref_interpolator'] == 'spline':
            self.__interpolator = self.__build_spline_interpolator(nlp_options, V_opt)

        return None

    def __build_spline_interpolator(self, nlp_options, V_opt):
        """ Build spline-based reference interpolating method.
        """

        variables_dict = self.__pocp_trial.model.variables_dict
        plot_dict = self.__pocp_trial.visualization.plot_dict
        cosmetics = self.__pocp_trial.options['visualization']['cosmetics']
        n_points = self.__t_grid_coll.shape[0]
        n_points_x = self.__t_grid_x_coll.shape[0]
        self.__spline_dict = {}

        for var_type in self.__var_list:
            self.__spline_dict[var_type] = {}
            for name in list(variables_dict[var_type].keys()):
                self.__spline_dict[var_type][name] = {}
                for j in range(variables_dict[var_type][name].shape[0]):
                    if var_type == 'x':
                        values, time_grid = viz_tools.merge_x_values(V_opt, name, j, plot_dict, cosmetics)
                        self.__spline_dict[var_type][name][j] = cas.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points_x)
                    elif var_type == 'z' or (var_type == 'u' and self.__ndi_options['u_param'] == 'poly'):
                        values, time_grid = viz_tools.merge_z_values(V_opt, var_type, name, j, plot_dict, cosmetics)
                        if all(v == 0 for v in values) or 'fict' in name:
                            self.__spline_dict[var_type][name][j] = cas.Function(name+str(j), [cas.SX.sym('t',n_points)], [np.zeros((1,n_points))])
                        else:
                            self.__spline_dict[var_type][name][j] = cas.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(n_points)
                    elif var_type == 'u' and self.__ndi_options['u_param'] == 'zoh':
                        control = V_opt['u',:,name,j]
                        values = viz_tools.sample_and_hold_controls(plot_dict['time_grids'], control)
                        time_grid = plot_dict['time_grids']['ip']
                        if all(v == 0 for v in values) or 'fict' in name:
                            self.__spline_dict[var_type][name][j] = cas.Function(name+str(j), [cas.SX.sym('t',self.__N)], [np.zeros((1,self.__N))])
                        else:
                            self.__spline_dict[var_type][name][j] = cas.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]]+values, {}).map(self.__N)

        def spline_interpolator(t_grid, name, j, var_type):
            """ Interpolate reference on specific time grid for specific variable.
            """

            values_ip = self.__spline_dict[var_type][name][j](t_grid)

            return values_ip

        return spline_interpolator

    
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


    @property
    def trial(self):
        """ awebox.Trial attribute containing model and OCP info.
        """
        return self.__trial

    @trial.setter
    def trial(self, value):
        awelogger.logger.info('Cannot set trial object.')

    @property
    def log(self):
        """ log attribute containing MPC info.
        """
        return self.__log

    @log.setter
    def log(self, value):
        awelogger.logger.info('Cannot set log object.')

    @property
    def t_grid_coll(self):
        """ Collocation grid time vector
        """
        return self.__t_grid_coll

    @t_grid_coll.setter
    def t_grid_coll(self, value):
        awelogger.logger.info('Cannot set t_grid_coll object.')

    @property
    def t_grid_u(self):
        """ ZOH control grid time vector
        """
        return self.__t_grid_u

    @t_grid_u.setter
    def t_grid_u(self, value):
        awelogger.logger.info('Cannot set t_grid_u object.')

    @property
    def t_grid_x_coll(self):
        """ Collocation grid time vector
        """
        return self.__t_grid_x_coll

    @t_grid_x_coll.setter
    def t_grid_x_coll(self, value):
        awelogger.logger.info('Cannot set t_grid_x_coll object.')

    @property
    def interpolator(self):
        """ interpolator
        """
        return self.__interpolator

    @interpolator.setter
    def interpolator(self, value):
        awelogger.logger.info('Cannot set interpolator object.')