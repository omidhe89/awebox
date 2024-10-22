import awebox as awe
import awebox.viz.tools as viz_tools
import casadi as ct
from awebox.logger.logger import Logger as awelogger
import matplotlib.pyplot as plt
import numpy as np
import copy
import awebox.mdl.architecture as archi

class OC():
    
    def __init__(self, sim_options, ts, trial):
        """ constructor"""

        self.__pocp_trial = trial
        self.__N = sim_options['N']
        self.__oc_options = sim_options
        self.__ts = ts
        # store model data
        self.__var_list = ['x', 'z', 'u']
        self.__nx = trial.model.variables['x'].shape[0]
        self.__nu = trial.model.variables['u'].shape[0]
        self.__nz = trial.model.variables['z'].shape[0]

         # create mpc trial
        options = copy.deepcopy(trial.options_seed)
        options['nlp.n_k'] = self.__N
        options['visualization.cosmetics.plot_ref'] = True
        fixed_params = {}
        for name in list(self.__pocp_trial.model.variables_dict['theta'].keys()):
            if name != 't_f':
                fixed_params[name] = self.__pocp_trial.optimization.V_final['theta',name].full()
        fixed_params['t_f'] = self.__N*self.__ts
        options['user_options.trajectory.fixed_params'] = fixed_params
        self.__trial = awe.Trial(seed = options)
        self.__build_trial()
        self.__create_reference_interpolator()
        self.__w0 = self.get_reference(*self.__compute_time_grids(0))

    def __build_trial(self):
        """ Build options, model, formulation and nlp of mpc trial.
        """

        awelogger.logger.info("Building trial for open loop flight...")

        # build
        import awebox.mdl.architecture as archi
        architecture = archi.Architecture(self.__trial.options['user_options']['system_model']['architecture'])
        self.__trial.options.build(architecture)
        self.__trial.model.build(self.__trial.options['model'], architecture)
        self.__trial.formulation.build(self.__trial.options['formulation'], self.__trial.model)
        self.__trial.nlp.build(self.__trial.options['nlp'], self.__trial.model, self.__trial.formulation)
        self.__trial.visualization.build(self.__trial.model, self.__trial.nlp, 'Open loop', self.__trial.options)

    def __create_reference_interpolator(self):
        """ Create time-varying reference generator for tracking on interpolation of
            optimal periodic steady state.
        """

        # OC time grid

        self.__t_grid_coll = self.__trial.nlp.time_grids['coll'](self.__N*self.__ts)
        self.__t_grid_coll = ct.reshape(self.__t_grid_coll.T, self.__t_grid_coll.numel(),1).full()
        self.__t_grid_x_coll = self.__trial.nlp.time_grids['x_coll'](self.__N*self.__ts)
        self.__t_grid_x_coll = ct.reshape(self.__t_grid_x_coll.T, self.__t_grid_x_coll.numel(),1).full()
        self.__t_grid_u = self.__trial.nlp.time_grids['u'](self.__N*self.__ts)
        self.__t_grid_u = ct.reshape(self.__t_grid_u.T, self.__t_grid_u.numel(),1).full()

        # interpolate steady state solution
        self.__ref_dict = self.__pocp_trial.visualization.plot_dict
        nlp_options = self.__pocp_trial.options['nlp']
        V_opt = self.__pocp_trial.optimization.V_opt
        if self.__oc_options['ref_interpolator'] == 'poly':
            self.__interpolator = self.__pocp_trial.nlp.Collocation.build_interpolator(nlp_options, V_opt)
        elif self.__oc_options['ref_interpolator'] == 'spline':
            self.__interpolator = self.__build_spline_interpolator(nlp_options, V_opt)
        
        
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
                        self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]] + values, {}).map(n_points_x)
                    elif var_type == 'z' or (var_type == 'u' and self.__oc_options['u_param'] == 'poly'):
                        values, time_grid = viz_tools.merge_z_values(V_opt, var_type, name, j, plot_dict, cosmetics)
                        if all(v == 0 for v in values) or 'fict' in name:
                            self.__spline_dict[var_type][name][j] = ct.Function(name+str(j), [ct.SX.sym('t', n_points)], [np.zeros((1,n_points))])
                        else:
                            self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [[0]+time_grid], [values[-1]] + values, {}).map(n_points)
                    elif var_type == 'u' and self.__oc_options['u_param'] == 'zoh':   
                        # values = V_opt['u',:,name,j]
                        # float_values = [float(value.full()) for value in values]
                        control = V_opt['u',:,name,j]
                        values = viz_tools.sample_and_hold_controls(plot_dict['time_grids'], control)
                        time_grid = plot_dict['time_grids']['ip']
                        if all(v == 0 for v in values) or 'fict' in name:
                            self.__spline_dict[var_type][name][j] = ct.Function(name+str(j), [ct.SX.sym('t',self.__N)], [np.zeros((1,self.__N))])
                        else:
                            self.__spline_dict[var_type][name][j] = ct.interpolant(name+str(j), 'bspline', [time_grid], values, {}).map(self.__N)
        
        def spline_interpolator(t_grid, name, j, var_type):
            """ Interpolate reference on specific time grid for specific variable.
            """

            values_ip = self.__spline_dict[var_type][name][j](t_grid)

            return values_ip

        return spline_interpolator


    def __compute_time_grids(self, index):
            """ Compute NLP time grids based in periodic index
            """
            # d = self.pocp_trial.options['nlp']['collocation']['d']
            Tref = self.__ref_dict['time_grids']['ip'][-1]
            t_grid = self.__t_grid_coll
            t_grid = ct.vertcat(*list(map(lambda x: x % Tref, t_grid))).full().squeeze()
            t_grid_x = self.__t_grid_x_coll
            t_grid_x = ct.vertcat(*list(map(lambda x: x % Tref, t_grid_x))).full().squeeze()
            t_grid_u = self.__t_grid_u
            t_grid_u = ct.vertcat(*list(map(lambda x: x % Tref, t_grid_u))).full().squeeze()
            return t_grid, t_grid_x, t_grid_u


    def get_reference(self, t_grid, t_grid_x, t_grid_u):
        """ Interpolate reference on NLP time grids.
        """
        ip_dict = {}
        V_ref = self.__trial.nlp.V(0.0)
        for var_type in self.__var_list:
            ip_dict[var_type] = []
            for name in list(self.__pocp_trial.model.variables_dict[var_type].keys()):
                for dim in range(self.__pocp_trial.model.variables_dict[var_type][name].shape[0]):
                    if var_type == 'x':
                        ip_dict[var_type].append(self.__interpolator(t_grid_x, name, dim, var_type))
                    elif (var_type == 'u') and self.__oc_options['u_param'] == 'zoh':
                        ip_dict[var_type].append(self.__interpolator(t_grid_u, name, dim, var_type))
                    else:
                        ip_dict[var_type].append(self.__interpolator(t_grid, name, dim, var_type))
            if self.__oc_options['ref_interpolator'] == 'poly':
                ip_dict[var_type] = ct.horzcat(*ip_dict[var_type]).T
            elif self.__oc_options['ref_interpolator'] == 'spline':
                ip_dict[var_type] = ct.vertcat(*ip_dict[var_type])

        counter = 0
        counter_x = 0
        counter_u = 0
        V_list = []

        for name in self.__pocp_trial.model.variables_dict['theta'].keys():
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

                    if self.__oc_options['u_param'] == 'zoh':
                        V_list.append(ip_dict['u'][:, counter_u])
                        V_list.append(np.zeros((self.__nx, 1)))
                        V_list.append(np.zeros((self.__nz, 1)))
                        counter_u += 1
                else:
                    for var_type in self.__var_list:
                        if var_type == 'x':
                            V_list.append(ip_dict[var_type][:,counter_x])
                            counter_x += 1
                        elif var_type == 'z' or (var_type == 'u' and self.__oc_options['u_param']=='poly'):
                            V_list.append(ip_dict[var_type][:,counter])
                    counter += 1

        V_list.append(ip_dict['x'][:,counter_x])

        V_ref = V_ref(ct.vertcat(*V_list))

        return V_ref

    def step(self, i):
        return self.__w0['u'][i]    
    @property
    def pocp_trial(self):
        """ awebox.Trial attribute containing model and OCP info.
        """
        return self.__pocp_trial

    @pocp_trial.setter
    def trial(self, value):
        awelogger.logger.info('Cannot set trial object.')


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
