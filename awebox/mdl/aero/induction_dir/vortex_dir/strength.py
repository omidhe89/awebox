#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
'''
constraints to create the on-off switch on the vortex strength
to be referenced/used from ocp.constraints
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2020
'''

import casadi.tools as cas
import numpy as np
import awebox.tools.struct_operations as struct_op
import awebox.mdl.aero.induction_dir.vortex_dir.tools as tools
from awebox.logger.logger import Logger as awelogger
import awebox.tools.print_operations as print_op
import awebox.ocp.collocation as collocation
import awebox.ocp.var_struct as var_struct


######## the constraints : see opti.constraints

def get_cstr_in_constraints_format(options, g_list, g_bounds, V, Outputs, model):

    resi = get_strength_constraint_all(options, V, Outputs, model)

    g_list.append(resi)
    g_bounds = tools.append_bounds(g_bounds, resi)

    return g_list, g_bounds


######## the placeholders : see ocp.operation

def get_cstr_in_operation_format(options, variables, model):
    eqs_dict = {}
    constraint_list = []

    if 'collocation' not in options.keys():
        message = 'vortex model is not yet set up for any discretization ' \
                  'other than direct collocation'
        awelogger.logger.error(message)

    n_k = options['n_k']
    d = options['collocation']['d']
    scheme = options['collocation']['scheme']
    Collocation = collocation.Collocation(n_k, d, scheme)

    model_outputs = model.outputs
    V_mock = var_struct.setup_nlp_v(options, model, Collocation)

    entry_tuple = (cas.entry('coll_outputs', repeat=[n_k, d], struct=model_outputs))
    Outputs_mock = cas.struct_symMX([entry_tuple])

    resi_mock = get_strength_constraint_all(options, V_mock, Outputs_mock, model)
    try:
        resi = cas.DM.ones(resi_mock.shape)
    except:
        resi = []

    eq_name = 'vortex_strength'
    eqs_dict[eq_name] = resi
    



    constraint_list.append(resi)

    return eqs_dict, constraint_list


################ actually define the constriants

def get_strength_constraint_all(options, V, Outputs, model):

    n_k = options['n_k']
    d = options['collocation']['d']

    control_intervals = n_k + 1

    comparison_labels = options['induction']['comparison_labels']
    wake_nodes = options['induction']['vortex_wake_nodes']
    rings = wake_nodes - 1
    kite_nodes = model.architecture.kite_nodes

    Xdot = struct_op.construct_Xdot_struct(options, model.variables_dict)(0.)

    resi = []

    any_vor = any(label[:3] == 'vor' for label in comparison_labels)
    if any_vor:

        for kite in kite_nodes:
            for ring in range(rings):
                wake_node = ring

                for ndx in range(n_k):
                    for ddx in range(d):

                        variables = struct_op.get_variables_at_time(options, V, Xdot, model.variables, ndx, ddx)
                        wg_local = tools.get_ring_strength(options, variables, kite, ring)

                        index = control_intervals - wake_node
                        # wake_node = n_k - index

                        strength_scale = tools.get_strength_scale(options)

                        wg_ref = 1. * strength_scale
                        # if index >= ndx:
                        #     wg_ref = 0. * strength_scale
                        print_op.warn_about_temporary_funcationality_removal(location='v.strength')

                        resi_local = (wg_local - wg_ref)/strength_scale
                        resi = cas.vertcat(resi, resi_local)

    return resi
