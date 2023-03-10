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
import numpy as np
from casadi.tools import vertcat

def data_dict():

    data_dict = {}
    data_dict['name'] = 'megawes'

    data_dict['geometry'] = geometry() # kite geometry

    stab_derivs, aero_validity = aero()
    data_dict['stab_derivs'] = stab_derivs # stability derivatives
    data_dict['aero_validity'] = aero_validity

    # (optional: on-board battery model)
    coeff_min = np.array([0, -80*np.pi/180.0])
    coeff_max = np.array([2, 80*np.pi/180.0])
    data_dict['battery'] = battery_model_parameters(coeff_max, coeff_min)

    return data_dict

def geometry():

     # Values from: Eijkelhof, D.; Schmehl, R. Six-degrees-of-freedom simulation model for future multi-megawatt airborne wind energy systems. Renew. Energy 	   2022, 196, 137–150

    geometry = {}

    geometry['b_ref'] = 42.47  # [m]
    geometry['s_ref'] = 150.45  # [m^2]
    geometry['c_ref'] = geometry['s_ref'] / geometry['b_ref']  # [m] #todo:check with AVL model

    geometry['m_k'] = 6885.2  # [kg]

    geometry['ar'] = geometry['b_ref'] / geometry['c_ref'] #12.0
    geometry['j'] = np.array([[5.768e5, 0.0, 0.0],
                              [0.0, 0.8107e5, 0.0],
                              [0.47, 0.0, 6.5002e5]])

    geometry['length'] = geometry['b_ref']  # only for plotting
    geometry['height'] = geometry['b_ref'] / 5.  # only for plotting
    geometry['delta_max'] = np.array([20., 20., 20.]) * np.pi / 180.
    geometry['ddelta_max'] = np.array([2., 2., 2.])

    geometry['c_root'] = 4.46
    geometry['c_tip'] = 2.11

    geometry['fuselage'] = True
    geometry['wing'] = True
    geometry['tail'] = True
    geometry['wing_profile'] = None

    # tether attachment point
    geometry['r_tether'] = np.reshape([0, 0, 0], (3,1)) 

    return geometry

def battery_model_parameters(coeff_max, coeff_min):

    battery_model = {}

    # guessed values for battery model
    battery_model['flap_length'] = 0.2
    battery_model['flap_width'] = 0.1
    battery_model['max_flap_defl'] = 20.*(np.pi/180.)
    battery_model['min_flap_defl'] = -20.*(np.pi/180.)
    battery_model['c_dl'] = (battery_model['max_flap_defl'] - battery_model['min_flap_defl'])/(coeff_min[0] - coeff_max[0])
    battery_model['c_dphi'] = (battery_model['max_flap_defl'] - battery_model['min_flap_defl'])/(coeff_min[1] - coeff_max[1])
    battery_model['defl_lift_0'] = battery_model['min_flap_defl'] - battery_model['c_dl']*coeff_max[0]
    battery_model['defl_roll_0'] = battery_model['min_flap_defl'] - battery_model['c_dphi']*coeff_max[1]
    battery_model['voltage'] = 3.7
    battery_model['mAh'] = 5000.
    battery_model['charge'] = battery_model['mAh']*3600.*1e-3
    battery_model['number_of_cells'] = 15.
    battery_model['conversion_efficiency'] = 0.7
    battery_model['power_controller'] = 50.
    battery_model['power_electronics'] = 10.
    battery_model['charge_fraction'] = 1.

    return battery_model

def aero():
    # commented values are not currently supported, future implementation

    # Aerodynamic model:
    # A reference model for airborne wind energy systems for optimization and control
    # Article
    # March 2019 Renewable Energy
    # Elena Malz Jonas Koenemann S. Sieberling Sebastien Gros

    # MegAWES stability derivatives:
    # Stability derivatives of MegAWES aircraft from AVL analysis
    # performed by Jolan Wauters (Ghent University, 2023)

    # commented values are not currently supported, future implementation

    stab_derivs = {}
    stab_derivs['frame'] = {}
    stab_derivs['frame']['force'] = 'control'
    stab_derivs['frame']['moment'] = 'control'

    # Force coefficients (MegAWES)
    stab_derivs['CX'] = {}
    stab_derivs['CX']['0'] = [-0.054978] #1
    stab_derivs['CX']['alpha'] = [0.95827, 5.6521] #alpha,alpha2
    stab_derivs['CX']['q'] =  [-0.62364,3.9554,0.76064] #1,alpha,alpha2		
    stab_derivs['CX']['deltae'] = [-0.031786,0.2491,0.076416] #1,alpha,alpha2	

    stab_derivs['CY'] = {}
    stab_derivs['CY']['beta'] = [-0.25065, -0.077264,-0.25836	] #1,alpha,alpha2 	
    stab_derivs['CY']['p'] =  [-0.11406, -0.52471	, 0.029861	] 
    stab_derivs['CY']['r'] = [0.094734,0.029343,-0.053453] 
    stab_derivs['CY']['deltae'] = [-0.0072147, 0.034068	, 0.0032198	]  
    stab_derivs['CY']['deltar'] = [0.22266, 0.0099775, -0.21563 ] 
    
    stab_derivs['CZ'] = {}
    stab_derivs['CZ']['0'] = [-1.2669]
    stab_derivs['CZ']['alpha'] = [-6.3358, 0.17935] #alpha,alpha2
    stab_derivs['CZ']['q'] =  [-8.5019, -1.0139, 3.5051] 	
    stab_derivs['CZ']['deltae'] = [-0.56729, -0.0090401, 0.53686	] 		

    # Moment coefficients (MegAWES)
    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['beta'] = [0.033447,0.29221,-0.008944] 	
    stab_derivs['Cl']['p'] = [-0.62105, -0.013963,0.19592] 	
    stab_derivs['Cl']['r'] =  [ 0.29112, 0.74473,  -0.09468] 		
    stab_derivs['Cl']['deltaa'] = [-0.22285, 0.072142, 0.20538] 		
    stab_derivs['Cl']['deltar'] = [0,0,0]

    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['0'] = [0.046452] 
    stab_derivs['Cm']['alpha'] = [0.19869,0.34447] 
    stab_derivs['Cm']['q'] =  [-9.81, 0.013751, 4.7861] 
    stab_derivs['Cm']['deltae'] =  [-1.435 ,-0.0079607, 1.3813]

    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['beta'] =  [0.043461 ,0.0014151, -0.034616] 
    stab_derivs['Cn']['p'] = [-0.10036 ,-1.0669, -0.016814] 	
    stab_derivs['Cn']['r'] =  [-0.043558 ,0.030031, -0.12246] 
    stab_derivs['Cn']['deltaa'] = [0.0066928 ,-0.086616, 0.0005028] 
    stab_derivs['Cn']['deltar'] = [-0.04988 ,-0.00096109, 0.04822] 

    # Aero validity (MegAWES)
    aero_validity = {}
    aero_validity['alpha_max_deg'] = 4.2
    aero_validity['alpha_min_deg'] = -14.
    aero_validity['beta_max_deg'] = 10.
    aero_validity['beta_min_deg'] = -10.


    return stab_derivs, aero_validity
