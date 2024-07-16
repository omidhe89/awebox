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
    stab_derivs['CX']['0'] = [-0.0460] #1  COR:  -0.0460-0.085[NOTCOR]
    stab_derivs['CX']['alpha'] = [0.5329, 3.6178] #alpha,alpha2
    stab_derivs['CX']['q'] =  [-0.1689,3.1142,-0.3229 ] #1,alpha,alpha2     
    stab_derivs['CX']['deltae'] = [-0.0203, 0.2281, 0.0541] #1,alpha,alpha2 

    stab_derivs['CY'] = {}
    stab_derivs['CY']['beta'] = [-0.2056,-0.1529 , -0.3609] #1,alpha,alpha2     
    stab_derivs['CY']['p'] = [0.0588,   0.3069, -0.0109 ] 
    stab_derivs['CY']['r'] = [0.0869,   0.0271, -0.0541]
    stab_derivs['CY']['deltaa'] = [0.0064,  -0.0365,    -0.0022]  
    stab_derivs['CY']['deltar'] = [0.1801,  0.0196, -0.1724] 

    stab_derivs['CZ'] = {}
    stab_derivs['CZ']['0'] = [-0.8781   ]
    stab_derivs['CZ']['alpha'] = [-4.7042, 0.0335 ] #alpha,alpha2
    stab_derivs['CZ']['q'] =  [-5.9365, -0.7263,    2.4422 ]    
    stab_derivs['CZ']['deltae'] = [-0.4867, -0.0070,    0.4642]         

    # Moment coefficients (MegAWES)
    stab_derivs['Cl'] = {}
    stab_derivs['Cl']['beta'] = [-0.0101,-0.1834,0.0023]    
    stab_derivs['Cl']['p'] = [-0.4888,  -0.0270 ,0.0920]    
    stab_derivs['Cl']['r'] = [0.1966,0.5629,-0.0498 ]       
    stab_derivs['Cl']['deltaa'] = [-0.1972,0.0574,0.1674]       
    stab_derivs['Cl']['deltar'] = [0.0077,-0.0091,-0.0092]
                                             
    stab_derivs['Cm'] = {}
    stab_derivs['Cm']['0'] = [-0.0650]  #COR:  -0.0650 + 0.0934
    stab_derivs['Cm']['alpha'] = [-0.3306,0.2245 ]
    stab_derivs['Cm']['q'] = [-7.7531,-0.0030,3.8925] 
    stab_derivs['Cm']['deltae'] = [-1.1885,-0.0007,1.1612]
                                                                                      
    stab_derivs['Cn'] = {}
    stab_derivs['Cn']['beta'] = [0.0385,0.0001,   -0.0441] 
    stab_derivs['Cn']['p'] = [-0.0597,-0.7602,0.0691] 
    stab_derivs['Cn']['r'] = [-0.0372,-0.0291,-0.2164] 
    stab_derivs['Cn']['deltaa'] = [0.0054,-0.0425,0.0354]
    stab_derivs['Cn']['deltar'] = [-0.0404,-0.0031,0.0385]

    #! # Force coefficients (MegAWES)
    #! stab_derivs['CX'] = {}
    #! stab_derivs['CX']['0'] = [-0.0411] #1
    #! stab_derivs['CX']['alpha'] = [1.0081, 5.9011] #alpha,alpha2
    #! # stab_derivs['CX']['beta'] =  [0., 0., 0.] #1,alpha,alpha2		
    #! # stab_derivs['CX']['p'] =  [0., 0., 0.] #1,alpha,alpha2		
    #! stab_derivs['CX']['q'] =  [-0.5274, 3.8884, 0.4307] #1,alpha,alpha2		
    #! # stab_derivs['CX']['r'] =  [0., 0., 0.] #1,alpha,alpha2		
    #! # stab_derivs['CX']['deltaa'] =  [0., 0., 0.] #1,alpha,alpha2		
    #! stab_derivs['CX']['deltae'] = [-0.0311, 0.2359, 0.0710] #1,alpha,alpha2	
    #! # stab_derivs['CX']['deltar'] =  [0., 0., 0.] #1,alpha,alpha2		

    #! stab_derivs['CY'] = {}
    #! # stab_derivs['CY']['0'] = [0.] #1
    #! # stab_derivs['CY']['alpha'] = [0., 0.] #alpha,alpha2
    #! stab_derivs['CY']['beta'] = [-0.2279, -0.0184, 0.1062] #1,alpha,alpha2 	
    #! stab_derivs['CY']['p'] = [-0.1069, -0.4699, 0.0327] 
    #! # stab_derivs['CY']['q'] = [0., 0., 0.] 
    #! stab_derivs['CY']['r'] = [0.0909, 0.0378, -0.0435]
    #! stab_derivs['CY']['deltaa'] = [-0.0070, 0.0368, 0.0024]  
    #! # stab_derivs['CY']['deltae'] = [0., 0., 0.]  
    #! stab_derivs['CY']['deltar'] = [0.2115, 0.0197, -0.2032] 
    #! 
    #! stab_derivs['CZ'] = {}
    #! stab_derivs['CZ']['0'] = [-1.2669]
    #! stab_derivs['CZ']['alpha'] = [-6.3358, 0.17935] #alpha,alpha2
    #! # stab_derivs['CZ']['beta'] = [0., 0., 0.] 
    #! # stab_derivs['CZ']['p'] = [0., 0., 0.] 
    #! stab_derivs['CZ']['q'] =  [-8.5019, -1.0139, 3.5051] 	
    #! # stab_derivs['CZ']['r'] = [0., 0., 0.] 
    #! # stab_derivs['CZ']['deltaa'] = [0., 0., 0.] 
    #! stab_derivs['CZ']['deltae'] = [-0.5384, -0.0088, 0.5239] 		
    #! # stab_derivs['CZ']['deltar'] = [0., 0., 0.] 

    # Moment coefficients (MegAWES)

    #! stab_derivs['Cl'] = {}
    #! # stab_derivs['Cl']['0'] = [0.] #1
    #! # stab_derivs['Cl']['alpha'] = [0., 0.] #alpha,alpha2
    #! stab_derivs['Cl']['beta'] = [1.5181, 13.2144, 0.1427] 	
    #! stab_derivs['Cl']['p'] = [0.5735, 0.0004, -0.2849] 	
    #! # stab_derivs['Cl']['q'] = [0., 0., 0.] 
    #! stab_derivs['Cl']['r'] = [0.2764, 0.6898, -0.1074] 		
    #! stab_derivs['Cl']['deltaa'] = [0.2303, -0.0776, -0.2370] 		
    #! # stab_derivs['Cl']['deltae'] = [0., 0., 0.] 
    #! stab_derivs['Cl']['deltar'] = [0.0070, -0.0108, -0.0086]

    #! stab_derivs['Cm'] = {}
    #! stab_derivs['Cm']['0'] = [0.0808] 
    #! stab_derivs['Cm']['alpha'] = [0.3672, 0.3390]
    #! # stab_derivs['Cm']['beta'] = [0., 0., 0.] 
    #! # stab_derivs['Cm']['p'] = [0., 0., 0.] 
    #! stab_derivs['Cm']['q'] = [-8.0606, 0.0349, 4.0070] 
    #! # stab_derivs['Cm']['r'] = [0., 0., 0.] 
    #! # stab_derivs['Cm']['deltaa'] = [0., 0., 0.]
    #! stab_derivs['Cm']['deltae'] = [-1.2832, -0.0098, 1.2499]
    #! # stab_derivs['Cm']['deltar'] = [0., 0., 0.]

    #! stab_derivs['Cn'] = {}
    #! # stab_derivs['Cn']['0'] = [0.] 
    #! # stab_derivs['Cn']['alpha'] = [0., 0.] 
    #! stab_derivs['Cn']['beta'] = [0.0423, 0.0109, -0.0215] 
    #! stab_derivs['Cn']['p'] = [0.1019, 1.0564, -0.0048] 
    #! # stab_derivs['Cn']['q'] = [0., 0., 0.] 
    #! stab_derivs['Cn']['r'] = [-0.0358, 0.0759, 0.0211] 
    #! stab_derivs['Cn']['deltaa'] = [-0.0055, 0.1119, -0.0104]
    #! # stab_derivs['Cn']['deltae'] = [0., 0., 0.]
    #! stab_derivs['Cn']['deltar'] = [-0.0476, -0.0033, 0.0458]

    # Aero validity (MegAWES)
    aero_validity = {}
    aero_validity['alpha_max_deg'] = +5.  #4.2
    aero_validity['alpha_min_deg'] = -15. #-14.5
    aero_validity['beta_max_deg'] = 10.
    aero_validity['beta_min_deg'] = -10.


    return stab_derivs, aero_validity
