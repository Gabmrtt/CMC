"""Robot parameters"""

import numpy as np
import farms_pylog as pylog


class RobotParameters(dict):
    """Robot parameters"""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, parameters):
        super(RobotParameters, self).__init__()

        # Initialise parameters
        self.n_body_joints = parameters.n_body_joints
        self.n_legs_joints = parameters.n_legs_joints
        self.n_joints = self.n_body_joints + self.n_legs_joints
        self.n_oscillators_body = 2*self.n_body_joints
        self.n_oscillators_legs = self.n_legs_joints
        self.n_oscillators = self.n_oscillators_body + self.n_oscillators_legs
        self.freqs = np.zeros(self.n_oscillators)
        self.coupling_weights = np.zeros([
            self.n_oscillators,
            self.n_oscillators
        ])
        self.phase_bias = np.zeros([self.n_oscillators, self.n_oscillators])
        self.rates = np.zeros(self.n_oscillators)
        self.nominal_amplitudes = np.zeros(self.n_oscillators)
        self.update(parameters)

    def update(self, parameters):
        """Update network from parameters"""
        self.set_frequencies(parameters)  # f_i
        self.set_coupling_weights(parameters)  # w_ij
        self.set_phase_bias(parameters)  # theta_i
        self.set_amplitudes_rate(parameters)  # a_i

        
        self.set_nominal_amplitudes(parameters)  # R_i

    def set_frequencies(self, parameters):
        """Set frequencies"""
        drive = parameters.drive
        for i in range(20):
            cv1 = 0.2
            cv0 = 0.3
            d_low = 1.0
            d_high = 5.0
            v_sat = 0.0
            
            if drive <= d_high and drive >= d_low:
                self.freqs[i] = cv1*drive + cv0
            else:
                self.freqs[i] = v_sat
                
        for j in range(20,24):
            cv1 = 0.2
            cv0 = 0.0
            d_low = 1.0
            d_high = 3.0
            v_sat = 0.0
            
            if drive <= d_high and drive >= d_low:
                self.freqs[j] = cv1*drive + cv0
            else:
                self.freqs[j] = v_sat



    def set_coupling_weights(self, parameters):
        
        for i in range(len(self.coupling_weights)-4):
            if i!=19 and i !=9:
                self.coupling_weights[i,i+1] = 10
            if i!=19 and i!=9 :
                self.coupling_weights[i+1,i] = 10
            for j in range(len(self.coupling_weights)-4):
                if np.abs(j-i) == 10:
                    self.coupling_weights[i,j] = 10
        for i in range(20,24):
            if i == 20:
                for j in range(5):
                    self.coupling_weights[j,i] = 30
                self.coupling_weights[i,21]=10
                self.coupling_weights[i,22]=10
            if i ==21:
                for j in range(5):
                    self.coupling_weights[j + 5,i] = 30
                self.coupling_weights[i,20]=10
                self.coupling_weights[i,23]=10
            if i ==22: 
                for j in range(5):
                    self.coupling_weights[j + 10,i] = 30
                self.coupling_weights[i,20]=10
                self.coupling_weights[i,23]=10
            if i ==23: 
                for j in range(5):
                    self.coupling_weights[j + 15,i] = 30
                self.coupling_weights[i,21]=10
                self.coupling_weights[i,22]=10
            
                    

    def set_phase_bias(self, parameters):
        """Set phase bias"""
        '''
        for i in range(len(self.phase_bias)-4):
            if i!=19 and i !=9:
                self.phase_bias[i,i+1]= np.pi * -2/8
            if i!=19 and i!=9 :
                self.phase_bias[i+1,i]= np.pi * 2/8
            for j in range(len(self.phase_bias)):
                if np.abs(j-i) == 10:
                    self.phase_bias[i,j]= np.pi
        for i in range(20,24):
            if i == 20:
                self.phase_bias[i,1]= np.pi
                self.phase_bias[i,21]=np.pi
                self.phase_bias[i,22]=np.pi
            if i ==21:
                self.phase_bias[i,6]=np.pi
                self.phase_bias[i,20]=np.pi
                self.phase_bias[i,23]=np.pi
            if i ==22: 
                self.phase_bias[i,11]=np.pi
                self.phase_bias[i,20]=np.pi
                self.phase_bias[i,23]=np.pi
            if i ==23: 
                self.phase_bias[i,16]=np.pi
                self.phase_bias[i,21]=np.pi
                self.phase_bias[i,22]=np.pi
                '''
        sub_mat1 = np.concatenate((np.diag(-2*np.pi/8*np.ones(self.n_body_joints-1),1) + np.diag(2*np.pi/8*np.ones(self.n_body_joints-1),-1), np.diag(np.pi*np.ones(self.n_body_joints))), axis= 1)
        sub_mat2 = np.concatenate(( np.diag(np.pi*np.ones(self.n_body_joints)), np.diag(-2*np.pi/8*np.ones(self.n_body_joints-1),1) + np.diag(2*np.pi/8*np.ones(self.n_body_joints-1),-1)), axis= 1)
        sub_mat3 = np.array([[0, np.pi, np.pi, 0], [np.pi, 0, 0, np.pi], [np.pi, 0, 0, np.pi], [0, np.pi, np.pi, 0]])
        sub_mat4 = np.concatenate(( np.tile([[np.pi, 0, 0, 0]], (5,1)), np.tile([[0, np.pi, 0, 0]], (5, 1)), np.tile([[0, 0, np.pi, 0]], (5, 1)), np.tile([[0, 0, 0, np.pi]], (5, 1)) ), axis=0)
        self.phase_bias = np.concatenate((np.concatenate((sub_mat1, sub_mat2, np.zeros([self.n_oscillators_legs,self.n_oscillators_body])), axis=0), np.concatenate((sub_mat4, sub_mat3), axis=0)), axis=1)

    def set_amplitudes_rate(self, parameters):
        
        self.amplitudes_rate = 20*np.ones([20])
        

    def set_nominal_amplitudes(self, parameters):
        
        drive = parameters.drive
        for i in range(20):
                cR1 = 0.065
                cR0 = 0.196
                d_low = 1.0
                d_high = 5.0
                R_sat = 0.0
                if drive <= d_high and drive >= d_low:
                
                    self.nominal_amplitudes[i] = cR1*drive + cR0
                else:
                    self.nominal_amplitudes[i] = R_sat
                
        for j in range(20,24):
                cR1 = 0.131
                cR0 = 0.131
                d_low = 1.0
                d_high = 3.0
                R_sat = 0.0
            
                if drive <= d_high and drive >= d_low:
                    self.nominal_amplitudes[j] = cR1*drive + cR0
                else:
                    self.nominal_amplitudes[j] = R_sat

