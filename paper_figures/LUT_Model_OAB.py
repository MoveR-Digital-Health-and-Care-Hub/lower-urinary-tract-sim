""" 
Lower Urinary Tract Model Simulation
====================================
This module contains the implementation of the bladder, sphincter, and kidney dynamics simulation, 
using normalised neural signals to predict pressure and volume. This version is modified to include
a greater parasympathetic input and a lower sympathetic input to simulate overactive bladder.
""" 
__author__ = "Elliot Lister"
__version__ = "1.1.0" 
__license__ = "MIT"

# Import Libraries
import pandas as pd
import numpy as np 
from scipy import optimize


# LUT Model
class LUT:
    def __init__(self):
        self.V_B = 0
        self.f_aD_s = 0
        self.f_aS_s = 0
        self.r_U = 0
        self.Q = 0
        self.Q_in = 0
        self.p_D = 0
        self.p_S = 0
        self.voiding = False
        self.w_s_s = 0
        self.w_i_s = 0
        self.w_e_s = 0
        self.bladder_args = {
        # Constants
        'alpha' : 2,
        'A_BN' : 7.07 * 10 ** -4,
        'A_C' : 8.0 * 10 ** -5,
        'A_muscleS' : 8.0 * 10 ** -6,
        'A_nomD' : 2.78 * 10 ** -4,
        'A_tissueS' : 4.0 * 10 ** -5,
        'C_l' : 5.0 * 10 ** 3,
        'C_u' : 2.0 * 10 ** 2,
        'C_Qin' : 5.0 * 10 ** -8,
        'C_p' : 1.5,
        'dr' : 1.0 * 10 ** -4,
        'h_D' : 8.19 * 10 ** -4,
        'h_nomS' : 2.65 * 10 ** -4,
        'k' : 0.3,
        'l_optD' : 4.68 * 10 ** -6,
        'l_optS' : 2.23 * 10 ** -6,
        'p_0S' : 100,
        'p_nomS' : 6.0 * 10 ** 3,
        'p_theta' : 3.0 * 10 ** 3,
        'rho' : 1.0 * 10 ** 3,
        'r_BN' : 1.5 * 10 ** -2,
        'r_optD' : 5.4 * 10 ** -2,
        'r_optS' : 4.8 * 10 ** -3,
        'r_0D' : 2.7 * 10 ** -2,
        'r_0S' : 4.8 * 10 ** -3,
        'R_1' : 3.0 * 10 ** 8,
        'R_2' : 2.4 * 10 ** 8,
        'sigma_isoD' : 4.0 * 10 ** 5,
        'sigma_isoS' : 2.0 * 10 ** 5,
        'tau_D' : 1.0,
        'tau_S' : 0.2,
        'u_maxD' : 0.2,
        'u_maxS' : 1.0,
        'V_muscleD' : 3.0 * 10 ** -5,
        'V_tissueD' : 2.0 * 10 ** -5,
        'tissue_pressed_in_BN' : False
        }

        self.bladder_args['max_V_B'] = 5 * 10 ** -4
        self.bladder_args['voiding_threshold'] = 1 * self.bladder_args['max_V_B']
        self.bladder_args['pressure_threshold'] = 5.3669 * 10 ** 3
        self.bladder_args['neuron_threshold'] = 0.50*self.bladder_args['voiding_threshold']
        self.bladder_args['filling_phase_I'] = 0.04 * self.bladder_args['max_V_B']
        self.bladder_args['filling_phase_II'] = 0.75 * self.bladder_args['max_V_B']
        self.bladder_args['filling_phase_III'] = 0.9 * self.bladder_args['max_V_B']

        self.n = 1
        self.noise = 1
        self.t = 0
    def get_p_S(self, A_U, f_aS_s, r_U):
        r_outS = ((1 / np.pi) * (A_U + self.bladder_args['A_tissueS'] + self.bladder_args['A_muscleS']) ) ** (1/2) 
        r_inS = ((1 / np.pi) * (A_U + self.bladder_args['A_tissueS']) ) ** (1/2) 
        r_S = (r_outS + r_inS) / 2 
        h_S = r_outS - r_inS

        dru = (r_U - self.r_U) / self.bladder_args['dT']
        u_S = - r_U / (2 * self.bladder_args['r_optS']) * (1 / r_outS + 1 / r_inS) * dru
        u_S_s = u_S / self.bladder_args['u_maxS']

        l_S = (self.bladder_args['l_optS'] / self.bladder_args['r_optS']) * r_S  
        sigma_nom_actS = f_aS_s * self.bladder_args['sigma_isoS'] * self.sigma_u_s(u_S_s) * self.sigma_lS_s(l_S)
        sigma_actS = (self.bladder_args['h_nomS'] / h_S) * sigma_nom_actS

        p_actS = sigma_actS * np.log(r_outS / r_inS)
        p_pasS = self.bladder_args['p_0S'] * (self.bladder_args['p_nomS'] / self.bladder_args['p_0S']) ** (r_U / self.bladder_args['r_0S'])
        p_S = p_actS + p_pasS
        return p_S

    def get_Q2(self, A_U, A_T, p_S):
        RA2 = self.bladder_args['R_1'] * A_U + (self.bladder_args['R_2'] / self.bladder_args['A_C']) * (A_U ** 2)
        Q2 = (p_S * A_U ** 2) / ((self.bladder_args['rho'] / 2) * (1 - (A_U ** 2 / A_T ** 2)) + RA2)
        return Q2

    def get_Qin(self, t):
        """
        This function generates a stochastic inflow to the bladder.
        """
        # Stochastic inflow
        ## Parameters
        a = 0.00625
        b = 2 * np.pi / 24
        c = 0.017
        p = 1/(60*60)
        o = 8 * 2 * np.pi / 24 * 1/p
        Q = a * np.sin(p * (b * t - o)) + c # Stochastic inflow in ml/s

        # Noise
        ## Noise is randomised every minute of operation to prevent overly smooth noise when dt is small
        if t % 60 == 0:
            self.n = np.random.uniform(-self.noise, self.noise)
        Q = (Q * 10 ** -6) # Scale inflow from ml/s to m^3/s
        Q += self.n * self.bladder_args['C_Qin']
        Q = max(Q, 0)
        return Q

    def f_0(self, V_B, f_aD_s, f_aS_s, r_U):
        A_U = np.pi * r_U ** 2
        A_T = np.pi * (r_U + self.bladder_args['dr']) ** 2
        A_BN = np.pi * self.bladder_args['r_BN'] ** 2
        
        r_inD = ((3 / (4 * np.pi)) * (V_B + self.bladder_args['V_tissueD']) ) ** (1/3)
        A_inD = 4 * np.pi * r_inD ** 2

        p_S = self.get_p_S(A_U, f_aS_s, r_U)

        Q2 = self.get_Q2(A_U, A_T, p_S)
        p_D = self.get_p_D(V_B, f_aD_s, np.sqrt(Q2))

        p_BN = p_D - ((self.bladder_args['rho'] * Q2) / 2) * (1 / A_BN ** 2 - 1 / A_inD ** 2)
        r_B = (3 * V_B / (4 * np.pi)) ** (1/3)
        A_B = 4 * np.pi * r_B ** 2
        dp = self.bladder_args['C_p'] * p_BN * ((A_BN - A_B) / A_BN) ** 2 if self.bladder_args['tissue_pressed_in_BN'] else 0

        p_T = p_D - (self.bladder_args['rho'] * Q2) / (2 * A_T ** 2) - dp
        return p_T - p_S

    def f1(self, V_B, f_aD_s, Q):
        self.p_D = self.get_p_D(V_B, f_aD_s, Q)
        return self.Q_in - Q 

    def f2(self, f_aD_s, w_e_s, w_i_s):
        return 1 / self.bladder_args['tau_D'] * (w_e_s- f_aD_s - w_i_s * f_aD_s)

    def f3(self, f_aS_s, w_s_s):
        return (1 / self.bladder_args['tau_S']) * (w_s_s - f_aS_s)

    def fmap(self, V_B, f_aD_s, f_aS_s):
        try:
            r_U = optimize.bisect(lambda r_U: self.f_0(V_B, f_aD_s, f_aS_s, r_U), 0, 5 * 10 ** -3)
        except ValueError:
            r_U = 0
        return r_U
    
    def sigma_u_s(self, u_s):
        if u_s < 0:
            sigma = 1.8 - (0.8 * (1 + u_s))/(1 - 7.56 * u_s / self.bladder_args['k'])
        elif u_s == 0:
            sigma = 1
        else:
            sigma = (1 - u_s)/(1 + (u_s / self.bladder_args['k']))
        return sigma

    def sigma_upasD(self, u_D_s):
        return self.bladder_args['C_u'] * u_D_s

    def sigma_lD_s(self, l_D):
        l_D_s = l_D / self.bladder_args['l_optD']
        if l_D_s <= 0.35:
            sigma = 0
        elif l_D_s <= 0.45:
            sigma = 5.5 * l_D_s - 1.925
        elif l_D_s <= 1.1:
            sigma = 0.643 * l_D_s + 0.293
        elif l_D_s <= 1.4:
            sigma = -3.33333 * l_D_s + 4.66667
        else:
            sigma = 0
        return sigma

    def line_eq(self, x1, y1, x2, y2):
        "Find equation of line between two (x,y) points"
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        return m, c

    def sigma_lS_s(self, l_S):
        l_S_s = l_S / self.bladder_args['l_optS']
        lower_l_S, upper_l_S = 1.73 * 10 ** -6, 2.89 * 10 ** -6
        lower_l_S_s, upper_l_S_s = lower_l_S / self.bladder_args['l_optS'], upper_l_S / self.bladder_args['l_optS']
        points = [(0.55, 0), (lower_l_S_s, 0.8), (1, 1), (1.1, 1), (1.75, 0)]
        for i in range(len(points) - 1):
            if l_S_s >= points[i][0] and l_S_s < points[i + 1][0]:
                m, c = self.line_eq(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                sigma = m * l_S_s + c
                return sigma
        return 0

    def get_p_D(self, V_B, f_aD_s, Q):
        r_outD = ((3 / (4 * np.pi)) * (V_B + self.bladder_args['V_tissueD'] + self.bladder_args['V_muscleD']) ) ** (1/3)
        r_inD = ((3 / (4 * np.pi)) * (V_B + self.bladder_args['V_tissueD']) ) ** (1/3)
        r_D = (r_outD + r_inD) / 2

        u_D = (Q / (8 * np.pi * self.bladder_args['r_optD'])) * (1 / r_outD ** 2 + 1 / r_inD ** 2)
        u_D_s = u_D / self.bladder_args['u_maxD']
        sigma_uD_s = self.sigma_u_s(u_D_s)

        l_D = (self.bladder_args['l_optD'] / self.bladder_args['r_optD']) * r_D

        sigma_lpasD = 0 if r_D < self.bladder_args['r_0D'] else self.bladder_args['C_l'] * ((r_D - self.bladder_args['r_0D']) / self.bladder_args['r_0D']) ** self.bladder_args['alpha']
        
        sigma_nomD = f_aD_s * self.bladder_args['sigma_isoD'] * sigma_uD_s * self.sigma_lD_s(l_D) + sigma_lpasD + self.sigma_upasD(u_D_s)
        A_D = np.pi * (r_outD ** 2 - r_inD ** 2)
        sigma_D = (self.bladder_args['A_nomD'] / A_D) * sigma_nomD
        p_D = sigma_D * np.log(r_outD / r_inD)
        return p_D

    def get_Q(self, f_aS_s, r_U):
        A_T = np.pi * (r_U + self.bladder_args['dr']) ** 2
        A_U = np.pi * r_U ** 2
        p_S = self.get_p_S(A_U, f_aS_s, r_U)
        self.p_S = p_S
        Q2 = self.get_Q2(A_U, A_T, p_S)
        return np.sqrt(Q2)

    def update_sympathetic_input(self):
        if self.voiding:
            w = 0
        else:
            # Gaussian
            fullness = self.V_B / self.bladder_args['max_V_B']
            a = 0.4  # Reduced magnitude
            b = 0.9  # Shifted right
            c = 0.1  # Wider curve
            d = 0.05 # lower minimum.
            s = a * np.exp(-((fullness - b) ** 2) / (2 * c ** 2)) + d
            w = s
        return max(min(w, 1), 0)
    
    def update_parasympathetic_input(self):
        if self.voiding:
            w = 0.8
        else:
            # Gaussian
            fullness = self.V_B / self.bladder_args['max_V_B']
            a = 1.2  # Increased magnitude
            b = 0.8  # Shifted left
            c = 0.1  # Wider curve
            d = 0.1 # higher minimum.
            linear_comp = fullness * 0.25
            s = a * np.exp(-((fullness - b) ** 2) / (2 * c ** 2)) + d + linear_comp
            w = s
        return max(min(w, 1), 0)

    def update_somatic_input(self):
        if self.voiding:
            w = 0.05
        else:
            # Gaussian
            a = 0.55
            b = 0.95
            c = 0.12
            d = 0.05
            s = a * (1 - (np.exp(-((self.r_U - b) ** 2) / (2 * c ** 2)) + d))
            w = s
        return max(min(w, 1), 0)
        
    def is_voiding(self):
        if self.voiding and self.V_B < 1 * 10 ** -8:
            return False # Reset voiding if bladder is empty
        elif self.trigger_metric == 'pressure' and self.p_D >= self.bladder_args['pressure_threshold']:
            return True # If pressure exceeds threshold, voiding
        elif self.V_B >= self.bladder_args['voiding_threshold']:
            return True # If volume exceeds threshold, voiding
        else:
            return self.voiding # Otherwise, maintain voiding state

    def process_neural_input(self, maxTime, dT, noise=1, V_unit='m^3', p_unit='Pa', verbose=False, seed=None, trigger_metric='pressure'):
        if seed is not None:
            np.random.seed(seed)
        self.trigger_metric = trigger_metric # Set trigger metric for voiding - pressure or volume
        self.Q_in = self.bladder_args['C_Qin']
        self.noise = noise
        datadict = [{'V_B': self.V_B, 'f_aD_s': self.f_aD_s, 'f_aS_s': self.f_aS_s, 'r_U': self.r_U, 'Q': self.Q, 'p_D': self.p_D, 'p_S': self.p_S, 'Q_in': self.Q_in}]
        self.bladder_args['dT'] = dT
        
        ts = np.arange(0, maxTime, dT)

        self.progress = 0

        for t in ts:
            self.t = t
            self.voiding = self.is_voiding()

            # Update neural inputs
            d_w_e_s = self.update_parasympathetic_input()
            d_w_i_s = self.update_sympathetic_input()
            d_w_s_s = self.update_somatic_input()

            self.w_e_s = d_w_e_s
            self.w_i_s = d_w_i_s
            self.w_s_s = d_w_s_s

            # Update state variables
            self.f_aD_s += dT * self.f2(self.f_aD_s, self.w_e_s, self.w_i_s)
            self.f_aS_s += dT * self.f3(self.f_aS_s, self.w_s_s)
            self.Q_in = self.get_Qin(t)
            self.V_B += dT * self.f1(self.V_B, self.f_aD_s, self.Q)
            self.V_B = min(max(self.V_B, 0), self.bladder_args['max_V_B'])
            self.r_U = self.fmap(self.V_B, self.f_aD_s, self.f_aS_s)
            self.Q = self.get_Q(self.f_aS_s, self.r_U)

            # Append data to dictionary at each time step
            datadict.append({'t': t, 'V_B': self.V_B, 'f_aD_s': self.f_aD_s, 'f_aS_s': self.f_aS_s, 'r_U': self.r_U, 'Q': self.Q, 'p_D': self.p_D, 'p_S': self.p_S, 'Q_in': self.Q_in, 'w_e_s': self.w_e_s, 'w_i_s': self.w_i_s, 'w_s_s': self.w_s_s, 'voiding': self.voiding})
            
            if verbose:
                self.progress = t / maxTime
                print(f"Progress: {self.progress * 100:.2f}%", end='\r')

        # Build DataFrame to export results
        df = pd.DataFrame(datadict)

        # Convert units if required
        if V_unit == 'ml':
            df['V_B'] = df['V_B'] * 10 ** 6
            df['Q'] = df['Q'] * 10 ** 6
            df['Q_in'] = df['Q_in'] * 10 ** 6
        if p_unit == 'cmH2O':
            df['p_D'] = df['p_D'] * 0.010197162129779282
            df['p_S'] = df['p_S'] * 0.010197162129779282
        
        return df