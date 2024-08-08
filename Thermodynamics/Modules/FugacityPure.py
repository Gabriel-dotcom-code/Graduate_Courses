import numpy as np
from scipy.optimize import fsolve

# Constants
R = 0.08206  # Gas constant in L·atm/(K·mol)

class PureComponentFugacity:
    def __init__(self, Pc, Tc, omega, name, T_stream, P_stream):
        """
        Initialize the compound with critical properties and system conditions.
        
        Parameters:
        - Pc: Critical pressure [atm]
        - Tc: Critical temperature [K]
        - omega: Acentric factor
        - T_stream: System temperature [K]
        - P_stream: System pressure [atm]
        """
        self.Pc = Pc
        self.Tc = Tc
        self.omega = omega
        self.name = name

        # System conditions
        self.T_system = T_stream
        self.P_system = P_stream

        # Reduced properties
        self.Tr = self.T_system / self.Tc

    def calculate_fugacity(self, eos_type):
        if eos_type == "vdw":
            a, b = self._van_der_waals()
        elif eos_type == "rk":
            a, b = self._redlich_kwong()
        elif eos_type == "srk":
            a, b = self._soave_redlich_kwong()
        elif eos_type == "pr":
            a, b = self._peng_robinson()
        else:
            raise ValueError("Invalid EOS type. Choose from 'vdw', 'rk', 'srk', or 'pr'.")

        # Solve for molar volume (Vm) using fsolve
        V_m_initial_guess = R * self.T_system / self.P_system  # Ideal gas approximation
        V_m = fsolve(self._eos_function, V_m_initial_guess, args=(a, b, eos_type))[0]

        # Calculate compressibility factor Z
        Z = self.P_system * V_m / (R * self.T_system)

        # Calculate A and B
        A = a * self.P_system / (R**2 * self.T_system**2)
        B = b * self.P_system / (R * self.T_system)

        # Calculate fugacity coefficient (phi) based on EOS
        if eos_type == "vdw":
            ln_phi = Z - 1 - np.log(Z - B)
        elif eos_type == "rk" or eos_type == "srk":
            ln_phi = Z - 1 - np.log(Z - B) - (A / B) * np.log(1 + B / Z)
        elif eos_type == "pr":
            ln_phi = Z - 1 - np.log(Z - B) - (A / B / np.sqrt(8)) * np.log((Z + (1 + np.sqrt(2)) * B) / (Z + (1 - np.sqrt(2)) * B))

        phi = np.exp(ln_phi)
        fugacity = phi * self.P_system

        return V_m, Z, phi, fugacity

    def _eos_function(self, V_m, a, b, eos_type):
        """
        EOS function to solve for molar volume.
        
        Parameters:
        - V_m: Molar volume [L/mol]
        - a: EOS-specific 'a' parameter
        - b: EOS-specific 'b' parameter
        - eos_type: Type of EOS ('vdw', 'rk', 'srk', 'pr')
        
        Returns:
        - Residual function value for fsolve
        """
        P = self.P_system
        T = self.T_system
        
        if eos_type == "vdw":
            return P - (R * T / (V_m - b)) + (a / V_m**2)
        elif eos_type == "rk":
            return P - (R * T / (V_m - b)) + (a / (V_m * (V_m + b) * np.sqrt(T)))
        elif eos_type == "srk":
            return P - (R * T / (V_m - b)) + (a / (V_m * (V_m + b)))
        elif eos_type == "pr":
            return P - (R * T / (V_m - b)) + (a / (V_m**2 + 2*b*V_m - b**2))

    def _van_der_waals(self):
        a = 27 * (R * self.Tc)**2 / (64 * self.Pc)
        b = R * self.Tc / (8 * self.Pc)
        return a, b

    def _redlich_kwong(self):
        a = 0.42748 * (R**2 * self.Tc**2.5) / (self.Pc * np.sqrt(self.T_system))
        b = 0.08664 * R * self.Tc / self.Pc
        return a, b

    def _soave_redlich_kwong(self):
        alpha = (1 + (0.48 + 1.574 * self.omega - 0.176 * self.omega**2) * (1 - np.sqrt(self.Tr)))**2
        a = 0.42748 * (R**2 * self.Tc**2) / self.Pc * alpha
        b = 0.08664 * R * self.Tc / self.Pc
        return a, b

    def _peng_robinson(self):
        alpha = (1 + (0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2) * (1 - np.sqrt(self.Tr)))**2
        a = 0.45724 * (R**2 * self.Tc**2) / self.Pc * alpha
        b = 0.07780 * R * self.Tc / self.Pc
        return a, b