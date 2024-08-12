import numpy as np
import pandas as pd

class Objective:
    def __init__(self, df, x, P):
        """
        Initialize the objective function with the data, mole fractions, and pressure.
        
        Parameters:
        - df: Pandas DataFrame with columns ['Component', 'A', 'B', 'C']
        - x: Mole fractions of each component (array-like)
        - P: System pressure [mmHg]
        """
        self.df = df
        self.x = np.array(x)
        self.P = P

    def antoine(self, T, A, B, C, *args):
        """
        Calculate the vapor pressure using the Antoine equation.
        
        Parameters:
        - T: Temperature [Celsius]
        - A, B, C: Antoine constants for the component
        
        Returns:
        - Vapor pressure [mmHg]
        """
        return np.exp(A - B / (T + C))

    def __str__(self):
        s = f'The bubble point temperature is {self.T:1.2f} degC, and the gas phase compositions are {np.round(self.y, 4)}.'
        return s

    def __call__(self, T):
        """
        Evaluate the objective function at temperature T.
        
        Parameters:
        - T: Temperature [Celsius]
        
        Returns:
        - Residual value for fsolve to minimize (1 - sum of y)
        """
        T = float(T)
        self.T = T

        # Calculate vapor pressures using the Antoine equation for each component
        self.df['Pvap'] = self.df.apply(lambda row: self.antoine(T, row['A'], row['B'], row['C']), axis=1)
        
        # Calculate the gas phase mole fractions
        self.y = self.x * self.df['Pvap'].values / self.P
        
        return 1 - self.y.sum()