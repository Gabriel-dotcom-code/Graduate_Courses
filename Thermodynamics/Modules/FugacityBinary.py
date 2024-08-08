import numpy as np
from scipy.optimize import fsolve

# Dados iniciais:
R = 83.14   # cm^3.bar/(g.K)

class Compound:
    records_from_compound = list()
    def __init__(self, Pc, Tc, MW, w, name,
                 T_stream, P_stream):
        """
        Tc: temperatura crítica do composto [K]
        Pc: pressão crítica do composto [bar]
        MW: peso molecular do composto [g/mol]
        w: fator acêntrico do composto
        name: nome do composto
        T_stream: temperatura do sistema [K]
        P_stream: pressão do sistema [bar]
        """
        self.Tc = Tc       
        self.Pc = Pc
        self.MW = MW
        self.w = w
        self.name = name

        # cálculos iniciais
        T_system = T_stream
        self.T_system = T_system

        P_system = P_stream
        self.P_system = P_system

        Tr_i = T_system/Tc   # cálculo da temperatura reduzida do composto
        self.Tr_i = Tr_i

        Pr_i = P_system/Pc   # cálculo da pressão reduzida do composto
        self.Pr_i = Pr_i

        alpha_SRK = (1 + (0.480 + 1.574*w - 0.176*w**2)*(1 - np.sqrt(Tr_i)))**2
        alpha_PR = (1 + (0.37464 + 1.54226*w - 0.26992*w**2)*(1 - np.sqrt(Tr_i)))**2

        # tupla que contém: a(T), b (parâmetros das equações de estado) 
        self.params = [
            (27*(R*Tc)*(R*Tc)/(64*Pc), (R*Tc)/(8*Pc)),  # Van der Waals  
            (0.42748*R*R*(Tc**2.)*(Tr_i**(-0.5))/Pc, 0.08664*R*Tc/Pc),   # Redlich/Kwong
            (0.42748*R*R*Tc*Tc/Pc, 0.0867*R*Tc/Pc),     # Soave-Redlich/Kwong
            (0.45724*R*R*Tc*Tc/Pc, 0.0778*R*Tc/Pc)      # Peng Robinson
        ]
        Compound.records_from_compound.append(self)
    
    # função definida apenas para misturas binárias
    def mixing_rule(comp1, comp2, eos, y_mole_fraction_comp1):
        # unpacking dos parâmetros do composto 1 e 2
        a_i, b_i = comp1.params[eos]  # parametros do composto 1
        a_j, b_j = comp2.params[eos]  # parametros do composto 2

        # regra de mistura de Van der Waals a partir dos parâmetros dos compostos 1 e 2, calculados anteriormente
        a_T = ((y_mole_fraction_comp1**2) * a_i ) + (2*y_mole_fraction_comp1*(1-y_mole_fraction_comp1))*(np.sqrt(a_i*a_j)) + ((1-y_mole_fraction_comp1)**2) * a_j
        bi = y_mole_fraction_comp1*b_i + (1-y_mole_fraction_comp1)*b_j    

        # cáculo do parâmetro q
        q = a_T/(R*bi*comp1.T_system)
        # cálculo do parâmetro betha
        betha = bi*comp1.P_system/(R*comp1.T_system)

        # fator de compressibilidade
        compress_factor = lambda Z: 1 + betha - q*betha*(Z - betha)/(Z*(Z + betha)) - Z
        Z = fsolve(compress_factor, 1)[0]
        
        # cálculo parâmetro I
        I = np.log((Z + betha)/Z)

        # parâmetros de interação para o cálculo do coeficiente de fugacidade
        a_bar_1 = 2*y_mole_fraction_comp1*a_i + 2*(1-y_mole_fraction_comp1)*np.sqrt(a_i*a_j) - a_T
        a_bar_2 = 2*(1-y_mole_fraction_comp1)*a_j + 2*y_mole_fraction_comp1*np.sqrt(a_i*a_j) - a_T
        b_bar_1 = b_i
        b_bar_2 = b_j

        q_bar_1 = q*((2*y_mole_fraction_comp1*a_i + 2*(1-y_mole_fraction_comp1)*np.sqrt(a_i*a_j))/a_T - b_bar_1/bi)
        q_bar_2 = q*((2*(1-y_mole_fraction_comp1)*a_j + 2*y_mole_fraction_comp1*np.sqrt(a_i*a_j))/a_T - b_bar_2/bi)

        # cálculo do coeficiente de fugacidade
        phi_1 = np.exp((b_i/bi)*(Z - 1) - np.log(Z - betha) - q_bar_1*I); comp1.phi = phi_1;
        phi_2 = np.exp((b_j/bi)*(Z - 1) - np.log(Z - betha) - q_bar_2*I); comp2.phi = phi_2;
        return phi_1, phi_2
    
        