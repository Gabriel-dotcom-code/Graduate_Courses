import math

def euler_formula():
    # Defina os valores de 'e' e 'i'
    e = math.e
    i = complex(0, 1)

    # Calcule o resultado da fórmula de Euler
    result = e**(i*math.pi) + 1

    return result

# Teste a função
print(euler_formula())