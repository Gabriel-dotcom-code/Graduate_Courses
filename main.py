import math

def euler_formula():
    # Defina os valores de 'e' e 'i'
    e = math.e
    i = complex(0, 1)

    # Calcule o resultado da fórmula de Euler explicito
    result = e**(i*math.pi) + 1

    return result

# Teste a função
print(euler_formula())

def runge_kutta_formula():
    # Defina os valores iniciais
    x0 = 0
    y0 = 1
    h = 0.1

    # Defina a função diferencial
    def f(x, y):
        return x + y

    # Implemente o método de Runge-Kutta
    k1 = h * f(x0, y0)
    k2 = h * f(x0 + h/2, y0 + k1/2)
    k3 = h * f(x0 + h/2, y0 + k2/2)
    k4 = h * f(x0 + h, y0 + k3)

    y1 = y0 + (k1 + 2*k2 + 2*k3 + k4)/6

    return y1

# Teste a função
print(runge_kutta_formula())