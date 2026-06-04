from scipy.integrate import quad
import scipy.constants as constants
import scipy
# imported the things that are used more often seperately, but might be better to use either full aliasing, or none
import numpy
from functools import cache

"""
here are the constants used in the calcs
you can go for natural coords with 
c=G=M=1
or you could use the classical values
"""
a_4 = 94/3 - 41/32*constants.pi**2
gamma = (a_4 + 4) / 2
c = 1
G = 1
M = 1
"""
c = constants.c
G = constants.G
M = constants.M
"""


@cache
def get_u(R):
  return (G*M) / (c**2*R)

def calc_D(nu, u):
  return 1 - 6*nu*u**2 + 2*(3*nu-26)*nu*u**3

def calc_A(nu, u):
  return (1-2*u) * ((1-(a_4+4)/2*u) / (1-(a_4+4)/2*u-2*nu*u**3))

def calc_A_derivative(nu, u):
  return -2*(1-gamma*u) / (1-gamma*u-2*nu*u**3) + (1-2*u) * (-4*nu*gamma*u**3+6*nu*u**2) / ((1-gamma*u-2*nu*u**3)**2)

def integrand(R, nu):
  return numpy.sqrt(calc_D(R, nu))/calc_A(R, nu)

def calc_tortoise(R, nu, R0):
  tortoise = quad(integrand, R0, R, args=(nu))
  return tortoise

def calc_squared_angular_momentum(nu, u):
  return -(calc_A_derivative(nu, u) / (2*u*calc_A(nu, u) + u**2*calc_A_derivative(nu, u)))

def calc_angular_momentum(nu, u):
  return scipy.sqrt(calc_squared_angular_momentum(nu, u))
calc_p_varphi = calc_angular_momentum

def calc_T_X(t, r):
  # calc V
  v_null =  1/2 * (t-r) 
  V = scipy.arctan(v_null)

  # calc U
  u_null = 1/2 * (t+r) 
  U = scipy.arctan(u_null)

  # calc the graph coords
  T = (V + U) / 2
  X = (V - U) / 2
  return T, X

# might needs to be a constant
def calc_z_3(nu):
  return 2*nu * (4 - 3*nu)

# H is the hamiltonian
def calc_H_eff(p_r, p_varphi, nu, r):
  return scipy.sqrt(p_r**2 + calc_A(nu, r) * (1 + p_varphi**2/r**2 + calc_z_3(nu) * p_r**4/r**2))

def calc_H(p_r, p_varphi, nu, r):
  return 1/nu * scipy.sqrt(1 + 2*nu*(calc_H_eff(p_r, p_varphi, nu, r) - 1))
