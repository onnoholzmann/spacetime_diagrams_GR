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
a_4 = 94/3 - 41/32*scipy.constants.pi**2
gamma = (a_4 + 4) / 2
c = 1
G = 1
M = 1
"""
c = scipy.constants.constants.c
G = scipy.constants.constants.G
M = scipy.constants.constants.M
"""

# prob needs to be removed
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
  return numpy.sqrt(calc_D(nu, 1/R))/calc_A(nu, 1/R)

def calc_tortoise(R, nu, R0):
  tortoise = scipy.integrate.quad(integrand, R0, R, args=(nu))
  return tortoise

def calc_squared_angular_momentum(nu, u):
  return -(calc_A_derivative(nu, u) / (2*u*calc_A(nu, u) + u**2*calc_A_derivative(nu, u)))

def calc_angular_momentum(nu, u):
  return numpy.sqrt(calc_squared_angular_momentum(nu, u))
calc_p_varphi = calc_angular_momentum

def calc_T_X(t, r):
  # calc V
  v_null =  1/2 * (t-r) 
  V = numpy.arctan(v_null)

  # calc U
  u_null = 1/2 * (t+r) 
  U = numpy.arctan(u_null)

  # calc the graph coords
  T = (V + U) / 2
  X = (V - U) / 2
  return T, X

# might needs to be a constant
def calc_z_3(nu):
  return 2*nu * (4 - 3*nu)

# H is the hamiltonian
def calc_H_eff(p_r, p_varphi, nu, r):
  return numpy.sqrt(p_r**2 + calc_A(nu, 1/r) * (1 + p_varphi**2/r**2 + calc_z_3(nu) * p_r**4/r**2))

def calc_H(p_r, p_varphi, nu, r):
  return 1/nu * numpy.sqrt(1 + 2*nu*(calc_H_eff(p_r, p_varphi, nu, r) - 1))

# nu = mu / M
# nu is the mass ratio
nu = 1/4
# since nu is constant, z_3 is also constant
z_3 = calc_z_3(nu)

def derivatives(t, y):
  r, varphi, p_varphi, p_r = y

  # calc all the values used in the derivatives
  u=1/r
  A = calc_A(nu, u)
  A_derivative = calc_A_derivative(nu, u)
  D = calc_D(nu, u)
  H = calc_H(p_r, p_varphi, nu, r)
  H_eff = calc_H_eff(p_r, p_varphi, nu, r)

  # the equations of motion
  dvarphi_dt = (A*p_varphi) / (nu*r**2*H*H_eff)
  dr_dt = A/numpy.sqrt(D) * 1/(nu*H*H_eff) * (p_r + z_3 * (2*A)/(r**2) * p_r**3)
  dp_varphi_dt = 0
  # it is assumed 0, since we assume no radiation(friction in this case), at least for now
  dp_r_dt = -A/numpy.sqrt(D) * 1/(2*nu*H*H_eff) * (A_derivative + p_varphi**2/r**2 * (A_derivative - 2*A/r) + z_3*(A_derivative/r**2 - 2*A/r**3) * p_r**4)

  return [dr_dt, dvarphi_dt, dp_varphi_dt, dp_r_dt]

# nu needs to be defined as a constant, as it isn't changing
def check_event_horizon(t, y):
  r = y[0]
  return calc_A(nu, 1/r)
check_event_horizon.terminal = True
check_event_horizon.direction = -1

# initial_conditions = [r_0, varphi_0, p_varphi_0, p_r_0]
initial_conditions = [15, 0, calc_p_varphi(nu, 1/15), 0]
solution = scipy.integrate.solve_ivp(derivatives, t_span = (0, 100000), y0=initial_conditions, events=check_event_horizon, dense_output=True)

# results
print(solution.t, solution.y[0], solution.y[1])
for i in range(len(solution.t)):
  print(i, solution.t[i], solution.y[0][i], solution.y[1][i])