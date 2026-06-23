import scipy
# imported the things that are used more often seperately, but might be better to use either full aliasing, or none
import numpy
import math
import mpmath
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
  v_null = t + r
  V = numpy.arctan(numpy.exp(v_null/(4*M)))

  # calc U
  u_null = t - r
  U = numpy.arctan(-numpy.exp(-u_null/(4*M)))

  # calc the graph coords
  T = (V + U) / 2
  X = (V - U) / 2
  return T, X

# might needs to be a constant
def calc_z_3(nu):
  return 2*nu * (4 - 3*nu)

nu_pole=1/numpy.sqrt(3)
"""dubble check if it needs nu or velocity(v)"""
def remove_calc_f_taylor(nu):
  f_taylor = 1 - 1247*nu**2/336 + 4*numpy.pi*nu**3 - 44711*nu**4/9072 - 8191*numpy.pi*nu**5/672 - 16285*numpy.pi*nu**7/504
  f_taylor += nu**6*(6643739519/6985440 - 1712*numpy.euler_gamma/105 + 16*numpy.pi**2/3 - 3424*math.log(2)/105 - 1712*math.log(nu)/105)
  f_taylor += nu**8*(-323105549467/323105549467 + 232597*numpy.euler_gamma/4410 - 1369*numpy.pi**2/126 + 39931*math.log(2)/294 - 47385*math.log(3)/1568 + 232597*math.log(nu)/4410)
  return f_taylor
def calc_f_taylor_cooficients(nu):
  return [1, 0, -1247*nu**2/336, 4*numpy.pi*nu**3, -44711*nu**4/9072, -8191*numpy.pi*nu**5/672,
                        nu**6*(6643739519/6985440 - 1712*numpy.euler_gamma/105 + 16*numpy.pi**2/3 - 3424*math.log(2)/105 - 1712*math.log(nu)/105),
                        -16285*numpy.pi*nu**7/504,
                        nu**8*(-323105549467/323105549467 + 232597*numpy.euler_gamma/4410 - 1369*numpy.pi**2/126 + 39931*math.log(2)/294 - 47385*math.log(3)/1568 + 232597*math.log(nu)/4410)]
# f_taylor_cooficients = [1, 0, -1247*nu**2/336, 4*numpy.pi*nu**3, -44711*nu**4/9072, -8191*numpy.pi*nu**5/672,
                        # nu**6*(6643739519/6985440 - 1712*numpy.euler_gamma/105 + 16*numpy.pi**2/3 - 3424*math.log(2)/105 - 1712*math.log(nu)/105),
                        # -16285*numpy.pi*nu**7/504,
                        # nu**8*(-323105549467/323105549467 + 232597*numpy.euler_gamma/4410 - 1369*numpy.pi**2/126 + 39931*math.log(2)/294 - 47385*math.log(3)/1568 + 232597*math.log(nu)/4410)]

# def calc_shifted_f_taylor(nu_varphi):
  # return (1 - nu_varphi/nu_pole) * calc_f_taylor(nu_varphi)

# p, q = mpmath.pade(calc_shifted_f_taylor, 4, 4)
# p = [float(i) for i in p]
# q = [float(i) for i in q]
def calc_f_resummed(nu_varphi):
  f_taylor_cooficients = calc_f_taylor_cooficients(nu_varphi)
  shifted_f_taylor_cooficients = [None for _ in range(9)]
  for i in range(9):
    shifted_f_taylor_cooficients[i] = f_taylor_cooficients[i]
    if i > 0:
      shifted_f_taylor_cooficients[i] -= f_taylor_cooficients[i]/nu_pole
    # if i < len(f_taylor_cooficients) - 1:
    #   shifted_f_taylor_cooficients[i+1] -= f_taylor_cooficients[i]/nu_pole
  p, q = mpmath.pade(shifted_f_taylor_cooficients, 4, 4)
  p = [float(i) for i in p]
  q = [float(i) for i in q]

  # part_formula = (1 - nu_varphi/nu_pole)
  # nu_dummy = numpy.linspace(0, 0.001, 9) # create 9 close points
  # shifted_dummy = part_formula * numpy.array([calc_f_taylor(nu) for nu in nu_dummy])
  # get the cooficients
  # cooficients = numpy.polyfit(nu_dummy, shifted_dummy, 8)
  numinator = numpy.polyval(p[::-1], nu_varphi)
  denuminator = numpy.polyval(q[::-1], nu_varphi)

  return (numinator / denuminator) / (1 - nu_varphi/nu_pole)

# H is the hamiltonian
# def calc_H_eff(p_r, p_varphi, nu, r):
  # return numpy.sqrt(p_r**2 + calc_A(nu, 1/r) * (1 + p_varphi**2/r**2 + calc_z_3(nu) * p_r**4/r**2))
def calc_H_eff(p_r, p_varphi, nu, r):
  p_r_safe = numpy.clip(p_r, -1e4, 1e4)
  return numpy.sqrt(p_r_safe**2 + calc_A(nu, 1/r) * (1 + p_varphi**2/r**2 + calc_z_3(nu) * p_r_safe**4/r**2))

def calc_H(p_r, p_varphi, nu, r):
  return 1/nu * numpy.sqrt(1 + 2*nu*(calc_H_eff(p_r, p_varphi, nu, r) - 1))

# nu = mu / M
# nu is the mass ratio
nu = 1/4
# since nu is constant, z_3 is also constant
z_3 = calc_z_3(nu)

def derivatives(t, y):
  r, varphi, p_varphi, p_r = y
  p_r = numpy.clip(p_r, -1e4, 1e4)  # add this line at the top

  # calc all the values used in the derivatives
  u=1/r
  A = calc_A(nu, u)
  A_derivative = calc_A_derivative(nu, u)
  # A_derivative == dA/du
  # use the chain rule to convert to dA/dr
  dA_dr = A_derivative * (-u**2)
  D = calc_D(nu, u)
  H = calc_H(p_r, p_varphi, nu, r)
  H_eff = calc_H_eff(p_r, p_varphi, nu, r)
  # 
  # 
  # f_varphi = -32/5*varphi**5 # needs continueing

  # the equations of motion
  # Omega == dvarphi_dt
  dvarphi_dt = (A*p_varphi) / (nu*r**2*H*H_eff)
  dr_dt = A/numpy.sqrt(D) * 1/(nu*H*H_eff) * (p_r + z_3 * (2*A)/(r**2) * p_r**3)

  # nu_varphi = r * dvarphi_dt
  # Omega**5*r**4 == (nu_varphi**15 / r)
  psi = 2/r**2 * (1/dA_dr) * (1 + 2*nu*(numpy.sqrt(A * (p_varphi**2/r**2)) - 1))
  r_omega = r*psi**(1/3)
  v_varphi = dvarphi_dt * r_omega
  f_resummed = calc_f_resummed(nu)
  f_varphi = -32/5 * nu * dvarphi_dt**5*r_omega**4 * f_resummed
  dp_varphi_dt = f_varphi
  # it is assumed 0, since we assume no radiation(friction in this case), at least for now
  dp_r_dt = -A/numpy.sqrt(D) * 1/(2*nu*H*H_eff) * (A_derivative + p_varphi**2/r**2 * (A_derivative - 2*A/r) + z_3*(A_derivative/r**2 - 2*A/r**3) * p_r**4)

  return [dr_dt, dvarphi_dt, dp_varphi_dt, dp_r_dt]

# nu needs to be defined as a constant, as it isn't changing
def check_event_horizon(t, y):
  r = y[0]
  return calc_A(nu, 1/r)
check_event_horizon.terminal = True
check_event_horizon.direction = -1
"""
def calc_r_event_horizon():
  # the event horizon is where A == 0
  return scipy.optimize.fsolve(lambda r: calc_A(nu, 1/r), 12)[0]
r_event_horizon = calc_r_event_horizon()
print(f"r_event_horizon = {r_event_horizon}")
"""
# initial_conditions = [r_0, varphi_0, p_varphi_0, p_r_0]
# initial_conditions = [15, 0, calc_p_varphi(nu, 1/15), 0]
def get_solution(initial_conditions=[15, 0, calc_p_varphi(nu, 1/15), 0]):
  solution = scipy.integrate.solve_ivp(derivatives, t_span = (0, 100000), y0=initial_conditions, events=check_event_horizon, dense_output=True)
  return solution

# if this file gets executed, output the solution
if(__name__ == "__main__"):
  # results
  solution = get_solution()
  print(solution.t, solution.y[0], solution.y[1])
  for i in range(len(solution.t)):
    print(i, solution.t[i], solution.y[0][i], solution.y[1][i])