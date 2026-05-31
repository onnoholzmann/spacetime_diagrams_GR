from scipy.integrate import quad
import scipy.constants as constants
import numpy
from functools import cache

"""
here are the constants used in the calcs
you can go for natural coords with 
c=G=M=1
or you could use the classical values
"""
a_4 = 94/3 - 41/32*constants.pi**2
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

def calc_D(R, nu):
    return 1 - 6*nu*get_u(R)**2 + 2*(3*nu-26)*nu*get_u(R)

def calc_A(R, nu):
    return (1-2*get_u(R)) * ((1-(a_4+4)/2*get_u(R)) / (1-(a_4+4)/2*get_u(R)-2*nu*get_u(R)**3))

def integrand(R, nu):
    return numpy.sqrt(calc_D(R, nu))/calc_A(R, nu)

def calc_tortoise(R, nu, R0):
    tortoise = quad(integrand, R0, R, args=(nu))

