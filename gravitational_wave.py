import calc_plot_points as calpp
import numpy as np
import matplotlib.pyplot as plt

def h22(t, y, R):
    """Jouw formule (164), letterlijk."""
    y = np.asarray(y, float)
    nu = calpp.nu
    r, phi, pphi, pr = y

    # Omega = dphi/dt, en de rates voor de kettingregel
    dr, Om, dpphi, dpr = calpp.derivatives(t, y)

    # v_phi = Omega * r_omega   (94)+(95)+(96)
    u     = 1/r
    A     = calpp.calc_A(nu, u)
    dA_dr = calpp.calc_A_derivative(nu, u)*(-u*u)
    psi   = 2/r**2*(1/dA_dr)*(1 + 2*nu*(np.sqrt(A*(1 + pphi**2/r**2)) - 1))
    v     = Om*r*psi**(1/3)

    # ddot r  (159)+(160)
    ddr = 0.0
    for i, rate in ((0, dr), (2, dpphi), (3, dpr)):
        h  = 1e-6*max(abs(y[i]), 1e-3)
        yp = y.copy(); yp[i] += h
        ym = y.copy(); ym[i] -= h
        ddr += (calpp.derivatives(t, yp)[0] - calpp.derivatives(t, ym)[0])/(2*h)*rate

    # f_NQC  (157)+(158)
    nqc = 1 - 0.036347*pr**2/(r*Om)**2 + 1.2468*ddr/(r*Om**2)

    # jouw formule
    return (nu * (-32*np.pi*np.sqrt(6)/15) * v**2 * nqc
            * np.sqrt(5/(64*np.pi)) * np.exp(-2j*phi) / R)

# sol = calpp.get_solution([15, 0, calpp.calc_p_varphi(calpp.nu, 1/15), 0])
def initial_conditions(r0):
    return [r0, 0.0, calpp.calc_p_varphi(calpp.nu, 1.0/r0), 0.0]

sol = calpp.get_solution(initial_conditions(20))

R = 410*3.0857e22/(70*1476.6)      # afstand in geometrische eenheden: 1.22e20
t = np.linspace(0, sol.t[-1], 5000)
Y = sol.sol(t)
h = np.array([h22(ti, Y[:, k], R) for k, ti in enumerate(t)])

plt.plot(t*70*4.9255e-6, h.real)   # t in seconden voor M = 70 zonsmassa's
plt.xlabel("t [s]")
plt.ylabel(r"$\Re\,h_{22}$")
plt.title(rf"Zwaartekrachtgolven, $R={R:.2e}$, $M=70\,M_\odot$")
plt.show()