import numpy, calc_plot_points as m

# --- 1: coefficients reproduce the Taylor series exactly ---
v = 0.1
c = m.calc_f_taylor_cooficients(v)
assert abs(sum(c[i]*v**i for i in range(9)) - m.remove_calc_f_taylor(v)) < 1e-12

# --- 2: shift identity, including the v^9 term that P44 drops ---
s  = [c[i] - (c[i-1]/m.nu_pole if i else 0) for i in range(9)]
s9 = -c[8]/m.nu_pole
lhs = sum(s[i]*v**i for i in range(9))
rhs = (1-v/m.nu_pole)*m.remove_calc_f_taylor(v)
assert abs(lhs + s9*v**9 - rhs) < 1e-14
print(f"flux OK  (dropped v^9 term = {s9*v**9:.3e})")

# --- 3: resummed agrees with Taylor in the weak field ---
print(f"F_resummed(0.1) = {m.calc_f_resummed(0.1):.6f}  (Taylor 0.974717)")
assert abs(m.calc_f_resummed(0.1) - m.remove_calc_f_taylor(0.1)) < 1e-3

# --- 4: domain scan ---
for r in numpy.linspace(m.R_STOP, 60, 500):
    u = 1/r
    assert m.calc_A(m.nu,u) > 0 and m.calc_D(m.nu,u) > 0, r
    assert -u*u*m.calc_A_derivative(m.nu,u) > 0, r
    assert m.calc_squared_angular_momentum(m.nu,u) > 0, r
print("domain OK")

# --- 5: circular orbit stays circular (radiation off) ---
m.RADIATION = False
sol = m.get_solution([15, 0, m.calc_p_varphi(m.nu, 1/15), 0], t_max=5000)
dev = numpy.max(numpy.abs(sol.y[0] - 15))
m.RADIATION = True
print(f"circular orbit: max|r-15| = {dev:.3e}  ({len(sol.t)} steps)")
assert dev < 1e-7, dev
# 6:
m.RADIATION = False
s = m.get_solution([15, 0, 0.9*m.calc_p_varphi(m.nu,1/15), 0], t_max=5000)
H = m.calc_H(s.y[3], s.y[2], m.nu, s.y[0])
print("max|H-H0| =", numpy.max(abs(H - H[0])))
m.RADIATION = True
print("ALL OK")