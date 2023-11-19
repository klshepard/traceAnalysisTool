#!/usr/bin/python

# This stuff comes from RMM Smeets' 2006 Nano letters paper and Binquan Luan Nanotechnology 2015

d = 4e-9  # m
L = 10e-9  # m
C = 0.01  # mol/liter
f = 0.001  # C/m^2
assert f > 0, "You may need the to change mu_cl to mu_k in G_surf."

v = 0.500  # Volt

mu_k = 7.616e-8  # m^2/(V*sec)
mu_cl = 7.909e-8  # m^2/(V*sec)
q = 1.602e-19  # Couloumb

pi = 3.14159
N = 6.023e23
n = N * 1e3 * C  # mol/liter to particle/m^3

sigma = (mu_k + mu_cl) * n * q
R = (1 / sigma) * ((4 / pi) * (L / d ** 2) + 1 / d)
G = 1 / R

# G_surf = (pi / 4) * ((d ** 2) / L) * (sigma + 4*mu_cl*f/d)
## Forget the above. It may be wrong, do a numerical approach right out of Binquan's 2015 paper

Ga = sigma * 2 * d
G0 = sigma * pi * d ** 2 / (4 * L)
Gs = mu_cl * pi * f * d / L

G_surf = 1 / (1 / Ga + 1 / (G0 + Gs))

print("Conductivity with access resistances: " + str(G))
print("Resistivity: {s1:.4g} Ohm".format(s1=R))
print("Voltage: " + str(v))
print("Current: " + str(G * v))
print(" ")
print("Below: conductivity assuming surface charge correction!")
print("Surface charge in C/m^2: " + str(f))
print("Current with surface charge: " + str(G_surf * v))

print(" ")
print("-------------- nanopore dna loading -------------")

dsDNA_size = 1000  # bases, of dsDNA
shipping_concentration = 0.5  # this is ug/uL -- micrograms per microliter
tube_volume = 20e-6  # liters for 10 ug at 0.5 ug/ul, per the package
molar_mass = 650  # per dsDNA base g/mol

tube_conc = 1 / ((molar_mass * dsDNA_size / shipping_concentration))

print("Concentration in the tube: {s1:} M".format(s1=tube_conc))

added_volume = 500e-6

final_conc = tube_conc * tube_volume / (added_volume + tube_volume)

print(
    "final concentration after dilution with {s1:} L: {s2:} M".format(
        s1=added_volume, s2=final_conc
    )
)
