import matplotlib.pyplot as plt
import numpy as np
from scipy import constants

def angle_pot(angle, a_0, f_c):
    pot = f_c / 2.0 * (angle - a_0) ** 2
    return pot

def convert_to_joule(energy):
    #converts from kJ/mol to J
    avogadro_const = constants.value(u'Avogadro constant')
    return energy * 1000.0 / avogadro_const

def convert_angle_rad(angle):
    return angle/180.0*np.pi

angles = [a for a in range(0, 181)]
rad_angles = [convert_angle_rad(a) for a in angles]
pot = [angle_pot(a, convert_angle_rad(111.0), 530.0) for a in rad_angles]
pot = [convert_to_joule(p) for p in pot]
print(pot)

k_B = constants.value(u'Boltzmann constant')

T = 400

probs = [np.exp(-e/(k_B*T)) for e in pot]

print(probs)

plt.plot(angles, probs)
plt.show()