import numpy as np

# Define constants class
class Constants():
    def __init__(self):
        # self.HI_FREQ = 1420.405752 # MHz
        self.HI_FREQ = 1.420405752 # GHz
        self.LIGHT_SPEED = 2.99792458e5 # km/s

# Convert redshift to frequency
def zTOfreq(z, rest_freq):
    freq = rest_freq/(1. + z)
    return freq

# Convert redshift to velocity
def zTOvel(z, option):
    if option == 'radio':
        velocity = z/(1.+z)
    if option == 'optical':
        velocity = z
    if option == 'relativistic':
        velocity = ((1.+z)**2 - 1.)/((1.+z)**2 + 1.) 
    return velocity

# Convert velocity to redshift
def velTOz(velocity, option):
    if option == 'radio':
        z = velocity/(1-velocity)
    if option == 'optical':
        z = velocity 
    if option == 'relativistic':
        z = np.sqrt((1.+velocity)/(1.-velocity)) - 1.
    return z

# Convert frequency to redshift
def freqTOz(freq, rest_freq):
    redshift = (rest_freq/freq) - 1.
    return redshift

# Convert frequency to velocity
def freqTOvel(freq, rest_freq, option):
    if option == 'radio':
        velocity = 1. - (freq/rest_freq)
    if option == 'optical':
        velocity = (rest_freq/freq) - 1.
    if option == 'relativistic':
        velocity = (1. - ((freq/rest_freq)**2))/(1. + ((freq/rest_freq)**2))
    return velocity

# Convert velocity to frequency
def velTOfreq(velocity, rest_freq, option):
    if option == 'radio':
        freq = (1. - velocity)*rest_freq
    if option == 'optical':
        freq = rest_freq/(1. + velocity)
    if option == 'relativistic':
        freq = np.sqrt((1. - velocity)/(1. + velocity))*rest_freq
    return freq

# Convert flux depth to optical depth
def fluxTOopd(dflux, flux, option='thick'):
    if option == 'thin':
        opd = -dflux/flux
    else:
        opd = -np.log((dflux/flux)+1)
    return opd

# Convert redshift from one rest frame to another
def shift_frame(z,z_shift):
    z = (1.+z)/(1.+z_shift) - 1.
    return z
