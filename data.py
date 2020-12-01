from conversions import *
from initialize import *
from scipy import special 
import numpy as np

# Define source class
class Source():
    def __init__(self):
        self.number = 0
        self.rank = 0
        self.info = {'name':'',
                    'flux':0.,
                    'e_flux':0.,
                    'z':0.,
                    'e_z':0.,
                    'ra':'00:00:00',
                    'dec':'00:00:00'}
        self.spectrum = Spectrum()

# Define spectrum class
class Spectrum():
    def __init__(self):
        self.filename = ''
        self.ndata = 0
        self.x = X()
        self.y = Y()

    def generate_data(self,options):
        
        # Initialize constants object
        constants = Constants()

        # Read data from file
        data = np.genfromtxt(self.filename, comments='#')
        self.x.data = data[:,0][(data[:,0]>=float(options.x_min))&(data[:,0]<=float(options.x_max))]
        self.x.data = data[:,0][(data[:,0]>=float(options.x_min))&(data[:,0]<=float(options.x_max))]    
        self.y.data = data[:,1][(data[:,0]>=float(options.x_min))&(data[:,0]<=float(options.x_max))] 
        self.y.data = data[:,1][(data[:,0]>=float(options.x_min))&(data[:,0]<=float(options.x_max))]         

        # Apply mask file
        if os.path.exists(options.mask_path):
            masks = np.genfromtxt(options.mask_path, comments='#')
            for mask in masks:
                self.y.data[(self.y.data > np.min(mask)) & (self.y.data < np.max(mask))] = 0.

        # Invert data if required
        if options.invert_spectra:
            self.y.data = -1*self.y.data

        # Calculate noise level
        if (np.size(data,1) > 2) and (not options.std_madfm):
            self.y.sigma = np.abs(data[:,2])
        else:
            filtered = self.y.data[~np.isnan(self.y.data) & (self.y.data!=0.)]
            sigma_rough = std_madfm(filtered)
            clipped = filtered[np.abs(filtered-np.median(filtered))<=3.0*sigma_rough]
            self.y.sigma = std_madfm(clipped)*np.ones(self.y.data.shape)
            print('noise = %g'%self.y.sigma[0])
        self.y.sigma *= options.noise_factor

        self.ndata = len(self.x.data)

        # Down weight all channels flagged with 'NaN' or "0"
        self.y.sigma[np.isnan(self.y.data) | (self.y.data==0.) | np.isnan(self.y.sigma) | (self.y.sigma==0.)] = 1.e99
        self.y.data[np.isnan(self.y.data)] = 0.

        # Generate covariance matrix if required
        if options.corr_switch:
            tmp = signal.correlate(self.y.chanfunc[1],self.y.chanfunc[1],mode='full')
            tmp = tmp[np.equal(np.mod(self.y.chanfunc[0], 1), 0)]
            tmp = tmp/np.max(tmp)
            if len(tmp) < self.ndata:
                zpad = np.zeros(self.ndata-len(tmp))
                tmp = np.append(zpad,tmp)
            tmp = linalg.circulant(np.flipud(tmp))
            self.y.corr = 0.5*np.transpose(tmp) + 0.5*tmp
            self.y.cov = np.multiply((np.transpose(self.y.sigma.view(np.matrix))* \
                                      self.y.sigma.view(np.matrix)),self.y.corr)

            # Invert covariance matrix and produce log determinant
            self.y.invcov = linalg.inv(self.y.cov)
            eigen = np.real(linalg.eig(self.y.cov)[0])
            self.y.logdetcov = np.sum(np.log(eigen[(eigen>0)]))

        # Convert data spectrum to redshift
        if options.x_units == 'frequency':
            self.x.data = freqTOz(self.x.data,options.rest_frequency)
        elif options.x_units == 'optvel':
            self.x.data /= constants.LIGHT_SPEED
        elif options.x_units == 'redshift':
            pass
        self.x.diff = np.abs(self.x.data[1] - self.x.data[0])

        # Construct frequency window function
        self.x.chansamp = 1
        if options.channel_function == 'file':
            self.y.chanfunc = np.genfromtxt(options.channel_path, unpack=True)
            self.x.chansamp = int(round(np.power(abs(self.y.chanfunc[0,0]-self.y.chanfunc[0,1]),-1)))
        elif options.channel_function == 'square':
            self.x.chansamp = 10
            inc = float(1.0/self.x.chansamp)
            self.y.chanfunc = np.arange(0.0,1.0+inc,inc)
            tmp = np.zeros(len(self.y.chanfunc))
            tmp[self.y.chanfunc <= 0.5] = 1.0 
            self.y.chanfunc = np.vstack([self.y.chanfunc,tmp])

        # Generate fine data sampling if required
        if self.x.chansamp != 1:
            for ind in range(0,len(self.x.data)):
                if ind == 0:
                    pass
                else:
                    upper_z = self.x.data[ind]
                    upper_freq = zTOfreq(upper_z,options.rest_frequency)
                    lower_z = self.x.data[ind-1]
                    lower_freq = zTOfreq(lower_z,options.rest_frequency)
                    dfreq = (upper_freq - lower_freq)/self.x.chansamp
                    fine_freq = lower_freq + np.array(range(0,self.x.chansamp))*dfreq
                    fine_z = freqTOz(fine_freq,constants.HI_FREQ)
                    if ind == 1:
                        self.x.fine = fine_z
                    else:
                        self.x.fine = np.hstack([self.x.fine,fine_z])
            self.x.fine = np.hstack([self.x.fine,self.x.data[-1]])
        elif self.x.chansamp == 1:
            self.x.fine = self.x.data
        self.x.nfine = len(self.x.fine)

        return self

# Define spectral x-axis data class
class X():
    def __init__(self):
        self.data = np.array([0])
        self.diff = np.array([0])
        self.ledge = np.array([0])
        self.hedge = np.array([0])
        self.fine = np.array([0])
        self.nfine = 0
        self.chansamp = 0

# Define spectrum y-axis data class
class Y():
    def __init__(self):
        self.data = np.array([0])
        self.sigma = np.array([0])
        self.contsub = np.array([0])
        self.res = np.array([0])
        self.corr = np.array([0])
        self.cov = np.array([0])
        self.invcov = np.array([0])
        self.chanfunc = np.array([0])
        self.logdetcov = 0




