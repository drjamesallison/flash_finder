import sys
import os
import argparse

# Set habs_nest.py options
parser = argparse.ArgumentParser()
parser.add_argument('--busyn', default=2, type=int, 
                    help='parabolic parameter value to be used with Busy Function')
parser.add_argument('--ceff', action='store_true', default=False,
                    help='constant efficiency mode used in MultiNest')
parser.add_argument('--channel_path', default=os.getcwd()+'/data/', type=str, 
                    help='full path to spectral channel response file')
parser.add_argument('--channel_function', choices = ['none','square','file'], default='none', type=str, 
                    help='spectral channel response function')
parser.add_argument('--corr_path', default=os.getcwd()+'/data/', type=str,
                    help='full path to the spectral correlation matrix')
parser.add_argument('--corr_switch',action='store_true', default=False,
                    help='account for correlation in the noise')
parser.add_argument('--data_path', default=os.getcwd()+'/data/', type=str,
                    help='full path to the data directory')
parser.add_argument('--detection_limit', default=1., type=float,
                    help='set detection limit based on the natural logarithm of the Bayes factor (i.e. ln(B_det/B_null).')
parser.add_argument('--efr', default=0.3, type=float,
                    help='sampling efficiency used in MultiNest')
parser.add_argument('--flux_limit', default=1.0e99, type=float,
                    help='maximum possible value for spectral line parametrisation')
parser.add_argument('--init_MPI', action='store_true', default=False,
                    help='use MPI in MultiNest')
parser.add_argument('--invert_spectra', action='store_true', default=False,
                    help='invert input spectra')
parser.add_argument('--mask_path', default=os.getcwd()+'/mask.txt', type=str,
                    help='full path to data mask file')
parser.add_argument('--mmodal', action='store_true', default=False,
                    help='mmodal parameter used in MultiNest')
parser.add_argument('--model_path', default=os.getcwd()+'/model.txt', type=str,
                    help='full path to model file')
parser.add_argument('--mpi_switch',action='store_true', default=False,
                    help='use MPI functionality')
parser.add_argument('--name_switch',action='store_true', default=False,
                    help='write source name on plot')
parser.add_argument('--nlive', default=3000, type=int,
                    help='nlive parameter used in MultiNest')
parser.add_argument('--nondet_vel', default=100.0, type=float,
                    help='spectral line FWHM used for non-detections')
parser.add_argument('--noise_factor', default=1.0, type=float,
                    help='factor by which to multiply the noise in the data')
parser.add_argument('--out_path', default=os.getcwd()+'/chains/', type=str,
                    help='full path to output')
parser.add_argument('--param_out', choices = ['maxl','median'], default='maxl', type=str,
                    help='component parameters values either maximum likelihood or median with uncertainties')
parser.add_argument('--plot_evidence', action='store_true', default=False,
                    help='add evidence value to plots')
parser.add_argument('--plot_switch', action='store_true', default=False,
                    help='produce plots')
parser.add_argument('--plot_nchans', default=200, type=int,
                    help='number of spectral channels to plot')
parser.add_argument('--plot_restframe', choices = ['none','source','peak'], default='peak', type=str,
                    help='restframe used for plotting')
parser.add_argument('--rest_frequency', default=1.420405752, type=float,
                    help='rest frequency of transition in GHz')
parser.add_argument('--small_plots', action='store_true', default=False,
                    help='plot a small range of the x-axis')
parser.add_argument('--std_madfm', action='store_true', default=False,
                    help='use MADFM estimator of the standard deviation in place of noise recorded in spectra')
parser.add_argument('--tol', default=0.5, type=float,
                    help='evidence tolerance used in MultiNest')
parser.add_argument('--verbose',action='store_true',default=False,
                    help='switch on verbose output (not recommended for large parallel jobs)')
parser.add_argument('--watch',action='store_true',default=False,
                    help='switch on progress plots')
parser.add_argument('--x_units', choices = ['frequency','optvel','redshift'], default='optvel', type=str,
                    help='units for the spectral x-axis (GHz, km/s or z)')
parser.add_argument('--x_max', default=1.e99, type=float,
                    help='maximum value for x-axis')
parser.add_argument('--x_min', default=-1.e99, type=float,
                    help='minimum value for x-axis')
parser.add_argument('--x0_sigma', default=50, type=float,
                    help='sigma width when using a normal prior for x0 (use same units as supplied source redshift)')
parser.add_argument('--y_units', choices = ['mJy','mJy/beam','Jy','Jy/beam','abs'], default='mJy', type=str,
                    help='units for the spectral y-axis (mJy, mJy/beam, Jy, Jy/beam or absorbed fraction)')
parser.add_argument('--ylim_scale', default=1.5, type=float,
                    help='y-axis scaling in plots')
options = parser.parse_args()


        
