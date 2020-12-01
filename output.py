import numpy as np
from conversions import *
from fitting import *

# Initialize the summary file containing the results
def initialize_resultsfile(options,model):

    # Set path to results file
    results_file = options.out_path + '/results.dat'

    # Open file for writing
    f = open(results_file,'w')

    # Source and mode number 
    text = '#Name ModeNum'

    # Add model component parameters
    index = 0
    for comp in model.input.types:

        # Loop over model priors
        for prior in model.input.priors[index]:

            if options.param_out == 'maxl':
                text += ' %s_%d_maxl' % (prior[0],index+1)
            else:
                text += ' %s_%d_median %s_%d_siglo %s_%d_sighi' % (prior[0],index+1,
                                            prior[0],index+1, prior[0],index+1)

        index += 1

    # Add emission parameters
    if 'emission' in model.input.types:
        if options.x_units == 'optvel':
            text += ' emi_peakz_median(km/s) emi_peakz_siglo(km/s) emi_peakz_sighi(km/s)' 
        else:
            text += ' emi_peakz_median emi_peakz_siglo emi_peakz_sighi'
        text += ' emi_peakS_median(mJy) emi_peakS_siglo(mJy) emi_peakS_sighi(mJy)'
        text += ' emi_intS_median(mJy.km/s) emi_intS_siglo(mJy.km/s) emi_intS_sighi(mJy.km/s)'
        text += ' emi_width_median(km/s) emi_width_siglo(km/s) emi_width_sighi(km/s)'

    # Add absorption parameters
    if 'absorption' in model.input.types:
        if options.x_units == 'optvel':
            text += ' abs_peakz_median(km/s) abs_peakz_siglo(km/s) abs_peakz_sighi(km/s)'
        else:
            text += ' abs_peakz_median abs_peakz_siglo abs_peakz_sighi'                         
        text += ' abs_peakopd_median abs_peakopd_siglo abs_peakopd_sighi'
        text += ' abs_intopd_median(km/s) abs_intopd_siglo(km/s) abs_intopd_sighi(km/s)'
        text += ' abs_width_median(km/s) abs_width_siglo(km/s) abs_width_sighi(km/s)'

    text += ' R_mean R_sigma'
    text += ' chisq_mean chisq_sigma\n'

    f.write(text)
    f.close()

# Write results to the summary file
def write_resultsfile(options,source,model):

    # Open results file for appending
    results_file = options.out_path + '/results.dat'
    f = open(results_file, 'a')
        
    # Loop over detection modes
    mode_index = 0
    print_index = 0
    for mode in model.output.sline.get_mode_stats()['modes']:

        # If no detections skip further iterations
        if (model.output.ndetections == 0) and (mode_index > 0):
            continue

        # Calculate mode evidence
        mode_evidence = mode['local log-evidence']
        mode_evidence_err =  mode['local log-evidence error']
        if 'continuum' in model.input.types:
            tmp = model.output.cont.get_mode_stats()['global evidence error']
            mode_evidence_err = np.sqrt(np.power(mode_evidence_err,2) + np.power(tmp,2))

        # If mode evidence less than zero then move to next iteration
        if (model.output.ndetections > 0) and (mode_evidence < options.detection_limit):
            mode_index += 1
            continue

        # Add source name and mode number
        if (model.output.ndetections == 0):
            text = '%s %i' % (source.info['name'], print_index)
        else:
            text = '%s %i' % (source.info['name'], print_index+1)

        # Loop over each model component
        comp_index = 0
        param_index = 0
        for typ in model.input.types:

            # Set number of parameters for this component
            nparams = len(model.input.priors[comp_index])

            # Add parameters for this component
            if model.output.ndetections == 0 and 'continuum' not in typ:
                for j in range(param_index,param_index+nparams):
                    if options.param_out == 'maxl':
                        text += ' %.8f' % (0.)
                    else:
                        text += ' %.8f %.8f %.8f' % (0.,0.,0.)
                comp_index += 1
                continue
            elif model.output.ndetections == 0:
                for j in range(param_index,param_index+nparams):
                    if options.param_out == 'maxl':
                        maxl = model.output.cont.get_best_fit()['parameters'][j]
                        text += ' %.8f' % (maxl)
                    else:
                        median = model.output.cont.get_stats()['marginals'][j]['median']
                        lower = model.output.cont.get_stats()['marginals'][j]['1sigma'][0]-median
                        higher = model.output.cont.get_stats()['marginals'][j]['1sigma'][1]-median
                        text += ' %.8f %.8f %.8f' % (median,np.abs(lower),np.abs(higher))                        
            else:
                for j in range(param_index,param_index+nparams):
                    if options.param_out == 'maxl':
                        maxl = model.output.sline.get_stats()['modes'][mode_index]['maximum'][j]
                        text += ' %.8f' % (maxl)
                    else:
                        if options.mmodal:
                            median = model.output.sline.separated_stats['marginals'][mode_index][j]['median']
                            lower = model.output.sline.separated_stats['marginals'][mode_index][j]['1sigma'][0]-median
                            higher = model.output.sline.separated_stats['marginals'][mode_index][j]['1sigma'][1]-median
                        else:
                            median = model.output.sline.get_stats()['marginals'][j]['median']
                            lower = model.output.sline.get_stats()['marginals'][j]['1sigma'][0]-median
                            higher = model.output.sline.get_stats()['marginals'][j]['1sigma'][1]-median
                        text += ' %.8f %.8f %.8f' % (median,np.abs(lower),np.abs(higher))

            param_index += nparams
            comp_index += 1

        # Add astrophysical quantities
        if model.output.ndetections == 0 :
            for j in range(model.input.all_ndims,model.input.nparams):
                text += ' %.8f %.8f %.8f' % (0.,0.,0.)
        else:
            for j in range(model.input.all_ndims,model.input.nparams):
                if options.mmodal:
                    median = model.output.sline.separated_stats['marginals'][mode_index][j]['median']
                    lower = model.output.sline.separated_stats['marginals'][mode_index][j]['1sigma'][0]-median
                    higher = model.output.sline.separated_stats['marginals'][mode_index][j]['1sigma'][1]-median
                else:
                    median = model.output.sline.get_stats()['marginals'][j]['median']
                    lower = model.output.sline.get_stats()['marginals'][j]['1sigma'][0]-median
                    higher = model.output.sline.get_stats()['marginals'][j]['1sigma'][1]-median                
                text += ' %.8f %.8f %.8f' % (median,np.abs(lower),np.abs(higher))

        # Add Bayesian Evidence and Chisquared statistics
        if model.output.ndetections == 0:
            text += ' %g %g' % (0.,0.)
            if 'continuum' in typ:
                loglhood = model.output.cont.get_best_fit()['log_likelihood']
            else:
                loglhood = 0.

        else:
            text += ' %g %g' % (mode_evidence, mode_evidence_err)
            if options.mmodal:
                loglhood = model.output.sline.separated_stats['best_fit'][mode_index]['log_likelihood']
            else:
                loglhood = model.output.sline.get_best_fit()['log_likelihood']
        
        nfreedegs = len(source.spectrum.x.data)-model.input.all_ndims
        mode_chisq = -2.*(loglhood+model.output.null.logZ)/(nfreedegs)
        mode_chisq_err = np.sqrt(2./float(len(source.spectrum.x.data)))
        text += ' %g %g\n' % (mode_chisq, mode_chisq_err)
        
        f.write(text)   

        # Increment
        mode_index += 1
        print_index += 1

    f.close()

    return

# Get separated statistics for each mode
def get_separated_stats(self):

    self.separated_posterior = get_separated_posterior(self)

    self.separated_stats = {'marginals':[],'best_fit':[]}

    for posterior in self.separated_posterior:
        
        stats = []
        
        for i in range(2, posterior.shape[1]):
            b = list(zip(posterior[:,0], posterior[:,i]))
            b.sort(key=lambda x: x[1])
            b = np.array(b)
            b[:,0] = b[:,0].cumsum()
            sig5 = 0.5 + 0.9999994 / 2.
            sig3 = 0.5 + 0.9973 / 2.
            sig2 = 0.5 + 0.95 / 2.
            sig1 = 0.5 + 0.6826 / 2.
            bi = lambda x: np.interp(x, b[:,0], b[:,1], left=b[0,1], right=b[-1,1])
            
            low1 = bi(1 - sig1)
            high1 = bi(sig1)
            low2 = bi(1 - sig2)
            high2 = bi(sig2)
            low3 = bi(1 - sig3)
            high3 = bi(sig3)
            low5 = bi(1 - sig5)
            high5 = bi(sig5)
            median = bi(0.5)
            q1 = bi(0.75)
            q3 = bi(0.25)
            q99 = bi(0.99)
            q01 = bi(0.01)
            q90 = bi(0.9)
            q10 = bi(0.1)
            
            stats.append({
                'median': median,
                'sigma': (high1 - low1) / 2.,
                '1sigma': [low1, high1],
                '2sigma': [low2, high2],
                '3sigma': [low3, high3],
                '5sigma': [low5, high5],
                'q75%': q1,
                'q25%': q3,
                'q99%': q99,
                'q01%': q01,
                'q90%': q99,
                'q10%': q10,
            })
        
        self.separated_stats['marginals'].append(stats)

        # Maximum likelihood stats
        i = (-0.5 * posterior[:,1]).argmax()
        lastrow = posterior[i]
        stats = {'log_likelihood': float(-0.5 * lastrow[1]),
                'parameters': list(lastrow[2:])}
        self.separated_stats['best_fit'].append(stats)

    return self

# Read separated posterior file for multiple modes
def get_separated_posterior(self):

    self.separated_posterior = []
    f = open(self.post_file, 'r')
    sep_count = 0
    mode_count = 0
    tmp_samples = []

    while 1:
        lines = f.readlines()
        if not lines:
            break
        for line in lines:
            if not line.split():
                sep_count += 1
            elif sep_count == 2: 
                sep_count = 0
                if mode_count == 0:
                    mode_count += 1
                else:
                    tmp_samples = np.array(tmp_samples)
                    self.separated_posterior.append(tmp_samples)
                    tmp_samples = []
                    mode_count += 1
            else:
                tmp_samples.append([float(x) for x in line.split()])
    f.close()
    tmp_samples = np.array(tmp_samples)
    if mode_count > 1:
        self.separated_posterior.append(tmp_samples)
    else:
        self.separated_posterior = [tmp_samples]

    return self.separated_posterior

