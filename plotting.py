import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
# rc('font',**{'family':'serif','serif':['serif'],'size':20})
from conversions import *
from model import *
import corner

# Make graphical plots of the best-fitting spectra for each mode
def bestfit_spectrum(options,source,model):

    # Initialize constants
    constants = Constants()

    # Loop over number of detections
    mode_index = 0
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
        if (mode_evidence < options.detection_limit):
            continue

        # Set x-axis data
        if options.plot_restframe == 'source':
            shift_z = source.info['z']
            if options.x_units == 'optvel':
                shift_z /= constants.LIGHT_SPEED
            rest_z = shift_frame(source.spectrum.x.data,source.info['z'])
            x_data = zTOvel(rest_z,'relativistic')*constants.LIGHT_SPEED            
        elif options.plot_restframe == 'peak':
            param_index = model.input.all_ndims
            comp_index = 0.0
            shift_z = 0.0
            if model.output.ndetections == 0:
                shift_z = 0.5*(source.spectrum.x.data[-1]+source.spectrum.x.data[0])
            else:
                if 'emission' in model.input.types:
                    shift_z += mode['maximum'][param_index]
                    comp_index += 1.0
                    param_index += 4
                if 'absorption' in model.input.types:
                    shift_z += mode['maximum'][param_index]
                    comp_index += 1.0
                shift_z /= comp_index
                if options.x_units == 'optvel':
                    shift_z /= constants.LIGHT_SPEED
            rest_z = shift_frame(source.spectrum.x.data,shift_z)
            x_data = zTOvel(rest_z,'relativistic')*constants.LIGHT_SPEED
        elif options.x_units == 'optvel':
            x_data = source.spectrum.x.data*constants.LIGHT_SPEED
        else:
            x_data = source.spectrum.x.data
        x_diff = np.abs(x_data[1]-x_data[0])

        # Set y-axis data
        y_contsub = np.copy(source.spectrum.y.data)
        y_res = np.copy(source.spectrum.y.data)
        y_line = []
        comp_index = 0
        param_index = 0
        line_index = 0        
        for typ in model.input.types:
            comp = Model()
            comp.input.priors = [np.copy(model.input.priors[comp_index])]
            comp.input.types = [typ]
            comp.input.models = [np.copy(model.input.models[comp_index])]
            comp.input.tmp.x = np.copy(source.spectrum.x.fine)
            ndims = len(comp.input.priors[0])
            comp.calculate_spectrum(options,source,mode['maximum'][param_index:param_index+ndims],ndims)
            comp.calculate_data(options,source)
            y_res -= comp.output.tmp.data
            if 'continuum' in typ:
                y_contsub -= comp.output.tmp.data
            elif (('emission' in typ) or ('absorption' in typ)) and (model.output.ndetections > 0):
                y_line.append(comp.output.tmp.data)
            param_index += ndims
            comp_index += 1
        y_line = np.array(y_line)

        # If convert y-units
        if options.y_units == 'abs':
            y_contsub *= 100.0
            y_res *= 100.0
            y_line *= 100.0
        elif (options.y_units == 'Jy') or (options.y_units == 'Jy/beam'):
            y_contsub *= 1.e3
            y_res *= 1.e3
            y_line *= 1.e3

        # Set font size
        font_size = 16

        # Calculate axis limits and aspect ratio
        x_min = np.min(x_data)
        x_max = np.max(x_data)
        if options.small_plots and (model.output.ndetections != 0):
            if options.plot_restframe != 'none':
                x_centre = 0.0
            else:
                x_centre = source.info['z']
            x_min = (x_centre-0.5*float(options.plot_nchans)*abs(x_diff))
            x_max = (x_centre+0.5*float(options.plot_nchans)*abs(x_diff))
        y1_array = y_contsub[np.where((x_data>np.min([x_min,options.x_min])) & (x_data<np.max([x_max,options.x_max])))]
        y1_min = np.min(y1_array)*float(options.ylim_scale)
        y1_max = np.max(y1_array)*float(options.ylim_scale)

        y2_array = y_res[np.where((x_data>np.min([x_min,options.x_min])) & (x_data<np.max([x_max,options.x_max])))]
        y2_min = np.min(y2_array)*float(options.ylim_scale)
        y2_max = np.max(y2_array)*float(options.ylim_scale)
        y_ratio = (y1_max - y1_min)/(y2_max - y2_min)

        # Initialize figure
        plt.ioff()
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.0)
        gs = gridspec.GridSpec(2, 1, height_ratios=[y_ratio, 1])
        plt.rc('xtick', labelsize=font_size-4)
        plt.rc('ytick', labelsize=font_size-4) 

        # Initialize subplots
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Set axis limits
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y1_min, y1_max)
        ax2.yaxis.set_ticks(ax1.get_yticks())
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y2_min, y2_max)

        # Plot spectra 
        #if options.channel_function == 'square':
        ax1.step(x_data, y_contsub, where='mid', color=[0.5,0.5,0.5], linestyle='-')
        if (model.output.ndetections > 0):
            ax1.step(x_data, np.sum(y_line,0), where='mid', color='k', linestyle='-') 
        ax2.step(x_data, y_res, where='mid', color='r', linestyle='-')
        # else:
        #    ax1.plot(x_data, y_contsub, color=[0.5,0.5,0.5], linestyle='-')
        #    ax1.plot(x_data, np.sum(y_line,0), color='k', linestyle='-')       
        #    ax2.plot(x_data, y_res, color='r', linestyle='-')

        # Add labelling for each component
        if model.output.ndetections != 0:
            ymax = np.zeros(len(y_line))
            for j in range(0, len(y_line)):
                ymax[j] = np.max(np.abs(y_line[j]))
            ymax_sort = np.flipud(np.sort(ymax))
            for j in range(0, len(y_line)):
                ax1.plot(x_data, y_line[j], color='g', linestyle='--',zorder=0)
                truth = [np.abs(y_line[j])==np.max(np.abs(y_line[j]))]
                comp_index = np.where(ymax_sort == ymax[j])
                ax1.text(x_data[truth]-1.*x_diff,np.max(y_contsub),'%d'%(comp_index[0]+1),color='g',fontsize=font_size-2)

        # Add evidence value to plot
        if model.output.ndetections != 0:
            if options.plot_evidence:
                plt.suptitle(r'$R = %0.2f \pm %0.2f$' % (mode_evidence, mode_evidence_err),
                             x=0.65, y=(0.2*y_ratio + 1)/(1+y_ratio), horizontalalignment='left', fontsize=font_size-2)

        # Add additional vertical and horizontal lines
        if options.plot_restframe != 'none':
            ax1.vlines(0.0,y1_min,y1_max,colors='k',linestyle=':')
            ax2.vlines(0.0,y2_min,y2_max,colors='k',linestyle=':')
        else:
            ax1.vlines(float(source.info['z'])/1e3,y1_min,y1_max,colors='k',linestyle=':')
            ax2.vlines(float(source.info['z'])/1e3,y2_min,y2_max,colors='k',linestyle=':')
        ax1.axhline(color='k', linestyle=':', zorder=0)
        ax2.axhline(color='k', linestyle=':', zorder=0)

        # Add axis labels
        ax1.set_xlabel('')
        ax1.set_xticklabels([])
        ax2.set_ylabel('')
        if options.plot_restframe != 'none':
            ax2.set_xlabel(r"$v\,[\mathrm{km}\,\mathrm{s}^{-1}]$", fontsize=font_size)
            # ax2.set_xlabel(r'$\mathrm{Relative}\,\mathrm{Gas}\,\mathrm{Velocity}\,(\mathrm{km}\,\mathrm{s}^{-1})$', fontsize=font_size)
        elif options.x_units == 'optvel':
            ax2.set_xlabel(r"$cz\,[\mathrm{km}\,\mathrm{s}^{-1}]$", fontsize=font_size)
        else:        
            ax2.set_xlabel(r"$z$", fontsize=font_size)
        if (options.y_units == 'mJy') or (options.y_units == 'Jy'):
            ylabh = ax1.set_ylabel(r'$S\,[\mathrm{mJy}]$', fontsize=font_size)
        if (options.y_units == 'mJy/beam') or (options.y_units == 'Jy/beam'):
            ylabh = ax1.set_ylabel(r"$S\,[\mathrm{mJy}\,\mathrm{beam}^{-1}]$", fontsize=font_size)
            # ylabh = ax1.set_ylabel(r"$S\,[\mathrm{Jy}\,\mathrm{beam}^{-1}]$", fontsize=font_size)
        elif options.y_units == 'abs':
            # ylabh = ax1.set_ylabel(r'$e^{-\tau}-1 [\mathrm{per}\,\mathrm{cent}]$', fontsize=font_size)
            ylabh = ax1.set_ylabel(r"$\Delta{S}/S_\mathrm{c} [\mathrm{per}\,\mathrm{cent}]$", fontsize=font_size)            
            # ylabh = ax1.set_ylabel(r"$\mathrm{Absorbed}\,\mathrm{Fraction}\,[\mathrm{per}\,\mathrm{cent}]$", fontsize=font_size)
        if options.name_switch:
            ax1.set_title('%s' % (source.info['name']), fontsize=font_size)
        ylabh.set_position((ylabh.get_position()[0],0.5*(y_ratio-1.0)/(y_ratio+1.0)))
        # ylabh.set_verticalalignment('center')

        # Nice tick mark behaviour
        ax1.minorticks_on()
        ax2.minorticks_on()
        ax1.tick_params(bottom=True,left=True,top=True,right=True,length=6,width=1,which='major',direction='in')
        ax1.tick_params(bottom=True,left=True,top=True,right=True,length=3,width=1,which='minor',direction='in')
        ax2.tick_params(bottom=True,left=True,top=True,right=True,length=6,width=1,which='major',direction='in')
        ax2.tick_params(bottom=True,left=True,top=True,right=True,length=3,width=1,which='minor',direction='in')

        # Save figure to file
        if model.output.ndetections == 0:
            line_number = 0
        else:
            line_number = mode_index+1
        plt.savefig(options.out_root+'_line_'+str(line_number)+'_bestfit_spectrum.pdf')
        # plt.savefig(options.out_root+'_line_'+str(line_number)+'_bestfit_spectrum.eps')

        # Increment
        mode_index += 1

# Make posterior plots of model component and derived parameters
def posterior_plot(options,source,model):

    # Check to see if any detections exist for this source
    if model.output.ndetections == 0:
        return

    # Loop over detection modes
    mode_index = 0
    plot_index = 0
    if options.mmodal:
        posterior_list = model.output.sline.get_separated_stats().separated_posterior
    else:
        posterior_list = [model.output.sline.get_data()]
   
    for posterior in posterior_list:

        # Calculate mode evidence
        mode = model.output.sline.get_mode_stats()['modes'][mode_index]
        mode_evidence = mode['local log-evidence']
        mode_evidence_err =  mode['local log-evidence error']
        if 'continuum' in model.input.types:
            tmp = model.output.cont.get_mode_stats()['global evidence error']
            mode_evidence_err = np.sqrt(np.power(mode_evidence_err,2) + np.power(tmp,2))

        # If mode evidence less than zero then move to next iteration
        if (mode_evidence < options.detection_limit):
            mode_index += 1
            continue

        # Initialize indexing
        offset_index = 2

        # Make plot of emission parameters 
        offset_index += model.input.all_ndims
        if 'emission' in model.input.types:
            
            if options.x_units == 'optvel':
                label_1 = r'$v_\mathrm{peak}$'
            else:
                label_1 = r'$z_\mathrm{peak}$'
            label_2 = r'$S_\mathrm{peak}\,\mathrm{[mJy]}$'
            label_3 = r'$\int{S\,\mathrm{d}v}\,\mathrm{[mJy\,\mathrm{km}\,\mathrm{s}^{-1}]}$'
            label_4 = r'$\Delta{v}_\mathrm{eff}\,[\mathrm{km}\,\mathrm{s}^{-1}]$'
            labels = [label_1,label_2,label_3,label_4]

            xs = np.array(posterior[:,offset_index:offset_index+4])
            weights = np.array(posterior[:,0])
            fig = corner.corner(xs, 
                            weights=weights,
                            labels=labels, label_kwargs={"fontsize": 15},
                            range=[0.999]*np.size(xs,1), 
                            quantiles=[0.5*(1.-0.682689492),0.5,0.5*(1.+0.682689492)],
                            show_titles=True, title_kwargs={"fontsize": 12},
                            plot_contours=True, use_math_text=True)
            fig.gca().get_xaxis().get_major_formatter().set_useOffset(False)

            # Adjust figure size
            fig.tight_layout()

            # Save figure to file
            fig.savefig(options.out_root+'_line_'+str(plot_index+1)+'_emission_posterior.pdf')

            offset_index += 4

        # Make plot of absorption parameters  
        if 'absorption' in model.input.types:
            
            if options.x_units == 'optvel':
                label_1 = r'$v_\mathrm{peak}\,\mathrm{[km\,s^{-1}]}$'
            else:
                label_1 = r'$z_\mathrm{peak}$'
            label_2 = r'$\tau_\mathrm{peak}$'            
            label_3 = r'$\int\tau\,\mathrm{d}v\,\mathrm{[\mathrm{km}\,\mathrm{s}^{-1}]}$'
            label_4 = r'$\Delta{v}_\mathrm{eff}\,[\mathrm{km}\,\mathrm{s}^{-1}]$'            
            labels = [label_1,label_2,label_3,label_4]            

            xs = np.array(posterior[:,[offset_index+0,offset_index+1,offset_index+2,offset_index+3]])
            weights = np.array(posterior[:,0])
            fig = corner.corner(xs, 
                            weights=weights,
                            labels=labels, label_kwargs={"fontsize": 15},
                            range=[0.999]*np.size(xs,1), 
                            quantiles=[0.5*(1.-0.682689492),0.5,0.5*(1.+0.682689492)],
                            show_titles=True, title_kwargs={"fontsize": 12},
                            plot_contours=True, use_math_text=True)
            fig.gca().get_xaxis().get_major_formatter().set_useOffset(False)

            # Adjust figure size
            fig.tight_layout()

            # Save figure to file
            fig.savefig(options.out_root+'_line_'+str(plot_index+1)+'_absorption_posterior.pdf')

        # Increment
        mode_index += 1
        plot_index += 1

