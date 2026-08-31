import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from Signal_source.Fitter import HornTransmissionFitter
from Signal_source.Measurement_Systems import FreqSignalMeasurementSystem
from Signal_source.Model_signals import HornSignal

from tools.dependencies import complex_to_mag_and_phase
from tools.effective_rank_modeling import compute_SVD, truncate_measurement, build_model

def plot_key_component_scheme(*, models, fits, scheme, freq, title, key_components : np.ndarray, fit_names):


    fig, ax = plt.subplots(ncols=1, nrows=1,figsize=(5.12*3, 2.88*3))

    #select component(s) to plot 
    for model, fit, name in zip(models, fits, fit_names):
        for i in key_components:
            model = np.asanyarray(model)

            model_select = model[np.where(model[:, 0] == i)]


            fit_select = fit[np.where(model[:, 0] == i)]

            #plot the components by creating individual measurement systems

            signal = HornSignal(component_matrix=model_select, component_amplitudes=fit_select)

            
            system = FreqSignalMeasurementSystem(scheme=scheme, signal=signal)

            measurement = system.Measure(freq=freq)

            mag, pha = complex_to_mag_and_phase(measurement[1])

            ax.plot(measurement[0], np.abs(measurement[1]), label=f'{name} N = {i}')
    ax.set_xlabel('Offset distance (mm)', fontsize=17)
    ax.set_ylabel('Magnitude', fontsize=17)
    ax.set_title(title)

    plt.legend(fontsize=17)
    plt.grid()
    plt.show()


def plot_complex_signals(*, measurements : np.ndarray, labels : np.ndarray, title):


    fig, ax = plt.subplots(nrows=2,
                           ncols=1)
    
    for measurement, label in zip(measurements, labels):

        # mag, pha = complex_to_mag_and_phase(measurement[1])
        
        ax[0].plot(measurement[0], np.abs(measurement[1]), label = label)
        ax[1].plot(measurement[0], np.angle(measurement[1]), label = label)


    ax[0].set_ylabel('Magnitude')
    ax[0].set_xlabel('Distance offset (mm)')

    ax[1].set_ylabel('Phase (deg)')
    ax[1].set_xlabel('Distance offset (mm)')


    ax[0].set_title(title)
    plt.legend()
    for axis in ax:
        axis.grid()
    plt.show()


def fit_signal(*,measurement_system, freq, measure_scheme,
                plot_scheme, comps, plotting_system, title='', 
                limit = False, show_Vh_text, mode_spacing=1, comp_spacing=1):
    
    
    measurement = measurement_system.Measure(freq=freq)
    fitter = HornTransmissionFitter(components=comps)
    fit = fitter.fit_points(Amplitudes=measurement[1], freq=freq, scheme=measure_scheme)

    fitted_system_1 = FreqSignalMeasurementSystem(scheme=plot_scheme,
                                                signal=HornSignal(component_matrix=comps, 
                                                                    component_amplitudes=fit))
    fitted_data = fitted_system_1.Measure(freq=freq)

    fit_mag, fit_pha = complex_to_mag_and_phase(fitted_data[1])
    measured_mag, measured_pha = complex_to_mag_and_phase(measurement[1])

    (U, S, Vh) = compute_SVD(freq=freq, 
                            measurement_scheme=measure_scheme, 
                            components=comps)

    # print(f'{U=}')
    # print(f'{S=}')
    # print(sum(S))
    # print(f'{Vh[0][0]=}')

    #real underlying signal
    real_measurement = plotting_system.Measure(freq=freq, noiseless=True)

    real_mag, real_pha = complex_to_mag_and_phase(real_measurement[1])
    
    fig = plt.figure(figsize=(5.12*3, 2.88*3), constrained_layout=True)
    
    outer = fig.add_gridspec(2, 2,
                         width_ratios=[1.1, 2.3])

    left = outer[:, 0].subgridspec(2, 1)
    right = outer[:, 1].subgridspec(2, 2,
                                    width_ratios=[0.3, 2.0],
                                    height_ratios=[0.8, 1.2])

    ax_mag = fig.add_subplot(left[0])
    ax_phase = fig.add_subplot(left[1])

    ax_U = fig.add_subplot(right[0, :])      # spans the whole top
    ax_S = fig.add_subplot(right[1, 0])      # narrow
    ax_Vh = fig.add_subplot(right[1, 1])     # large square
    

    
    fig.suptitle(title)

    ax_mag.sharex(ax_phase)
    ax_mag.set_ylabel('Magnitude', fontsize=17)
    ax_mag.tick_params(labelbottom=False)

    #plot magnitude
    ax_mag.scatter(measurement[0], measured_mag, marker='x', color='r')
    ax_mag.plot(real_measurement[0], real_mag, color='blue', linestyle='--')
    ax_mag.plot(fitted_data[0], fit_mag, color='black')
    ax_mag.set_title('Magnitude', fontsize=17)
    ax_mag.tick_params(axis='both', which='major', labelsize=15)


    #plot phase
    ax_phase.scatter(measurement[0], measured_pha, marker='x', color='r')
    ax_phase.set_ylabel('Phase', fontsize=17)
    ax_phase.plot(real_measurement[0], real_pha, color='blue', linestyle='--')
    ax_phase.plot(fitted_data[0], fit_pha, color='black')
    ax_phase.set_title('Phase', fontsize=17)
    ax_phase.tick_params(axis='both', which='major', labelsize=15)
    ax_phase.set_xlabel("Offset distance (mm)", fontsize=17)

    S = (S/S[0])
    S_sum = (S/sum(S))

    cmap = plt.cm.summer.reversed()

    im_r = ax_S.imshow(np.abs(np.atleast_2d(S).T),cmap=cmap, aspect='auto')

    tot = 0
    prob_modes = []

    for (j,i),label in np.ndenumerate(np.abs(np.atleast_2d(S).T)):
        ax_S.text(i,j,np.round(label, 5),ha='center',va='center', fontsize=17)
        tot += label
        if limit != False:
            if label < limit:
                rect = Rectangle(
                    (i-0.5, j-0.5),
                    1,
                    1,
                    fill=True,
                    edgecolor='red',
                    linewidth=5
                )
                ax_S.add_patch(rect)
                prob_modes.append(j)

    ax_S.set_yticks(range(len(S)), [str(i) for i, _ in enumerate(S)], fontstyle='italic', fontsize=17)

    for tick in ax_S.get_yticklabels():
        tick.set_bbox(dict(
            boxstyle="circle,pad=0.15",
            facecolor="none",
            edgecolor="black",
            linewidth=1.5
        ))
    ax_S.set_ylabel(f'Model modes', fontsize=17)
    ax_S.set_title('SVD (Σ/Σ[0])', fontsize=17)
    ax_S.set_xticks([])


    im_x = ax_U.imshow(np.abs(U).T)#[0:4, -1:-4:-1])
    ax_U.set_yticks(range(len(S)), [str(i) for i, _ in enumerate(S)], fontstyle='italic', fontsize=17)
    ax_U.set_xticks(range(len(measure_scheme.get_points())), [str(i) for i in np.round(measure_scheme.get_points(),2)], rotation=90, fontsize=17)
    for tick in ax_U.get_yticklabels():
        tick.set_bbox(dict(
            boxstyle="circle,pad=0.2",
            facecolor="none",
            edgecolor="black",
            linewidth=1.5
        ))

    ax_U.set_ylabel(f'Model modes', fontsize=17)
    ax_U.set_title('SVD (Out modes)', fontsize=17)




    V_abs = np.flip(np.abs(Vh), axis=0)

    im_y = ax_Vh.imshow(np.atleast_2d(V_abs))
    ax_Vh.set_title('SVD (In modes)', fontsize=17)

    if show_Vh_text:
        for (j,i),label in np.ndenumerate(V_abs):
            ax_Vh.text(i,j,np.round(label, 2),ha='center',va='center', fontsize=11)


    for row in prob_modes:
        # Column containing the maximum value in this row
        col = np.argmax(V_abs[row])

        # Draw a red rectangle around the maximum
        rect = Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            fill=False,
            edgecolor='red',
            linewidth=3
        )
        ax_Vh.add_patch(rect)

    ax_Vh.set_xticks(range(len(comps)), [str(i) for i in comps], rotation= 45, fontsize=17)
    ax_Vh.set_yticks(range(len(S)), [str(i) for i, _ in enumerate(S)], fontstyle='italic', fontsize=17)



    ax_Vh.set_ylabel(f'Model modes', fontsize=17)
    ax_Vh.set_xlabel(f'Model components', fontsize=17)

    for tick in ax_Vh.get_yticklabels():
        tick.set_bbox(dict(
            boxstyle="circle,pad=0.2",
            facecolor="none",
            edgecolor="black",
            linewidth=1.5
        ))

    ax = 0.0
    # U tick visibility
    for i, tick in enumerate(ax_U.get_xticklabels()):
    
        if i %3 != 0:
            tick.set_visible(False)

    for i, tick in enumerate(ax_U.get_yticklabels()):
        
        if i %mode_spacing != 0:
            tick.set_visible(False)

    # S tick visibility
    for i, tick in enumerate(ax_S.get_yticklabels()):
        
        if i %mode_spacing != 0:
            tick.set_visible(False)

    # Vh tick visibility
    for i, tick in enumerate(ax_Vh.get_xticklabels()):
        
        if i %comp_spacing != 0:
            tick.set_visible(False)

    for i, tick in enumerate(ax_Vh.get_yticklabels()):
        
        if i %mode_spacing != 0:
            tick.set_visible(False)

    return fit

def compare_build_vs_truncate(*,build_comps = [(1,0)], measurement_system, 
                    freq, measure_scheme_old, 
                    plot_scheme, truncate_comps, limit, N_limit, M_limit,
                    plotting_system, threshold, base_components):

    

    truncated_comps, new_scheme = truncate_measurement( comps=truncate_comps,
                                                       freq=freq, scheme=measure_scheme_old,
                                                       limit=limit, threshold=threshold, base_components=base_components)
    
    build_comps = build_model(components=build_comps, freq=freq, 
                                 scheme=measure_scheme_old, limit=limit,
                                 N_limit=N_limit, M_limit=M_limit, threshold=threshold)
    if type(freq) in (list, np.ndarray):
        
        freqs = [freq[0], freq[int(np.round(len(freq)/2))], freq[-1]]
        
        for f in freqs:
            fit_signal(measurement_system=measurement_system, freq= f, measure_scheme=new_scheme,
                    plot_scheme=plot_scheme, comps=truncated_comps, plotting_system=plotting_system, title=f'Truncation method \n {f} GHz')
            
            fit_signal(measurement_system=measurement_system, freq= f, measure_scheme=measure_scheme_old,
                    plot_scheme=plot_scheme, comps=build_comps, plotting_system=plotting_system, title=f'Additive Method \n {f} GHz')
        
    else:
        fit_signal(measurement_system=measurement_system, freq= freq, measure_scheme=new_scheme,
                plot_scheme=plot_scheme, comps=truncated_comps, plotting_system=plotting_system, title=f'Truncation method {freq} GHz')
            
        fit_signal(measurement_system=measurement_system, freq= freq, measure_scheme=measure_scheme_old,
                plot_scheme=plot_scheme, comps=build_comps, plotting_system=plotting_system, title=f'Additive Method {freq} GHz')

def printmodel(*, model):

    prev = model[0][0]

    for i, comp in enumerate(model):
        if comp[0] > prev:
            prev = comp[0]
            print()
        print(f'{comp},', end='')
        