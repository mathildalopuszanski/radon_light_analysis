### In this file there a functions that I used frequently for the BiPo selection 


import numpy as np
import h5py
from h5flow.data import dereference 
import glob
import os
from scipy.optimize import curve_fit
import math
import matplotlib.pyplot as plt

# this function can be used to get the hits associated with a event
def get_hits_in_event(event, hit_event_refs):
    """
    event: event you want the associated hits for
    hit_event_refs: refs in 2d array [[e, h], []]
    """
    _, hits, hits_per_event  = np.unique(hit_event_refs[:,0], return_index=True,sorted=False, return_counts=True)
    hit = hits[event]
    num_hits = hits_per_event[event]
    hits_in_event = np.linspace(hit, hit+(num_hits-1), num_hits, dtype=np.int64)
    return hits_in_event, num_hits

# function that can be used to print the items in a h5 file
def h5_tree(val, pre='',skip_ref=False):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5py._hl.group.Group:
                print(pre + '└── ' + key)
                if key=="ref" and skip_ref:
                    continue
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5py._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))

# the function that is used to find the 
def calculate_dt(hits, hits_refs, tpc_or_ch_sum, dt_min=8, dt_max=125):
    """
    Function that finds multiple hits in either tpc or sum waveforms and applies cut if they are to close or to far
    hits: dictionary with information about hit (tpc, det, etc)
    hits_refs: refs of events to hits (2d array with [event, hit])
    tpc_or_ch_sum: can be 'tpc' to find BiPos in tpc waveforms or 'ch' to find them in trap waveforms
    dt_min, dt_max are given in ticks

    returns:
    hit_pairs: list with the pairs of hits [[hit1, hit2], [hit1, hit2], ...]
    events_with_nhits: list of events that have multiple hits 
    dt_list: time differences between the hits
    all list have the same length so you can associate the hit pair to an event and the dt between the two hits
    """
    # need to know the tpc and trap_type to check if the hits are in the same waveform
    if tpc_or_ch_sum=='tpc':
        tpc_of_hit=hits['tpc']
        det_of_hit=hits['trap_type']
    elif tpc_or_ch_sum=='ch':
        tpc_of_hit=hits['tpc']
        det_of_hit=hits['det']
    sample_idx_of_hit=hits['sample_idx']

    events_with_nhits = []
    hit_pairs = []
    dt_list = []
    
    # looping over all events and getting consectutive hits
    for prev_ref, curr_ref in zip(hits_refs, hits_refs[1:]):
        prev_event=prev_ref[0]
        prev_hit=prev_ref[1]
        curr_event=curr_ref[0]
        curr_hit=curr_ref[1]

        # check if they are in the same event
        if prev_event == curr_event:
            # check if they are in the same waveform
            if tpc_of_hit[prev_hit] == tpc_of_hit[curr_hit] and det_of_hit[prev_hit] == det_of_hit[curr_hit]:
                # calculate dt of the two hits
                prev_t = sample_idx_of_hit[prev_hit]
                curr_t = sample_idx_of_hit[curr_hit]

                if dt_min <= (curr_t - prev_t) <= dt_max: # cut on distance between hits
                    dt = (curr_t - prev_t) * 16.0
                    dt_list.append(dt)
                    hit_pairs.append([int(prev_hit), int(curr_hit), dt])
                    # store events with multiple hits in one waveform
                    events_with_nhits.append(curr_event)

    return hit_pairs, events_with_nhits, dt_list


# function to calculate the fpromt on a hit (similar to the one in ndlar-flow)
def calculate_fprompt(wvfm, peak, prompt_window_ns=200.0, long_window_ns=3200.0, n_samples=400):
    """
    wvfm: one waveform
    peak: position of the hit in the waveform 
    prompt_window_ns, long_window_ns: short and long integration window in ns 
    n_samples: length of a waveform in ticks
    """
    tick_duration_ns = 16.0 
    
    prompt_bins = int(np.ceil(prompt_window_ns / tick_duration_ns))
    total_bins  = int(np.ceil(long_window_ns   / tick_duration_ns))

    t0_bin = peak - np.minimum(5, peak)
    start_idx = np.clip(t0_bin, 0, n_samples)
    end_prompt = np.clip(t0_bin + prompt_bins, 0, n_samples)
    end_total  = np.clip(t0_bin + total_bins,  0, n_samples)

    prompt = np.sum(wvfm[start_idx : end_prompt])
    total = np.sum(wvfm[start_idx : end_total])

    fprompt = prompt / total
    
    return fprompt, total, [start_idx, end_prompt, end_total]


# function to get the light events that are charge light matched
def get_reconstructed_position(file):
    """
    file: location of a h5 charge-light matched file 
    returns a dictionary of the matched events
    each event is a dictionary that contains the cluster information (position, id, charge)
    """
    with h5py.File(file, "r") as f:
        # all the light events that have a matched charge cluster
        light_events_idx = np.array(
            f['charge/clusters_matched/ref/light/events/ref'][:, 1]
        )
        # use unique to get rid of duplicate light events
        # vals is the light event index
        # start_idx is the index in the matched sets so can be used to relate a light event to 
        # end_idx is last occurence of a light event in the light event array
        vals, start_idx, counts = np.unique(light_events_idx, return_index=True, return_counts=True)
        end_idx = start_idx + counts  # exclusive end index, so use [s:e] directly

        # dictionary of light event and indeces in cluster related to that light event
        index_ranges = {
            v: (int(s), int(e))
            for v, s, e in zip(vals, start_idx, end_idx)
            }

        matched_events = {}

        for key, (s, e) in index_ranges.items():
            # clusters for one event
            # light event = key
            clusters = np.array(f['charge/clusters_matched/data'][s:e])
            charge = clusters['Q'][:]
            # charge = clusters['Q']
            id = clusters['id'][:]
            x = clusters['x'][:,1]
            y = clusters['y_pix'][:,1]
            z = clusters['z_pix'][:,1]
            x_anode = clusters['x_pix'][:,1]

            event = {
                'x': x,
                'y': y,
                'z': z,
                'x_anode': x_anode,
                'charge': charge,
                'chargeID': id
            }

            matched_events.update({key: event})

        # time length of one file 
        utime = f['light/events/data']['utime_ms']
        length = (np.max(utime) - np.min(utime)) / 1e3 # convert to seconds
        
        return matched_events


# function to make the paths to the CL and light files given a run
def build_runs(run_names, CL_path, light_path):
    """
    run_names: list of runs for example 'rctl_775_p1' 
    CL_path: location of the CL matched files
    light_path: location of light files 
    returns dictionary with full path to all files (CL, light h5 file and light npy files)
    """
    runs = []
    for name in run_names:
        # CL file has a variable timestamp prefix, so use glob to find it
        cl_matches = glob.glob(f'{CL_path}*{name}_CLmatched.FLOW_LE.hdf5')
        if len(cl_matches) == 0:
            print(f'WARNING: no CL file found for {name}, skipping')
            continue
        elif len(cl_matches) > 1:
            print(f'WARNING: multiple CL files found for {name}: {cl_matches}')
            print(f'         using {cl_matches[0]}')

        runs.append(dict(
            name      = name,
            CL        = cl_matches[0],
            sum_light = f'{light_path}mpd_run_data_{name}.FLOW_sum_hits_results.npy',
            tpc_light = f'{light_path}mpd_run_data_{name}.FLOW_tpc_hits_results.npy',
        ))

        # check the light files exist too
        for key in ['sum_light', 'tpc_light']:
            if not os.path.exists(runs[-1][key]):
                print(f'WARNING: {key} file not found for {name}: {runs[-1][key]}')

    return runs




###--- Functions for plotting --- 

def plot_avg_pe_vs_position(ax, positions, pe_values, bins, color='blue', label='', include_0=True, fmt='o'):
    """Bin positions and plot mean PE per bin with std error bars."""
    bin_indices = np.digitize(positions, bins)
    bin_means = []
    bin_errors = []
    bin_centers = []
    for i in range(1, len(bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(pe_values[mask].mean())
            bin_errors.append(pe_values[mask].std() / np.sqrt(mask.sum()))  # standard error
            # bin_errors.append(pe_values[mask].std()) # std
            bin_centers.append((bins[i-1] + bins[i]) / 2)

        # include 0 bins
        elif include_0==True:
            bin_means.append(0)
            bin_errors.append(0)
            bin_centers.append((bins[i-1] + bins[i]) / 2)
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_errors = np.array(bin_errors)
    ax.errorbar(bin_centers, bin_means, yerr=bin_errors, fmt='o', color=color,
                markersize=4, capsize=3, label=label)
    

    max_idx = np.argmax(bin_means)
    min_idx = np.argmin(bin_means)

    print(
        f"{label}\n"
        f"  max:  {bin_means[max_idx]:.1f} ± {bin_errors[max_idx]:.1f}\n"
        f"  min:  {bin_means[min_idx]:.1f} ± {bin_errors[min_idx]:.1f}\n"
        f"  mean: {np.average(bin_means):.1f} ± {np.std(bin_means):.1f}"
    )


###--- Making histograms ---
def make_hist(data, mask, bins):
    hist, _ = np.histogram(data[mask], bins=bins)
    return hist

def draw_hist(ax, data, bins, label='label', **kwargs):
    hist, _ = np.histogram(data, bins=bins)
    # hist = hist/len(data)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ax.step(bin_centers, hist, where='mid', label=label,alpha=0.7,**kwargs)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(True, which='both', ls='--', lw=0.5)


def make_2dhist(x, y, bins=[100, 100], range=[[0,1e4], [0,1.5]]):
    h, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
    return h, xedges, yedges


def plot_2dhist(ax, fig, cmap, x, y, bins=[100, 100], range=[[0,1e4], [0,1.5]]):
    h, xedges, yedges = make_2dhist(x, y, bins=bins, range=range)
    pcm = ax.pcolormesh(xedges, yedges, h.T, 
                        cmap=cmap,
                        norm="log")
    fig.colorbar(pcm, ax=ax, label="", pad=0)
    ax.grid(True, which='both', ls='--', lw=0.5)



###--- dt Distribution ---

# fit the decay after the peak
def exp_fit(t, A, tau, C):
        # y = A * np.exp(-1/tau*t) + C
        y = A * 0.5**(t/tau) + C
        return y
    
def delta_t_fit(delta_t, bin_size=64, start_fit=130, end_fit=2000, p0=[100, 300, 0]):
    len_wvfm = 400 * 16
    bins = np.arange(0, len_wvfm, bin_size) # bin the whole length of the waveform
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    # create the histogram of delta t
    hist, _ = np.histogram(delta_t, bins)

    start_fit = math.ceil((start_fit/bin_size)) # math.ceil rounds upwards
    end_fit = math.floor((end_fit/bin_size)) # math.floor rounds down

    x = 0.5*bin_size+ bins[start_fit-1:end_fit] # x values to fit 
    y = hist[start_fit-1:end_fit] # y values for fit
    mask = y > 0
    xfit = x[mask]
    yfit = y[mask]

    sigma_y = np.sqrt(yfit)
    # get the second max value
    p0 = [100, 300, 0]
    fitted_par, pcov = curve_fit(exp_fit, xfit, yfit, p0,
                                sigma=sigma_y,      # for weighted fit
                                absolute_sigma=True)


    # chi squared 
    # Calculate parameter uncertainty.
    sigma_p1 = [np.absolute(pcov[i][i])**0.5 for i in range(len(fitted_par))]
    # Calculate chisquare value and degrees of freedom
    chi2 = np.sum((exp_fit(xfit, *fitted_par) - yfit)**2 / sigma_y**2)
    print('chi2', chi2)
    ndof = len(yfit) - len(fitted_par)
    chi2_red = chi2 / ndof
    print('reduced chi2', chi2_red)

    return fitted_par, pcov, [hist, bins, xfit, yfit]


def plot_delta_t_fit(ax, delta_t, bin_size=64, start_fit=130, end_fit=2000, 
                     p0=[100, 300, 0], plot_directory='name'):

    fitted_par, pcov, hist_data = delta_t_fit(delta_t, bin_size=bin_size, start_fit=start_fit, end_fit=end_fit, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    A, tau, C = fitted_par
    sA, sTau, sC = perr

    units='ns'

    label = (
        rf"$\mathrm{{Exponential\ fit:}}$" "\n"
        rf"$\mathrm{{Theoretical\ }} t_{{1/2}} = 294 \ \mathrm{{{units}}}$" "\n"
        rf"$N_0 = {A:.2f} \pm {sA:.2f}$" "\n"
        rf"$t_{{1/2}} = {tau:.2f} \pm {sTau:.2f}\ \mathrm{{{units}}}$" "\n" 
        rf"$C = {C:.2f} \pm {sC:.2f}$" "\n"
        )
    hist = hist_data[0]
    bins = hist_data[1]
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    xfit = hist_data[2]
    yfit = hist_data[3]

    fit_eq = exp_fit(bin_centers, fitted_par[0],fitted_par[1], fitted_par[2])
    
    ax.plot(bin_centers, fit_eq, 
            color = 'red', label=f'{label}', alpha=0.8, zorder=3)
    ax.errorbar(xfit, yfit, yerr=0,
                fmt='o', color='black', markersize=4, capsize=1, alpha=0.7, 
                label='fitted data points', zorder=1)
    ax.step(bin_centers, hist, 
            alpha=1, where='mid', color='grey', zorder=2)
    
    ax.set_xlabel(f"delta t ({units})")
    ax.set_ylabel("Counts")
    ax.set_xlim(-1,3000)
    ax.legend()
    ax.grid(True, lw=0.5)
    plt.savefig(f'{plot_directory}')




###--- Fprompt plots ---

# Gaussian fit
def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def gaussian_fit(data, bins=np.linspace(0, 1, 100), start_fit=0, end_fit=-1, p0=[1, 0.5, 0.1]):
    """
    p0 = [A, mu, sigma]
    """
    # create histogram
    hist, _ = np.histogram(data, bins)
    bin_centers = 0.5*(bins[1:]+bins[:-1])

    x = bin_centers[start_fit:end_fit]
    y = hist[start_fit:end_fit]

    mask = y > 0
    xfit = x[mask]
    yfit = y[mask]

    sigma_y = np.sqrt(yfit)

    # fit gaussian
    fitted_par, pcov  = curve_fit(gauss, xfit, yfit, p0)
                            # sigma=sigma_y,
                            # absolute_sigma=True) ### need to double check this
    print("Fitted parameters:", fitted_par)
    
    # Calculate chisquare value and degrees of freedom
    chi2 = np.sum((gauss(xfit, *fitted_par) - yfit)**2 / sigma_y**2)
    print('chi2', chi2)
    ndof = len(yfit) - len(fitted_par)
    chi2_red = chi2 / ndof
    print('reduced chi2', chi2_red)

    return fitted_par, pcov, [chi2, ndof], [hist, bins, xfit, yfit]


def plot_gauss_fit(ax, data, bins=np.linspace(0, 1, 100), 
                   start_fit=0, end_fit=100, p0=[600, 0.6, 0.03]):
    
    fitted_par, pcov, chi2_ndof, hist_data = gaussian_fit(data=data, bins=bins, start_fit=start_fit, end_fit=end_fit, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    A, mu, sigma = fitted_par
    sA, smu, ssigma = perr

    label = (
        rf"$\mathrm{{Gaussian\ fit:}}$" "\n"
        rf"$A = {A:.2f} \pm {sA:.2f}$" "\n"
        rf"$\mu = {mu:.2f} \pm {smu:.2f} $" "\n" 
        rf"$\sigma = {sigma:.2f} \pm {ssigma:.2f}$" "\n"
        rf"$\chi^2/ndof = {chi2_ndof[0]:.2f} / {chi2_ndof[1]:.2f}$" "\n" 
        )


    hist = hist_data[0]
    bins = hist_data[1]
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    xfit = hist_data[2]
    yfit = hist_data[3]

    # fit_eq = exp_fit(bin_centers, fitted_par[0],fitted_par[1], fitted_par[2])
    fit_eq = gauss(bin_centers,fitted_par[0],fitted_par[1], fitted_par[2])
    
    
    ax.step(bin_centers, hist, 
            alpha=1, where='mid', color='grey', zorder=2)
    ax.plot(bin_centers[start_fit:end_fit], fit_eq[start_fit:end_fit], 
            color = 'red', label=f'{label}', alpha=0.8, zorder=3)
    ax.errorbar(xfit, yfit, 
                # yerr=np.sqrt(yfit),
                fmt='o', color='black', markersize=4, capsize=1, alpha=0.7, 
                label='fitted data points', zorder=1)

    ax.set_xlabel(f"Fprompt")
    ax.set_ylabel("Counts")
    ax.set_xlim(0,1)
    ax.legend()
    ax.grid(True, lw=0.5)  
