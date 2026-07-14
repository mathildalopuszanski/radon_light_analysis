# this file is for plotting the light collected for alpha events in the entire TPC depending on the position
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import h5py
from h5flow.data import dereference 
import yaml
from scipy.optimize import curve_fit
import math
import os
import glob
import matplotlib
from matplotlib.gridspec import GridSpec
from DrawEvent import *
from BiPoSelection import *

# plt.style.use('/global/cfs/cdirs/dune/users/mlopuszanski/sty3.mplstyle')
# Set the default text font size
plt.rc('font', size=16)
# Set the axes title font size
plt.rc('axes', titlesize=18)
# Set the axes labels font size
plt.rc('axes', labelsize=18)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)
# Set the legend font size
plt.rc('legend', fontsize=16)
# Set the font size of the figure title
plt.rc('figure', titlesize=20)
# Colors for tpc
colors = [matplotlib.color_sequences['Paired'][i] for i in [0,1,2,3,4,5,8,9]]


# function to get the light collected for a alpha or beta in entire TPC
def get_tpc_integral_bipo(hit_pairs, sum_tpc_hits, light_events, ttype):
    """
    hit_pairs: hits in the same waveform
    light_events: light events that are BiPo selected (and have only one cluster)
    ttype: acl or lcm

    returns three arrays: PE for alpha hit, PE for beta hit and dt between the hits
    """
    alpha_pe = np.zeros(len(light_events))
    beta_pe = np.zeros(len(light_events))
    dt_selection = np.zeros(len(light_events)) 
    # now get the tpc hits
    first_hit_idx  = set(np.array([p[0] for p in hit_pairs], dtype=int))
    second_hit_idx = set(np.array([p[1] for p in hit_pairs], dtype=int))

    tpc_hit_integral = sum_tpc_hits['integral']
    events, hits, hits_per_event  = np.unique(sum_tpc_hits['refs'][:,0],
                                              return_index=True,
                                              sorted=False,
                                              return_counts=True)
    for i,e in enumerate(light_events):
        first_hit_in_event = hits[e]
        num_hits_in_event = hits_per_event[e]
        hits_in_event = np.arange(first_hit_in_event, 
                                  first_hit_in_event+num_hits_in_event, dtype=np.int64)

        for h in hits_in_event:
            if h in first_hit_idx: h1=h
            if h in second_hit_idx: h2=h
        

        # Get sample indices
        ticks = sum_tpc_hits['sample_idx'][hits_in_event]
        t1 = sum_tpc_hits['sample_idx'][h1]
        t2 = sum_tpc_hits['sample_idx'][h2]

        dt = t2 - t1

        # Compute distances to each reference hit
        dist_to_h1 = np.abs(ticks - t1)
        dist_to_h2 = np.abs(ticks - t2)

        if ttype =='acl': 
            ttype_mask = sum_tpc_hits['trap_type'][hits_in_event] == 0
        elif ttype == 'lcm': 
            ttype_mask = sum_tpc_hits['trap_type'][hits_in_event] == 1


        beta_mask = (dist_to_h1 <= dist_to_h2) & ttype_mask  # does the equal sign make sense here?
        alpha_mask = (dist_to_h2 < dist_to_h1) & ttype_mask

        beta_hits = hits_in_event[beta_mask]
        beta_pe[i] = np.sum(tpc_hit_integral[beta_hits])
        
        alpha_hits = hits_in_event[alpha_mask]
        alpha_pe[i] = np.sum(tpc_hit_integral[alpha_hits])

        dt_selection[i] = dt

    return beta_pe, alpha_pe, dt_selection



# this function is used to get all the data for the alpha and beta hit including CL matching
def get_results_per_particletype(runs, selection='single_bipo'):
    """runs is the dictionary with the file names
    returns x, y, z of cluster
    beta_acl, alpha_acl, beta_lcm, alpha_lcm: PE for alpha and beta hit per trap type
    tpc of light event
    dt_acl, dt_lcm: dt distributions per trap type"""
    
    all_x, all_y, all_z, all_pe = [], [], [], []
    all_beta_pe_lcm, all_alpha_pe_lcm, all_beta_pe_acl, all_alpha_pe_acl = [], [], [], []
    all_tpc, all_dt_acl, all_dt_lcm  = [], [], [] 
    num_all_multiple = 0 
    num_bipo_multiple = 0 
    num_CL_matched_multiple = 0 
    num_CL_bipo_multiple = 0 
    num_CL_bipo_single_tpc_multiple = 0 


    CL_matched_files = [f'processed_files/CL_{run["name"]}.npz' for run in runs]
    hit_data_tpc_files = [f'{run["tpc_light"]}' for run in runs]
    
    for CL_file, tpc_light_file, run in zip(CL_matched_files, hit_data_tpc_files, runs):
        # --- load files ---
        d = np.load(CL_file, allow_pickle=True)
        light_id_to_idx = {lid: i for i, lid in enumerate(d['light_id'])}
        num_CL_matched = len(np.unique(d['light_id']))
        

        tpc_hit_data = np.load(tpc_light_file, 
                        allow_pickle=True).item() 
        num_all = len(np.unique(tpc_hit_data['refs'][:,0])) # counting
        
        hit_pairs, events_with_nhits, dt_list = calculate_dt(tpc_hit_data, tpc_hit_data['refs'], tpc_or_ch_sum='tpc')
        # print(f'Found {len(events_with_nhits)} light events with multiple hits in one wvfm')
        num_bipo = len(np.unique(events_with_nhits)) # counting

        # --- selection --- 
        # choose between single cluster matched to event, BiPo and single cluster BiPo
        if selection == 'single':
            mask = d['num_cl'] == 1
        elif selection == 'bipo':
            mask = d['is_bipo']
        elif selection == 'single_bipo':
            mask = d['is_bipo'] & (d['num_cl'] == 1)

        selected_ids = d['light_id'][mask]
        num_CL_bipo = len(np.unique(selected_ids)) # counting

        # filter single tpc events
        single_tpc_mask = np.array([
                len(d['tpc'][light_id_to_idx[lid]]) == 1
                for lid in selected_ids
            ])
        selected_ids = selected_ids[single_tpc_mask]  # stays a numpy array, ordered
        num_CL_bipo_single_tpc = len(np.unique(selected_ids))

        # rebuild idx dict for the filtered ids only
        light_id_to_idx = {lid: i for i, lid in enumerate(d['light_id']) 
                        if lid in selected_ids}
        

        # --- cluster mask ---
        cluster_mask = np.array([lid in selected_ids for lid in d['cluster_light_ref']])


        # --- PEs ---
        beta_pe_acl, alpha_pe_acl, dt_selection_acl = get_tpc_integral_bipo(hit_pairs, tpc_hit_data, 
                                    light_events=selected_ids, ttype='acl')

        beta_pe_lcm, alpha_pe_lcm, dt_selection_lcm= get_tpc_integral_bipo(hit_pairs, tpc_hit_data, 
                                    light_events=selected_ids, ttype='lcm')
        
        # --- append ---
        all_x.append(d['cluster_x_rec'][cluster_mask])
        all_y.append(d['cluster_y'][cluster_mask])
        all_z.append(d['cluster_z'][cluster_mask])
        all_beta_pe_acl.append(beta_pe_acl)
        all_alpha_pe_acl.append(alpha_pe_acl)
        all_beta_pe_lcm.append(beta_pe_lcm)
        all_alpha_pe_lcm.append(alpha_pe_lcm)
        all_tpc.append(np.array([d['tpc'][light_id_to_idx[lid]][0]  # scalar since single TPC
                                for lid in d['cluster_light_ref'][cluster_mask]]))
        all_dt_acl.append(dt_selection_acl)
        all_dt_lcm.append(dt_selection_lcm)
        
        
        print(f'Run {run["name"]} with {num_all} events')
        print(f'BiPo {num_bipo}')
        print(f'CL matched {num_CL_matched}')
        print(f'CL matched BiPo {num_CL_bipo}')
        print(f'CL BiPo single tpc {num_CL_bipo_single_tpc}')

        num_all_multiple += num_all
        num_bipo_multiple += num_bipo
        num_CL_matched_multiple += num_CL_matched   
        num_CL_bipo_multiple += num_CL_bipo   
        num_CL_bipo_single_tpc_multiple += num_CL_bipo_single_tpc

    # concatenate only the final arrays needed for plotting
    x  = np.concatenate(all_x)
    y  = np.concatenate(all_y)
    z  = np.concatenate(all_z)
    beta_acl = np.concatenate(all_beta_pe_acl)
    alpha_acl = np.concatenate(all_alpha_pe_acl)
    beta_lcm = np.concatenate(all_beta_pe_lcm)
    alpha_lcm = np.concatenate(all_alpha_pe_lcm)
    tpc = np.concatenate(all_tpc)
    dt_acl = np.concatenate(all_dt_acl)
    dt_lcm = np.concatenate(all_dt_lcm)

    # print(f'Total: {total_bipo} events across {len(files)} files')

    # print(f'In {len(runs)} runs: {num_all_multiple} events')
    # print(f'BiPo {num_bipo_multiple}')
    # print(f'CL matched {num_CL_matched_multiple}')
    # print(f'CL matched BiPo {num_CL_bipo_multiple}')
    # print(f'CL BiPo single tpc {num_CL_bipo_single_tpc_multiple}')

    return x, y, z, beta_acl, alpha_acl, beta_lcm, alpha_lcm, tpc, dt_acl, dt_lcm


# function to annotate the position of cathode, anode, LRS etc.
def annotate_detector_geometry(
    axs,
    x_bounds,
    z_bounds,
    det_y_bins,
    *,
    # which gaps in x_bounds contain a cathode (0-indexed pairs)
    cathode_gap_indices=((1, 2), (5, 6)),
    # one label per gap between consecutive det_y_bins; consecutive same-type are merged
    y_pattern=('ACL', 'LCM', 'LCM', 'LCM', 'ACL', 'LCM', 'LCM', 'LCM'),
    cathode_width=1.0,       # visual width [cm] of orange band
    lrs_width=.5,           # visual width [cm] of green band
    bounds_ymax=85,          # height of the black dashed vlines (data coords)
    bar_height_frac=1.03,    # ACL/LCM bars drawn at y_max * this fraction
    fontsize=14,
):
    """
    Draw all detector-geometry overlays on a 3-panel (x, z, y) figure.

    Positions are derived entirely from x_bounds / z_bounds / det_y_bins,
    so there is one source of truth and nothing is hardcoded.

    Parameters
    ----------
    axs : (3,) array of Axes   — axs[0]=x, axs[1]=z, axs[2]=y
    x_bounds, z_bounds, det_y_bins : array-like
        Same arrays you would otherwise pass to vlines.
    cathode_gap_indices : sequence of (i, j)
        0-indexed index-pairs into x_bounds; cathode centred at their midpoint.
    y_pattern : sequence of 'ACL' | 'LCM'
        One entry per gap in det_y_bins. Consecutive identical labels are
        merged into a single coloured bar.
    bounds_ymax : float
        Manually adjustable height for the black dashed boundary lines.
    bar_height_frac : float
        Manually adjustable vertical position of the ACL/LCM bars.
    """
    ax_x, ax_z, ax_y = axs
    x_bounds   = np.asarray(x_bounds)
    z_bounds   = np.asarray(z_bounds)
    det_y_bins = np.asarray(det_y_bins)

    _, y_max = ax_x.get_ylim()
    y_max=bounds_ymax

    # ── black boundary vlines (replaces the standalone calls) ──────────────
    vline_kw = dict(ymin=0, ymax=bounds_ymax,
                    colors='black', alpha=0.5, linestyle='--')
    ax_x.vlines(x_bounds,   **vline_kw)
    ax_z.vlines(z_bounds,   **vline_kw)
    ax_y.vlines(det_y_bins, **vline_kw)


    # ── x panel: cathode bands ─────────────────────────────────────────────
    for i, j in cathode_gap_indices:
        xc = 0.5 * (x_bounds[i] + x_bounds[j])
        ax_x.fill_betweenx([0, bounds_ymax],
                   xc - cathode_width / 2, xc + cathode_width / 2,
                   color='tab:orange', alpha=0.3, zorder=-1)
        ax_x.text(xc, y_max * 1.1, 'Cathode',
                  ha='center', va='top', color='tab:orange', fontsize=fontsize)

    # ── z panel: LRS band at every z bound ────────────────────────────────
    for zc in z_bounds:
        ax_z.fill_betweenx([0, bounds_ymax],
                   zc - lrs_width / 2, zc + lrs_width / 2,
                   color='tab:green', alpha=0.3, zorder=-1)
    ax_z.text(z_bounds[-1] + 1.5, y_max * 1.1, 'LRS',
              ha='left', va='top', color='tab:green', fontsize=fontsize)

    # ── y panel: ACL / LCM bars ────────────────────────────────────────────
    n_gaps = len(det_y_bins) - 1
    if len(y_pattern) != n_gaps:
        raise ValueError(
            f"y_pattern needs {n_gaps} entries (one per gap in det_y_bins), "
            f"got {len(y_pattern)}."
        )

    # merge consecutive same-type regions into one bar
    merged = []
    for k, label in enumerate(y_pattern):
        y0, y1 = det_y_bins[k], det_y_bins[k + 1]
        if merged and merged[-1][0] == label:
            merged[-1][2] = y1          # extend right edge
        else:
            merged.append([label, y0, y1])

    color_map = {'ACL': 'blue', 'LCM': 'red'}
    y_bar = y_max * bar_height_frac

    for label, y0, y1 in merged:
        c = color_map[label]
        ax_y.hlines(y_bar, y0, y1, colors=c, linewidth=4, clip_on=False, alpha=0.5)
        ax_y.text(0.5 * (y0 + y1), y_bar * 1.01, label,
                  color=c, ha='center', va='bottom', fontsize=fontsize)

    # Bottom / Top labels
    x_lo, x_hi = ax_y.get_xlim()
    ax_y.text(x_lo, 0, 'Bottom', ha='left', va='top',
              transform=ax_y.get_xaxis_transform(), fontsize=fontsize - 2)
    ax_y.text(x_hi,0, 'Top',    ha='right', va='top',
              transform=ax_y.get_xaxis_transform(), fontsize=fontsize - 2)


### --- FILE PATH ---
# directory to store the plots
plots_dir = ''
# light files (h5 files)
CL_path = '' # /global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_CLmatching_low_energy/flowed_light_low_energy_v3/source_rn_bin1/injection/'
# light files (h5 files)
light_path = '' # '/global/cfs/cdirs/dune/users/mlopuszanski/Rn_injection/results/'

# runs to use for the plots
# run_i = list(range(1, 21)) + list(range(120, 141)) + list(range(241, 260))
run_i = list(range(1, 21))
run_names = [f'rctl_775_p{i}' for i in run_i]  # p1 to p5

runs = build_runs(run_names, CL_path, light_path)
print(f'Found {len(runs)} complete runs')



### --- GEOMETRY ---

#We can load the detector center position and the SiPM position from the geometry 
lrs_geometry_file = '/global/cfs/cdirs/dune/users/mlopuszanski/ndlar_flow/data/proto_nd_flow/light_module_desc-5.0.1.yaml'
with open(lrs_geometry_file) as gf:
    lrs_geometry_yaml = yaml.load(gf, Loader=yaml.FullLoader)

tpc_ids =  np.array([v for v in lrs_geometry_yaml['tpc_center_offset'].keys()])
det_ids = np.array([v for v in lrs_geometry_yaml['det_center'].keys()])


det_type_array = np.zeros((len(tpc_ids), len(det_ids)), dtype=np.int32)
for i, tpc in enumerate(tpc_ids):
    for j, det in enumerate(det_ids):
        det_type = lrs_geometry_yaml['det_geom'][tpc][det]
        det_type_array[i,j] = det_type

CL_file = '/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_CLmatching_low_energy/flowed_light_low_energy_v3/source_rn_bin1/injection/packet-0060156-2025_11_20_20_06_56_CST_mpd_run_data_rctl_775_p1_CLmatched.FLOW_LE.hdf5'
tpc_bounds = extract_tpc_bounds(h5py.File(CL_file))

tpc=0 # for y tpc doesnt matter
bounds = tpc_bounds[tpc]
det_position_y = np.zeros((16, 2), dtype=np.float64)
for det_id in sorted(lrs_geometry_yaml['det_center'].keys(), key=int):
    tpc_shift = np.mean(bounds, axis=0)
    plate_corners = get_plate_corners(int(det_id), tpc_shift, lrs_geometry_yaml)
    det_position_y[det_id, 0] = plate_corners[0, 1]
    det_position_y[det_id, 1] = plate_corners[-1, 1]


det_y_bins = np.append(det_position_y[:8, 0], det_position_y[7,1])
x_bounds = np.unique(np.round(tpc_bounds[:,:,0].flatten()))
z_bounds = np.unique(np.round(tpc_bounds[:,:,2].flatten()))
y_bounds = np.unique(np.round(tpc_bounds[:,:,1].flatten()))


### --- PROCESS FILES --- 
x, y, z, beta_acl, alpha_acl, beta_lcm, alpha_lcm, tpcs_of_event, dt_acl, dt_lcm = get_results_per_particletype(runs[:], selection='single_bipo')


### --- Divide PE to the PE/MeV ---
# divide by 4 because of ADC gain 
# this should be done earlier but it was not unfortunately 
beta_acl = beta_acl/4
beta_lcm = beta_lcm/4
alpha_acl = alpha_acl/4
alpha_lcm = alpha_lcm/4

# dividing by 9 because to get units of PE/MeV_alpha
alpha_acl = alpha_acl/9.0
alpha_lcm = alpha_lcm/9.0


### ---- PLOTS ----

## DEFINE BINS
bins_x = np.linspace(min(x_bounds), max(x_bounds), 25)
bins_z = np.linspace(min(z_bounds), max(z_bounds), 25)
bins_y = np.linspace(min(y_bounds), max(y_bounds), 25)


## --- Averages for all TPCS ---

fig, axs = plt.subplots(3, 1, figsize=(12,12), sharey='row')
axs = axs.flatten()

print('ACL averages X')
plot_avg_pe_vs_position(axs[0], x,  alpha_acl, bins_x, color='blue', label='ACL')
print('ACL averages Z')
plot_avg_pe_vs_position(axs[1], z,    alpha_acl, bins_z, color='blue', label='ACL')
print('ACL averages Y')
plot_avg_pe_vs_position(axs[2], y,    alpha_acl, bins_y, color='blue', label='ACL')

print('LCM averages X')
plot_avg_pe_vs_position(axs[0], x, alpha_lcm, bins_x, color='red', label='LCM')
print('LCM averages Z')
plot_avg_pe_vs_position(axs[1], z,   alpha_lcm, bins_z, color='red', label='LCM')
print('LCM averages Y')
plot_avg_pe_vs_position(axs[2], y,   alpha_lcm, bins_y, color='red', label='LCM')


alpha_pe = np.add(alpha_lcm, alpha_acl)
print('LCM+ACL averages X')
plot_avg_pe_vs_position(axs[0], x,  alpha_pe, bins_x, color='black', label='ACL+LCM')
print('LCM+ACL averages Z')
plot_avg_pe_vs_position(axs[1], z,    alpha_pe, bins_z, color='black', label='ACL+LCM')
print('LCM+ACL averages Y')
plot_avg_pe_vs_position(axs[2], y,    alpha_pe, bins_y, color='black', label='ACL+LCM')


annotate_detector_geometry(axs, x_bounds, z_bounds, det_y_bins, bounds_ymax=85)



for ax in axs:
    ax.set_ylabel(r'PE / MeV$_\alpha$')
    ax.grid(True, which='both', ls='--', lw=0.5)
    ax.set_ylim(0, 85+10)

axs[0].legend(loc='upper left', ncols=3, bbox_to_anchor=(0, 1.2))
axs[0].set_xlabel("Reconstructed x [cm]")
axs[1].set_xlabel("z [cm]")
axs[2].set_xlabel("y [cm]")
    

plt.suptitle(r'Detected PE / MeV$_\alpha$ per TPC for 9MeV alpha', y=0.97)
plt.tight_layout()
plt.savefig(f'{plots_dir}/PE_per_tpc_LCM_ACL_seperate_perMeV.png')
