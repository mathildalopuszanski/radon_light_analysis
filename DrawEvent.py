# functions that can be used to draw a TPC from different sides and plotting waveforms

import matplotlib.pyplot as plt
import numpy as np

# drawing tpc from top (xz plane)
def draw_top_view(ax, bounds, tpc_id, n_hits=None, **kwargs):
    lower, upper = np.array(bounds[0]), np.array(bounds[1])
    x0, y0, z0 = lower
    x1, y1, z1 = upper

    # Detector view: x vs y
    rect = plt.Rectangle(
        (z0, x0),
        z1 - z0,
        x1 - x0,
        alpha=1, 
        edgecolor='black',
        **kwargs
    )

    ax.add_patch(rect)

    if n_hits==None:
        if tpc_id % 2 == 0: 
            x_pos = (x1 + 1.5)
            z_pos = z0 + 5
        else: 
            x_pos = (x0 - 1.5)
            z_pos = z1 - 5
        ax.text(
            z_pos, x_pos, f'TPC {tpc_id}',
            ha='center', va='center',
            fontsize=9
        )
    else:
        z_pos = z0+0.5*(z1 - z0)
        x_pos = x0+0.5*(x1 - x0)
        ax.text(
            z_pos, x_pos, f'TPC {tpc_id} \n Hits{n_hits}',
            ha='center', va='center',
            fontsize=12
        )

    ax.set_ylabel("x (drift) [cm]")
    ax.set_xlabel("z (beam) [cm]")

# drawing the tpc from the side in the yz plane
def draw_tpc_side(ax, bounds, tpc_id, set_axis_limit=False, **kwargs):
    lower, upper = np.array(bounds[0]), np.array(bounds[1])
    x0, y0, z0 = lower
    x1, y1, z1 = upper

    # Detector view: x vs y
    rect = plt.Rectangle(
        (z0, y0),
        z1 - z0,
        y1 - y0,
        fill=False,
        alpha=1, 
        edgecolor='grey',
        **kwargs
    )

    ax.add_patch(rect)

    if set_axis_limit==True:
        ax.set_xlim(-(abs(z0)+10), z1+10)
        ax.set_ylim(-(abs(y0)+2), y1+2)

    # label the tpcs
    z_pos = z0+0.5*(z1 - z0)
    y_pos = y1+2

    ax.text(
        z_pos, y_pos, f'TPC {tpc_id}',
        ha='center', va='center',
        fontsize=10
    )

# compute the geometry of the detectors
def get_plate_corners(det_id, tpc_shift, geom_dict):
    #det_geom = geom_dict['det_geom']
    shape_key = 0 if (det_id % 4) == 0 else 1 #Didnt have a better plan, but we know ACLs are every modulo 4 
    offs_min  = np.array(geom_dict['geom'][shape_key]['min'], float)
    offs_max  = np.array(geom_dict['geom'][shape_key]['max'], float)

    xmin,ymin,_ = offs_min
    xmax,ymax,_ = offs_max
    # rectangle in the xy plane -> that is size of the det type
    local_rect  = np.array([[xmin,ymin, 0],[xmax,ymin, 0],
                            [xmax,ymax, 0],[xmin,ymax, 0]])

    det_ctr_local = np.array(geom_dict['det_center'][det_id], float) # relative position of det on tpc
    # add the position of the tpc -> center of tpc and position of det type on the tpc
    return local_rect + det_ctr_local + tpc_shift


# drawing detectors into the tpc 
def draw_det_outline(ax, corners, det_id, plane='xy', **kwargs):
    p1, p2, p3, p4 = corners
    if plane == 'xy':
        rect = plt.Polygon([p1[:2], p2[:2], p3[:2], p4[:2]], 
                        edgecolor='grey',
                        alpha=0.7,
                        **kwargs)
        ax.add_patch(rect)
        # label det id
        cx = np.mean([p1[0], p2[0], p3[0], p4[0]])
        cy = np.mean([p1[1], p2[1], p3[1], p4[1]])
        ax.text(
            cx, cy, str(det_id),
            ha='center', va='center',
            fontsize=7
        )
        ax.set_ylabel("y (height) [cm]")
        ax.set_xlabel("x (drift) [cm]")


    elif plane == 'zy': 
        cz = np.mean([p1[2], p2[2], p3[2], p4[2]])
        cy = np.mean([p1[1], p2[1], p3[1], p4[1]])
        corners = [[cz-1, p1[1]], [cz+1, p2[1]], [cz+1, p3[1]], [cz-1, p4[1]]]
        rect = plt.Polygon(corners,
                        alpha=0.7,
                        **kwargs)
        ax.add_patch(rect)
        ax.set_xlabel("z (beam) [cm]")

# get the tpc boundaries from the geometry part in a h5 file 
def extract_tpc_bounds(file_obj):
    mod_bounds = np.array(file_obj['geometry_info'].attrs['module_RO_bounds']) 
    max_drift = file_obj['geometry_info'].attrs['max_drift_distance']
    tpc_bounds = []
    for mod in mod_bounds:
        x_min, y_min, z_min = mod[0]
        x_max, y_max, z_max = mod[1]
        #Two TPC boundaries per module:
        tpc_bounds.append(((x_max - max_drift, y_min, z_min), (x_max, y_max, z_max)))
        tpc_bounds.append(((x_min, y_min, z_min), (x_min + max_drift, y_max, z_max)))
    return np.array(tpc_bounds)

def get_tpc_from_xz(x_val, z_val, tpc_bounds):
    for tpc, bounds in enumerate(tpc_bounds):
        lower, upper = np.array(bounds[0]), np.array(bounds[1])
        x0, y0, z0 = lower
        x1, y1, z1 = upper
        if x0 <= x_val <= x1 and z0 <= z_val <=z1:
            return tpc
    return -1  # not found



###--- Functions to draw waveforms only ---
def plotByTPCswvfm(fig, axes, wvfm_data, tpc, event, hits_in_event, sum_hits):
    # Create 2-column, 4-row grid
    fig.suptitle(f"Waveforms per TPC, Event {event}")

    
    for j in range(16):
        ax = axes[j]
        wvfm= wvfm_data[event][tpc][j]

        ax.plot(wvfm, label=f'{j}', alpha=0.7)
        ax.grid(True, alpha=0.2)
        
    # hits in the event
    for h in hits_in_event:
        tpc = int(sum_hits['tpc'][h]) # tpc
        det = int(sum_hits['det'][h]) # det

        sample_idx = sum_hits['sample_idx'][h]
        wvfm= wvfm_data[event][tpc][det]
        axes[det].plot(sample_idx, wvfm[sample_idx], 'rx', label=f'Hit {h}, {det}')
    
    
def plotByTPC(axes, wvfm_data, event, hits_in_event, tpc_hits, tpc_list=[0,1,2,3,4,5,6,7]):
    
    # fig.suptitle(f"Waveforms per TPC, Event {event}")

    for i, tpc in enumerate(tpc_list):
        ax = axes[i]
        wvfm_acl = wvfm_data[event][tpc][0]
        wvfm_lcm = wvfm_data[event][tpc][1]

        ax.plot(wvfm_acl, color='green', label='ACL', alpha=0.7)
        ax.plot(wvfm_lcm, color='blue', label='LCM', alpha=0.7)

        ax.set_title(f"TPC {tpc}")
        ax.grid(True, alpha=0.2)

    for h in hits_in_event:
        tpc = int(tpc_hits['tpc'][h]) # tpc
        ttype = int(tpc_hits['trap_type'][h]) # det

        sample_idx = tpc_hits['sample_idx'][h]
        wvfm= wvfm_data[event][tpc][ttype]
        axes[tpc].plot(sample_idx, wvfm[sample_idx], 'rx', label=f'Hit {h}, {tpc}')

    # Set x-label only on the bottom row
    for ax in axes[-2:]:
        ax.set_xlabel("Sample")


def plot_wvfm(sum_hits, wvfm_data, fprompt_data, event, hit, tpc, det):
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(5, 3))
    sample_idx = sum_hits['sample_idx'][hit]
    # wvfm_acl = wvfm_data[event][tpc][0]
    # wvfm_lcm = wvfm_data[event][tpc][1]
    wvfm= wvfm_data[event][tpc][det]

    t0_bin = sample_idx-5
    prompt_bins = int(np.ceil(200 / 16)) 
    total_bins = int(np.ceil(3200 / 16)) 
    x_short = np.arange(t0_bin, t0_bin+prompt_bins, 1)
    x_long = np.arange(t0_bin, t0_bin+total_bins, 1)

    axs.plot(wvfm, color='green', label=f'{det}', alpha=0.7)
    
    axs.axvline(x=150, ymin=0, color='black', linestyle="--", alpha=0.8, label='Trigger')

    axs.plot(sample_idx, wvfm[sample_idx], 'rx', label='Hit')
    axs.fill_between(x_short, wvfm[t0_bin:t0_bin+prompt_bins], color='red', alpha=0.5)
    axs.fill_between(x_long, wvfm[t0_bin:t0_bin+total_bins],  color='red', alpha=0.2)

    axs.set_title(f'TPC {tpc}', size=9, y=1.) 
    fprompt = (fprompt_data[hit])
    plt.subplots_adjust(hspace=0.1)
    fig.suptitle(f'Event {event}, fprompt {fprompt:.5f}', y = 0.95)
    fig.supxlabel('Ticks',y=0.0)
    fig.supylabel('PE/Tick',x=0.05)
    axs.legend(loc="upper left")
    axs.grid(True)
    plt.show()