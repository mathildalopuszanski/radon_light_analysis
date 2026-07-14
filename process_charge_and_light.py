# This file is used to process the CL matched files and light files and 
# to save the necessary information in one file
# also includes finding BiPo coincidences 
# The combined file makes it easier to later analyse the data to only open one smaller file

import numpy as np
import os
import h5py
from h5flow.data import dereference 
import glob
from BiPoSelection import *


# function that finds the BiPo coincidences and the CL matched events
# it returns a dictionary that can be saved as a file
def process_run(CL_file, sum_light_file, tpc_light_file):
    """
    CL_file: h5 CL matched file
    sum_light_file: npy file with the light data for trap waveforms
    tpc_light_file: npy file with the light data for tpc waveforms
    returns dictionary with the cluster information and light event information
    """
    # light tpc level
    tpc_hit_data = np.load(tpc_light_file, allow_pickle=True).item()
    sum_tpc_hits = tpc_hit_data
    sum_tpc_hits_refs = tpc_hit_data['refs']

    # light sum level
    sum_hit_data = np.load(sum_light_file, allow_pickle=True).item() 

    # finding multiple hits in event
    hit_pairs, events_with_nhits, dt_list = calculate_dt(sum_tpc_hits, sum_tpc_hits_refs, 
                                                         tpc_or_ch_sum='tpc')
    print(f'Found {len(hit_pairs)} light events with multiple hits in one wvfm')

    # CL matching
    matched_CLevents  = get_reconstructed_position(CL_file)
    matched_light = matched_CLevents.keys()
    print(f'Found {len(matched_light)} light events matched to charge cluster')

    # CL matching and BiPo coincidences
    matched_nhit_events = [e for e in events_with_nhits if e in matched_CLevents]
    print(f'Found {len(matched_nhit_events)} events with multiple hits in one wvfm AND matched charge')

    # events are all events in the file, hits is the first hit associated with the event
    events, hits, hits_per_event  = np.unique(sum_hit_data['refs'][:,0], return_index=True,sorted=False, return_counts=True)
    # BiPo mask 
    matched_BiPo_events = set(matched_nhit_events) 

    # preparing lists
    light_ids, tot_PE, is_bipo, first_hits, num_hits  = [], [], [], [], []
    tpc         = []   # tpc of hits
    charge_id, cl_ref = [], []
    cl_x_rec, cl_x_anode, cl_y, cl_z = [], [], [], []
    cl_charge_id, num_cl, first_cl = [], [], []

    # for all CL matched events get the cluster and light info and store in dictionary
    for e in matched_light:
        first_hit = hits[e]
        num_hits_in_event = hits_per_event[e]
        hits_in_event = np.arange(first_hit, first_hit+num_hits_in_event, dtype=np.int64)
        
        # --- light event ---
        light_ids.append(e)
        tot_PE.append(np.sum(sum_hit_data['integral'][hits_in_event]))
        tpc.append(np.unique(sum_hit_data['tpc'][hits_in_event])) # tpcs that see light in that event
        is_bipo.append(e in matched_BiPo_events) # BiPo tag
        first_hits.append(first_hit) # hits in the event
        num_hits.append(num_hits_in_event) 

        # --- clusters matched ---
        cl = matched_CLevents[e] # clusters in that event
        n  = len(cl['x'])
        first_cl.append(len(cl_x_rec))   # current length = start index for this event (length of num light ids)
        num_cl.append(n) # lengths of num light ids

        cl_ref.extend([e] * n) # this has the lengths of the number of CL matched clusters
        
        cl_x_rec.extend(cl['x']) 
        cl_x_anode.extend(cl['x_anode'])
        cl_y.extend(cl['y'])
        cl_z.extend(cl['z'])
        cl_charge_id.extend(cl['chargeID'])


    return dict(light_id  = np.array(light_ids, dtype=np.int64),
                tot_PE    = np.array(tot_PE),
                dt = np.array(dt_list),
                tpc       = np.array(tpc, dtype=object), # object to have different length possible 
                is_bipo   = np.array(is_bipo, dtype=bool), # mask that can be used to find the BiPos
                first_hits = np.array(first_hits, dtype=np.int64), # hits in the event
                num_hits = np.array(num_hits, dtype=np.int64),
                first_cl = np.array(first_cl, dtype=np.int64),
                num_cl = np.array(num_cl, dtype=np.int64),
                cluster_light_ref = np.array(cl_ref, dtype=np.int64),
                cluster_x_rec     = np.array(cl_x_rec),
                cluster_x_anode   = np.array(cl_x_anode),
                cluster_y         = np.array(cl_y),
                cluster_z         = np.array(cl_z),
                cluster_charge_id = np.array(cl_charge_id)
                )


###--- FILES ---
# provide your file paths here 

# CL matched files (h5 files)
CL_path = '' # '/global/cfs/cdirs/dune/www/data/2x2/nearline_run2/flowed_CLmatching_low_energy/flowed_light_low_energy_v3/source_rn_bin1/injection/'
# light files (h5 files)
# this is were I stored some of the files I flowed myself
# '/global/cfs/cdirs/dune/users/mlopuszanski/Rn_injection/results/'
light_path = ''
# paths where to store the processed files
results_path = ''

# build all the file paths
run_names = [f'rctl_775_p{i}' for i in range(1, 5)]
runs = build_runs(run_names, CL_path, light_path)
print(f'Found {len(runs)} complete runs')
# check the file names
for r in runs:
    print(f'  {r["name"]}')
    print(f'  {r["CL"]}')

###--- Process the files ---
for run in runs:
    print(f'Processing {run["name"]}...')
    result = process_run(run['CL'], run['sum_light'], run['tpc_light'])
    np.savez(f'{results_path}CL_{run["name"]}.npz', **result)
    print(f'  -> saved CL_{run["name"]}.npz  '
          f'({len(result["light_id"])} light events, '
          f'{result["is_bipo"].sum()} BiPo candidates)')


