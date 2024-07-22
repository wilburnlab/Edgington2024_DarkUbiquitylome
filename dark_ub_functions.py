#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
import numpy as np
import itertools
from scipy.interpolate import interp1d
import re


# In[2]:


masses = { 'proton' : 1.00727646688, 'hydrogen' : 1.007825035, 'carbon' : 12.000000,
           'nitrogen' : 14.003074, 'oxygen' : 15.99491463, 'phosphorus' : 30.973762,
           'sulfur' : 31.9720707, 
           
           'A' : 71.037113805,  'C' : 103.009184505, 'D' : 115.026943065, 'E' : 129.042593135,
           'F' : 147.068413945, 'G' : 57.021463735,  'H' : 137.058911875, 'I' : 113.084064015,
           'K' : 128.094963050, 'L' : 113.084064015, 'M' : 131.040484645, 'N' : 114.042927470,
           'P' : 97.052763875,  'Q' : 128.058577540, 'R' : 156.101111050, 'S' : 87.032028435,
           'T' : 101.047678505, 'V' : 99.068413945,  'W' : 186.079312980, 'Y' : 163.063328575, }


# Modifications
masses['H2O'] = masses['hydrogen']*2 + masses['oxygen']
masses['NH3'] = masses['nitrogen'] + masses['hydrogen']*3
masses['Ox'] = masses['oxygen']
masses['Cam'] = masses['hydrogen']*3 + masses['carbon']*2 + masses['nitrogen'] + masses['oxygen']
masses['Phospho'] = masses['hydrogen'] + masses['oxygen']*3 + masses['phosphorus']
masses['Ac'] = masses['hydrogen']*2 + masses['carbon']*2 + masses['oxygen']
masses['Me'] = masses['carbon'] + masses['hydrogen']*2
masses['Succ'] = masses['carbon']*4 + masses['oxygen']*3 + masses['hydrogen']*5
masses['Ub'] = masses['G']*2
masses['TMT0'] = 224.152478
masses['TMT10'] = 229.162932


# In[3]:


amino_acids = [
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
]


# In[4]:


mzs_to_drop = [
    round(masses[AA] + masses['proton'], 3)
    for AA in amino_acids
] + [
    round(masses[AA] + masses['proton'] + masses['H2O'], 3)
    for AA in amino_acids
]

mzs_to_drop_114 = [
    round(masses[AA] + masses['proton'] + masses['Ub'], 3)
    for AA in amino_acids
] + [
    round(masses[AA] + masses['proton'] + masses['H2O'], 3)
    for AA in amino_acids
]


# In[5]:


def keep_specific_mod(sequence, mod_to_keep):
    pattern = rf'\[\+(?!{re.escape(mod_to_keep)}).+?\]'
    cleaned_sequence = re.sub(pattern, '', sequence)
    return cleaned_sequence


# In[6]:


def annotate_ions_for_plotting(sequence, MODSEQ, charge):
    
    '''
    This function generates fragment ion m/z values for a given peptide sequence and annotates whether each fragment ion contains a specified modification.
    '''
    modseq = keep_specific_mod(MODSEQ, '114.042927')
    
    
    mod_index = modseq.find('[+114.042927]')
    mod_index_reverse = len(sequence) - mod_index

    charge = min(charge, 3)  # max fragment ion charge is 3
    
    mono_mass_list = [masses[aa] for aa in sequence]
    cumsum_list = np.cumsum(mono_mass_list)
    cumsum_list_reverse = np.cumsum(mono_mass_list[::-1])
    
    data_list = []
    
    # generate b ions mzs
    for i in range(0,charge):
        for length, val in enumerate(cumsum_list[:-1], start = 1):
            if length < mod_index:
                ion_mz = (val + masses['proton']*(i+1)/(i+1))
                data = {'ion_type' : 'b'+str(length),
                        'Mod': 0, # 0 indicates false -> mod is not on fragment ion
                        'ion_mz' : ion_mz,
                        'ion_charge' : (i+1), 'mass_shift' : 0}
            if length >= mod_index:
                ion_mz = (val + masses['proton']*(i+1)/(i+1))
                data = {'ion_type' : 'b'+str(length),
                        'Mod': 1, # 1 indicates true -> mod is on fragment ion
                        'ion_mz' : ion_mz,
                        'ion_charge' : (i+1), 'mass_shift' : 0}
            data_list.append(data)


    # generate y ions mzs
    for i in range(0,charge):
        for length, val in enumerate(cumsum_list_reverse[:-1], start = 1):
            
            if length <= mod_index_reverse:
                ion_mz = (val + masses['oxygen'] + masses['proton']*(i+3))/(i+1)
                data = {'ion_type' : 'y'+str(length),
                    'Mod': 0,
                    'ion_mz' : ion_mz,
                    'ion_charge' : (i+1), 'mass_shift' : 0}
            if length > mod_index_reverse:
                ion_mz = (val + masses['oxygen'] + masses['proton']*(i+3))/(i+1)
                data = {'ion_type' : 'y'+str(length),
                    'Mod': 1,
                    'ion_mz' : ion_mz,
                    'ion_charge' : (i+1),
                    'mass_shift' : 0}
            data_list.append(data)
            
            
            
    df_result = pd.DataFrame(data_list)

    return (df_result)


# In[7]:


def return_mzs_to_shift(sequence, modseq, charge, prosit_mz, prosit_int):
    '''
    This function identifies specific m/z values from Prosit predictions that need to be mass-shifted based on the presence of a modification in a peptide sequence.
    '''
    
    mod_index = modseq.find('[+114.042927]')
    mod_index_reverse = len(sequence) - mod_index

    charge = min(charge, 3)  # Set max fragment ion charge to precursor charge or 3 if precursor charge > 3
    
    mono_mass_list = [masses[aa] for aa in sequence]
    cumsum_list = np.cumsum(mono_mass_list)
    cumsum_list_reverse = np.cumsum(mono_mass_list[::-1])
    
    ion_list = []

    # generate b ion mzs for unmodified fragment ions that retain the modification
    for i in range(0,charge):
        for length, val in enumerate(cumsum_list[:-1], start = 1):
            if length >= mod_index:
                ion_mz = (val + masses['proton']*(i+1))/(i+1)
                ion_list.append((ion_mz, i+1))



    # generate y ion mzs for unmodified fragment ions that retain the modification
    for i in range(0,charge):
        for length, val in enumerate(cumsum_list_reverse[:-1], start = 1):
            if length > mod_index_reverse:
                ion_mz = (val + masses['oxygen'] + masses['proton']*(i+3))/(i+1)
                ion_list.append((ion_mz, i+1))

    rounded_mz_to_mz = {round(mz, 2): mz for mz in prosit_mz}

    mzs_to_shift = {rounded_mz_to_mz[round(ion_mz[0], 2)]: ion_mz[1] for ion_mz in ion_list if round(ion_mz[0], 2) in rounded_mz_to_mz}
    MZS = list(mzs_to_shift.keys())

    mz_to_int = dict(zip(prosit_mz, prosit_int))
    
    frag_prosit_mz = [mz for mz in mz_to_int if mz in MZS]
    frag_prosit_int = [mz_to_int[mz] for mz in frag_prosit_mz]

    # return mzs_to_shift ( a dict of mzs that need to be shifted : frag ion charge), prosit mzs to shift, prosit intensities of those mzs
    return mzs_to_shift, np.array(frag_prosit_mz), np.array(frag_prosit_int)


# In[8]:


def return_modified_frags(sequence, modseq, charge, diagnostic = False):
    '''
    This function returns a dictionary of fragment ions that contain a KGG modification.
    The dictionary is in the form {unmodified_mz: {mod_mz, frag_ion, ion_charge}}.
    '''
    
    mod_index = modseq.find('[+114.042927]')
    mod_index_reverse = len(sequence) - mod_index

    charge = min(charge, 3)
    
    mono_mass_list = [masses[aa] for aa in sequence]
    cumsum_list = np.cumsum(mono_mass_list)
    cumsum_list_reverse = np.cumsum(mono_mass_list[::-1])
        
    mass_shift_dict = {}
    
    if not diagnostic:
        for i in range(charge):
            for length, val in enumerate(cumsum_list[:-1], start=1):
                if length >= mod_index:
                    ion_mz_nomod = (val + masses['proton'] * (i + 1)) / (i + 1)
                    ion_mz = ion_mz_nomod + 2 * masses['G'] / (i + 1)
                    mass_shift_dict[round(ion_mz_nomod, 2)] = {
                        'mod_ion_mz': ion_mz,
                        'ion_type': f'b{length}',
                        'ion_charge': i + 1
                    }

        for i in range(charge):
            for length, val in enumerate(cumsum_list_reverse[:-1], start=1):
                if length > mod_index_reverse:
                    ion_mz_nomod = (val + masses['oxygen'] + masses['proton'] * (i + 3)) / (i + 1)
                    ion_mz = ion_mz_nomod + 2 * masses['G'] / (i + 1)
                    mass_shift_dict[round(ion_mz_nomod, 2)] = {
                        'mod_ion_mz': ion_mz,
                        'ion_type': f'y{length}',
                        'ion_charge': i + 1
                    }
                    
    if diagnostic:
        for i in range(charge):
            for length, val in enumerate(cumsum_list[:-1], start=1):
                if length >= mod_index:
                    ion_mz_nomod = (val + masses['proton'] * (i + 1)) / (i + 1)
                    ion_mz_intact = ion_mz_nomod + 2 * masses['G'] / (i + 1)
                    ion_mz_18 = ion_mz_nomod + (2 * masses['G'] - masses['H2O']) / (i + 1)
                    ion_mz_57 = ion_mz_nomod + masses['G'] / (i + 1)
                    ion_mz_114 = ion_mz_nomod
                    mass_shift_dict[round(ion_mz_nomod, 2)] = {
                        'mod_ion_mz': [ion_mz_intact, ion_mz_18, ion_mz_57, ion_mz_114],
                        'ion_type': f'b{length}',
                        'ion_charge': i + 1
                    }

        for i in range(charge):
            for length, val in enumerate(cumsum_list_reverse[:-1], start=1):
                if length > mod_index_reverse:
                    ion_mz_nomod = (val + masses['oxygen'] + masses['proton'] * (i + 3)) / (i + 1)
                    ion_mz_intact = ion_mz_nomod + 2 * masses['G'] / (i + 1)
                    ion_mz_18 = ion_mz_nomod + (2 * masses['G'] - masses['H2O']) / (i + 1)
                    ion_mz_57 = ion_mz_nomod +  masses['G'] / (i + 1)
                    ion_mz_114 = ion_mz_nomod
                    mass_shift_dict[round(ion_mz_nomod, 2)] = {
                        'mod_ion_mz': [ion_mz_intact, ion_mz_18, ion_mz_57, ion_mz_114],
                        'ion_type': f'y{length}',
                        'ion_charge': i + 1
                    }           

    return mass_shift_dict


# In[9]:


def cosine_similarity(exp_mz, exp_int, prosit_MZ, prosit_INT, modseq, softmax=True, dropb1y1=True, summed_error=False):
    
    epsilon = 1e-7

    # remove b1 and y1 ions
    if dropb1y1:
        prosit_mz_rounded = np.round(prosit_MZ, 3)
        mod_index = modseq.find('[+114.042927]')  # for unmodified peptides, modseq = seq and mod_index = -1
        indices_to_remove = []
        mz_to_rounded = dict(zip(prosit_MZ, prosit_mz_rounded))

        # if mod is on the first AA, use (b1 + 114) mz
        if mod_index == 1:
            mzs_to_check = mzs_to_drop_114
        else:
            mzs_to_check = mzs_to_drop

        for index, mz in np.ndenumerate(prosit_MZ):
            if mz_to_rounded[mz] in mzs_to_check:
                indices_to_remove.append(index)

        prosit_MZ = np.delete(prosit_MZ, indices_to_remove)
        prosit_INT = np.delete(prosit_INT, indices_to_remove)

    # score spectra    
    # reshape arrays
    prosit_mz_reshape = prosit_MZ.reshape(-1, 1)  # (n,1)
    exp_mz_reshape = exp_mz.reshape(1, -1)  # (1,m)
    int_reshape = exp_int.reshape(1, -1)  # (1,n)

    # find experimental mzs that correspond to prosit predicted mzs
    z = np.abs(prosit_mz_reshape - exp_mz_reshape)  # (n,m)
    min_values = np.min(z, axis=1, keepdims=True)  # (n,1)
    result_array = (z == min_values) & (z <= 20e-6 * prosit_mz_reshape)

    # map experimental mzs to experimental intensity
    exp_int_2 = np.multiply(int_reshape, result_array)  # (n,m) array with only int values that correspond to mapped mzs, filled with zeros elsewhere
    important_exp_int = np.amax(exp_int_2, axis=1)  # (n,1) get rid of zeros!

    # return only important mzs for plotting purposes
    exp_mz_2 = np.multiply(exp_mz_reshape, result_array)  # (n,m) array with only int values that correspond to mapped mzs, filled with zeros elsewhere
    important_exp_mz = np.amax(exp_mz_2, axis=1)  # (n,1) get rid of zeros!

    if softmax:
        # normalize experimental intensities - normalize via softmax
        final_int = important_exp_int / (np.sum(important_exp_int) + epsilon)
        # normalize prosit intensities
        prosit_INT = prosit_INT / (np.sum(prosit_INT) + epsilon)
    else:
        # normalize experimental intensities
        final_int = important_exp_int / (np.max(important_exp_int) + epsilon)
        # normalize prosit intensities
        prosit_INT = prosit_INT / (np.max(prosit_INT) + epsilon)

    # compute cosine similarity
    dot_product = np.dot(final_int, prosit_INT)
    exp_mag = np.linalg.norm(final_int)
    prosit_mag = np.linalg.norm(prosit_INT)
    norm_dot_product = dot_product / ((exp_mag * prosit_mag) + epsilon)

    arccos = np.arccos(norm_dot_product)  # compute angle between two vectors
    score = 1 - (2 * arccos / np.pi)

    if not summed_error:
        return score, important_exp_mz, final_int, prosit_MZ, prosit_INT

    # compute average mass error for scored frags 
    zero_mask = important_exp_mz != 0  # only compute matched ion ppm values
    diff = np.abs(prosit_MZ[zero_mask] - important_exp_mz[zero_mask])  # calculate mass error
    if diff.size > 0:
        ppm_error = (diff / prosit_MZ[zero_mask]) * 1e6  # calculate ppm error
        avg_ppm_error = np.sum(ppm_error) / len(prosit_MZ[zero_mask])  # return average mass error for all frags
    else:  # if no ions match, error = 'nan'
        avg_ppm_error = 'nan'

    return score, important_exp_mz, final_int, prosit_MZ, prosit_INT, avg_ppm_error


# In[10]:


class cosine_gaussian:
    def __init__(self, mean=0.0, sd=1.0, prior=1.0):
        self.mean = mean
        self.sd = sd
        self.prior = prior
        self.A = np.pi / (8.0*sd)
        self.M = np.pi / (4.0*sd)
        self.min = mean - 2.0*sd
        self.max = mean + 2.0*sd
    def compute_pdf( self, x ):
        if x < self.min or x > self.max:
            return 0.0
        else:
            return self.A * np.cos( (self.mean - x) * self.M )
    def compute_cdf( self, x ):
        if x < self.min or x > self.max:
            return 0.0
        else:
            d = (x-self.mean) / self.sd
            return 1.0 / (1.0+np.exp(-0.07056*np.power(d,3)-1.5976*d))
    def return_sd( self ):
        return self.sd


# In[11]:


def define_stamp( dist ):
    sd = dist.return_sd()
    stamp_radius = int( np.round( 2.0 * sd ) )
    stamp_width = 2*stamp_radius + 1
    stamp = np.zeros( (stamp_width, stamp_width) )
    for i,j in itertools.product( range(stamp_width), range(stamp_width) ):
        distance = np.sqrt( (i-stamp_radius)**2 + (j-stamp_radius)**2 )
        stamp[i,j] = dist.compute_pdf( distance )
    return stamp


# In[12]:


def KDE_align( x_vals, y_vals, n=3000, min_x=None, max_x=None, min_y=None, max_y=None ):
    # Transform input values to 0-1 scale
    if min_x == None: min_x = np.min( x_vals )
    if max_x == None: max_x = np.max( x_vals )
    range_x = max_x - min_x
    if min_y == None: min_y = np.min( y_vals )
    if max_y == None: max_y = np.max( y_vals )
    range_y = max_y - min_y

    # Discretize the normalized values to the number of array points
    xs = np.array( [ int( np.round( (n-1) * (x-min_x)/range_x ) )
                     for x in np.array(x_vals) ] )
    ys = np.array( [ int( np.round( (n-1) * (y-min_y)/range_y ) )
                     for y in np.array(y_vals) ] )

    # Compute Silverman kernel and define the "stamp"
    bandwidth = np.power( float(len(xs)), -1/6 ) * ( np.std(xs) + np.std(ys) ) / 2
    kernel_sd = bandwidth / 2 / np.sqrt( 2 * np.log(2) )
    distribution = cosine_gaussian( sd=kernel_sd )
    stamp = define_stamp( distribution )
    stamp_width = stamp.shape[0]
    stamp_radius = (stamp_width-1)/2

    # Generate the KDE map
    array = np.zeros( (n,n) )
    for i,j in zip( xs, ys ):
        l_bound = int( np.max( [ 0, i-stamp_radius ] ) )
        r_bound = int( np.min( [ n-1, i+stamp_radius ] ) )
        b_bound = int( np.max( [ 0, j-stamp_radius ] ) )
        t_bound = int( np.min( [ n-1, j+stamp_radius ] ) )

        if l_bound == 0:
            l_trim = stamp_width - r_bound - 1
            r_trim = stamp_width
        elif r_bound == n-1:
            l_trim = 0
            r_trim = n - l_bound
        else:
            l_trim = 0
            r_trim = stamp_width
        if b_bound == 0:
            b_trim = stamp_width - t_bound - 1
            t_trim = stamp_width
        elif t_bound == n-1:
            b_trim = 0
            t_trim = n - b_bound
        else:
            b_trim = 0
            t_trim = stamp_width

        local_stamp = stamp[ l_trim:r_trim, b_trim:t_trim ]
        array[ l_bound:r_bound+1, b_bound:t_bound+1 ] += local_stamp

    # Identify the KDE apex and perform forward ridge walk
    peak_i, peak_j = np.unravel_index( array.argmax(), array.shape)
    trace_i = peak_i
    trace_j = peak_j
    steps = [ (0,1), (1,0), (1,1) ]
    fit_points = [ (peak_i, peak_j) ]
    while trace_i < n-1 and trace_j < n-1:
        best_step_index = np.argmax( [ array[ trace_i+step_i, trace_j+step_j ]
                                     for step_i, step_j in steps  ] )
        trace_i += steps[best_step_index][0]
        trace_j += steps[best_step_index][1]
        if best_step_index > 0:
            fit_points.append( [ trace_i, trace_j ] )
    
    # Begin reverse ridge walk
    trace_i = peak_i
    trace_j = peak_j
    while trace_i > 0 and trace_j > 0:
        best_step_index = np.argmax( [ array[ trace_i-step_i, trace_j-step_j ]
                                     for step_i, step_j in steps  ] )
        trace_i -= steps[best_step_index][0]
        trace_j -= steps[best_step_index][1]
        if best_step_index > 0:
            fit_points.append( [ trace_i, trace_j ] )
            
    # Sort ridge walk along x-axis, reshape + rescale values, and compute spline
    fit_points.sort( key=lambda x:x[0] )
    if fit_points[0][0] != 0: fit_points = [ [0, fit_points[0][1]] ] + fit_points
    if fit_points[-1][0] != n-1: fit_points = fit_points + [ [n-1, fit_points[-1][1]] ]
    fit_x, fit_y = np.array(fit_points).T / float(n-1)
    fit_x = fit_x*range_x + min_x
    fit_y = fit_y*range_y + min_y
    spline = interp1d( fit_x, fit_y, kind='slinear' )
    interpolation_range = ( np.min( fit_x ), np.max( fit_x ) )
    return spline, interpolation_range, array, fit_x, fit_y


# In[13]:


def return_ion_types(sequence,charge):
    
    # function that returns dict of fragment ions and ion type

    mono_mass_list = []

    
    if charge > 3:
        charge = 3
        
        
    for AA in sequence:
        mono_mass_list.append(masses[AA])
        
    cumsum_list = np.cumsum(mono_mass_list)
    mono_mass_list_reverse = mono_mass_list[::-1]
    cumsum_list_reverse = np.cumsum(mono_mass_list_reverse)
        
    frag_mzs = {}
    
    # generate b ions
    for i in range(0,charge):
        for length, val in enumerate(cumsum_list[:-1], start = 1):
            ion_mz = (val + masses['proton']*(i+1))/(i+1)
            ion_type = 'b'+str(length)
            ion_charge = i+1
            frag_mzs[ion_mz] = (ion_type, i+1)


    # generate y ions

    for i in range(0,charge):
        for length, val in enumerate(cumsum_list_reverse[:-1], start = 1):
            ion_mz = (val + masses['oxygen'] + masses['proton']*(i+3))/(i+1)
            ion_type = 'y'+str(length)
            ion_charge = i+1
            frag_mzs[ion_mz] = (ion_type, i+1)            
   

    return (frag_mzs)

