import warnings
warnings.filterwarnings('ignore')
import ase
from ase import Atoms, build
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import sys
import pickle
import scipy as sp
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler

from mendeleev import element
from mendeleev.fetch import fetch_table, fetch_ionization_energies

df_elements = fetch_table('elements')
df_ionenergy  = fetch_ionization_energies()


def get_scaled_features(X):

    #MinMax Normalization
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(X)
    mm_X = mm_scaler.transform(X)

    scaler = mm_scaler

    return mm_X, scaler


def get_conventional_cell(structure):
    
    """
       Function that returns the conventional cell given a primitive structure as obtained from Materials Project
    """

    sga = SpacegroupAnalyzer(structure)
    conventional_structure = sga.get_conventional_standard_structure()

    return conventional_structure

def _large_dist(slab, dis_thres, repeat = False):

    minus_dict = { '4' : 0,
                   '6' : 2,
                   '8' : 6,
                   '10' : 12,
                   '12' : 20,
                   '14' : 30,
                   '16' : 42,
                  }
  
    atom_positions = np.array(slab.get_positions())
    n_atoms = atom_positions.shape[0]
    max_large = ((n_atoms - 1) * (n_atoms - 2)) / 2
    if repeat:
        max_large -= minus_dict[str(n_atoms)]
    dist_arr = pdist(atom_positions, metric='euclidean')
    n_large = dist_arr[dist_arr > dis_thres].shape[0]

    if n_large > max_large:
        return True
    else:
        return False

def check_slab_validity(slab, miller, planes_frac):

    dis_thres = 4
    n_atoms = np.array(slab.get_positions()).shape[0]
    if n_atoms > 8:
        print(miller, ': There are '+str(n_atoms)+' atoms in the unit cell')
        return False
    if _large_dist(slab, dis_thres):
        print(miller,': Atoms are too far away from each other')
        return False
    else:
        #Repeat on both axis
        slab_rep = slab.repeat((1,2,1))
        if _large_dist(slab_rep, dis_thres, repeat =  True):
            print(miller,': Atoms are too far away from each other')
            return False
        else:
            slab_rep = slab.repeat((2,1,1))
            if _large_dist(slab_rep, dis_thres, repeat =  True):
                print(miller,': Atoms are too far away from each other')
                return False
            else:
                scaled_pos = np.array(slab.get_scaled_positions())
                unique = True
                for plane in planes_frac.keys():
                    diff = scaled_pos - planes_frac[plane]
                    if np.all(diff < 0.01):
                        print(miller,': It is equal to', plane)
                        unique = False
                        break

                if unique:
                    return True
                else:
                    return False

def get_slab(ID, crystal, API_key):

    with MPRester(str(API_key)) as mpr:
        structure = mpr.get_structure_by_material_id(ID)
        cell = get_conventional_cell(structure)
        frac_coords  = structure.frac_coords
        chem_symb = structure.species

        a,b,c = cell._lattice.abc
        alpha, beta, gamma = cell._lattice.angles

    lattice_param = [a,b,c, alpha, beta, gamma]

    #Constructing the slab
    formula = ''
    for symb in chem_symb:
        formula += str(symb)

    syst = Atoms(formula,
            scaled_positions = frac_coords,
            cell = lattice_param,
            pbc = False)

    init_idx = 0
    n_range = 5 if crystal == 'Hexagonal' else 4
    final_idx = 2 if crystal == 'Hexagonal' else 3

    miller_dict = {}
    for h in range(init_idx, final_idx+1):
        for k in range(init_idx, final_idx+1):
            for l in range(init_idx, final_idx+1):
                if h == 0 and k == 0 and l == 0:
                    continue
                else:
                    miller_array = np.array([h,k,l]) if crystal != 'Hexagonal' else np.array([h,k, -1*(h+k), l])
                 
                    miller_string = ''
                    for index in miller_array:
                        miller_string += str(index)
                        
                    if miller_string not in miller_dict.keys():
                        miller_dict[miller_string] = tuple(miller_array) if crystal != 'Hexagonal' else tuple([miller_array[0], miller_array[1], miller_array[-1]])

    valid_slabs = {}
    notvalid_slabs = {}
    layers = 4 if crystal == 'Cubic' else 2
    planes_frac = {}
    for plane in miller_dict.keys():
        slab_tmp = build.surface(syst, miller_dict[plane], 1, vacuum = 10.0)

        is_valid = check_slab_validity(slab_tmp, plane, planes_frac)
        
        slab4GCN = build.surface(syst, miller_dict[plane], 5, vacuum= 10.0)
        slab = build.surface(syst, miller_dict[plane], 4, vacuum= 10.0)

        slab4GCN = slab4GCN.repeat((1,2,1))
        slab = slab.repeat((1,2,1))

        if is_valid:
            planes_frac[plane] = np.array(slab_tmp.get_scaled_positions())
            valid_slabs[plane] = (slab, slab4GCN)
        else:
            notvalid_slabs[plane] = slab

    return valid_slabs, notvalid_slabs

def get_circumcenter(A, B, C):

    D = 2 * ((A[0] * (B[1] - C[1])) + (B[0] * (C[1] - A[1])) + (C[0] * (A[1] - B[1])))

    Ux = (1. / D) * ((((A[0] ** 2) + (A[1] ** 2)) * (B[1] - C[1])) + (((B[0] ** 2) + (B[1] ** 2)) * (C[1] - A[1])) + (((C[0] ** 2) + (C[1] ** 2)) * (A[1] - B[1])))

    Uy = (1. / D) * ((((A[0] ** 2) + (A[1] ** 2)) * (C[0] - B[0])) + (((B[0] ** 2) + (B[1] ** 2)) * (A[0] - C[0])) + (((C[0] ** 2) + (C[1] ** 2)) * (B[0] - A[0])))


    return np.array([Ux, Uy, A[2]])


def get_sites(slab):

    unit_cell = np.copy(np.array(slab.get_cell()))
    x_length = np.linalg.norm(unit_cell[0,:])
    y_length = np.linalg.norm(unit_cell[1,:])
    

    large_slab = slab.repeat((5,5,1))
    atom_positions = np.array(large_slab.get_positions())
    z_sort = np.argsort(atom_positions[:,-1])[::-1]
    atom_positions = atom_positions[z_sort,:]
    
    atom_symbols = np.array(large_slab.get_chemical_symbols())
    atom_symbols = atom_symbols[z_sort]
    
    n_atoms = atom_positions.shape[0]

    atomsperlayer = int(n_atoms / 4)

    
    top_atoms = atom_positions[:atomsperlayer, :]
    top_at_symb = atom_symbols[:atomsperlayer]

    if unit_cell[1,0] < 0:
        lims4sites = [((x_length * 2) + (2 * unit_cell[1,0]), (x_length * 3) + (2 * unit_cell[1,0])), (unit_cell[1,1] * 2, unit_cell[1,1] * 3)]
    else:
        lims4sites = [((x_length * 2.) + (2 * unit_cell[1,0]) , (x_length * 3.) + (3 * unit_cell[1,0])), (unit_cell[1,1] * 2., unit_cell[1,1] * 3.)]

    atom_idxs = np.where((top_atoms[:,0] >= lims4sites[0][0]) & (top_atoms[:,0] <= lims4sites[0][1]) & (top_atoms[:,1] >= lims4sites[1][0]) & (top_atoms[:,1] <= lims4sites[1][1]))[0]
    atom_idxs = np.sort(atom_idxs)[::-1]
    top_atoms = top_atoms[atom_idxs, :]
    top_at_symb = top_at_symb[atom_idxs]

    sites = {'ontop' : [],
             'bridge': [],
             'threefold' : [],
             'hollow': [],
            }

    body_thres = 4

    #1-body sites
    one_body_done = {}
    two_body_done = {}
    three_body_done = {}
    for i, atom_i in enumerate(top_at_symb):
        if atom_i in one_body_done.keys() and np.round(top_atoms[i, -1] * 10.) / 10. in one_body_done[atom_i]:
            pass
        else:
            if np.round(top_atoms[i, -1] * 10.) / 10. < np.amax(np.round(atom_positions[:,-1] * 10.) / 10.):
                sites['hollow'].append((top_atoms[i,0], top_atoms[i,1],top_atoms[i,2]))
            else:
                sites['ontop'].append((top_atoms[i,0], top_atoms[i,1], top_atoms[i,2]))
            if atom_i not in one_body_done.keys():
                one_body_done[atom_i] = [np.round(top_atoms[i,-1] * 10.) / 10.]
            else:
                one_body_done[atom_i].append(np.round(top_atoms[i,-1] * 10.) / 10.)
        #2-body sites
        for j, atom_j in enumerate(top_at_symb):
            if j > i:
                height_i, height_j = np.round(top_atoms[i,-1] * 10.) / 10., np.round(top_atoms[j,-1] * 10.) / 10.
                if height_i != height_j:
                    pass
                else:
                    sorted_symb = np.sort([atom_i, atom_j])
                    site_label = sorted_symb[0]+'-'+sorted_symb[1]
                    ij_dist = np.linalg.norm(top_atoms[i,:] - top_atoms[j,:])
                    if ij_dist > body_thres:
                        pass
                    else:
                        site_coords = (top_atoms[i, :] + top_atoms[j, :]) / 2.
                        if site_label in two_body_done.keys() and np.round(site_coords[-1] * 10.) / 10. in two_body_done[site_label]:
                            pass
                        else:
                            sites['bridge'].append((site_coords[0], site_coords[1], site_coords[2]))
                            if site_label not in two_body_done.keys():
                                two_body_done[site_label] = [np.round(site_coords[-1] * 10.) / 10.]
                            else:
                                two_body_done[site_label].append(np.round(site_coords[-1] * 10.) / 10.)
                    for k, atom_k in enumerate(top_at_symb):
                        if k > j:
                            height_k = np.round(top_atoms[k,-1] * 10.) / 10.
                            if height_k != height_i or height_k != height_j:
                                pass
                            else:
                                sorted_symb = np.sort([atom_i, atom_j, atom_k])
                                site_label = sorted_symb[0]+'-'+sorted_symb[1]+'-'+sorted_symb[2]
                                ik_dist = np.linalg.norm(top_atoms[i,:] - top_atoms[k,:])
                                jk_dist = np.linalg.norm(top_atoms[j,:] - top_atoms[k,:])
                                if np.any(np.array([ij_dist, ik_dist, jk_dist]) > body_thres):
                                    pass
                                else:
                                    #Site coords are those of the circumcenter of the triangle defined by the centers of the three atoms involved
                                    site_coords = get_circumcenter(top_atoms[i,:], top_atoms[j,:], top_atoms[k,:])
                                    if site_label in three_body_done.keys() and  np.round(site_coords[-1] * 10.) / 10. in three_body_done[site_label]:
                                        pass
                                    else:
                                        sites['threefold'].append((site_coords[0], site_coords[1], site_coords[2]))
                                        if site_label not in three_body_done.keys():
                                            three_body_done[site_label] = [np.round(site_coords[-1] * 10.) / 10.]
                                        else:
                                            three_body_done[site_label].append(np.round(site_coords[-1] * 10.) / 10.)

    return sites

def get_outerE(symb):

    RE_dict = {'La' : 3.,
               'Ce' : 4.,
               'Pr' : 5.,
               'Nd' : 6.,
               'Pm' : 7.,
               'Sm' : 8.,
               'Eu' : 9.,
               'Gd' : 10.,
               'Tb' : 11.,
               'Dy' : 12.,
               'Ho' : 13.,
               'Er' : 14.,
               'Tm' : 15.,
               'Yb' : 16.,
               'Lu' : 17.,
               'Pa' : 5.,
               'Np' : 7.,
               'Th' : 4.,
               'U' : 6.,
              }

    group = df_elements['group_id'][df_elements['symbol'] == symb].values[0]

    if symb in RE_dict.keys():
        return RE_dict[symb]
    elif group < 13:
        return group
    else:
        return group - 10

def get_CNmax(slab, crystal):

    large_slab = slab.repeat((5,5,1))

    atom_positions = np.array(large_slab.get_positions())
    z_sort = np.argsort(atom_positions[:,-1])[::-1]
    atom_positions = atom_positions[z_sort,:]

    large_cell = np.copy(np.array(large_slab.get_cell()))
    x_length = np.linalg.norm(large_cell[0,:])
    y_length = np.linalg.norm(large_cell[1,:])

    n_atoms = atom_positions.shape[0]

    atomsperlayer = int(n_atoms / 5)
    layers_pos = [atom_positions[:atomsperlayer,:], atom_positions[atomsperlayer:atomsperlayer * 2, :], atom_positions[atomsperlayer * 2 : atomsperlayer * 3, :]]

    #Calculate GCN
    ####Get CNmax
    if crystal != 'Hexagonal':
       if large_cell[1,0] < 0:
           cell_pseudocenter = (np.array([large_cell[0,0] + np.abs(large_cell[1,0]), large_cell[1,1] + np.abs(large_cell[0,1])]) / 2.) + np.array([large_cell[1,0], 0])
       else:
           cell_pseudocenter = np.array([large_cell[0,0] + np.abs(large_cell[1,0]), large_cell[1,1] + np.abs(large_cell[0,1])]) / 2.
    
       atom_ref2ndlayer = np.argmin(np.linalg.norm(layers_pos[2][:,:2] - cell_pseudocenter, axis=1))
       atom_ref = layers_pos[2][atom_ref2ndlayer,:]
       CNmax = 0
       all_dists = np.round(np.linalg.norm(atom_positions - atom_ref, axis = 1) * 10.) / 10.
       min_dist = np.amin(all_dists[all_dists != 0])
       CNmax = np.where(all_dists == min_dist)[0].shape[0]
    else:
       CNmax = 12

    return CNmax

def get_features(slab, site, crystal, site_type, plane):

    unit_cell = np.copy(np.array(slab[0].get_cell()))
    large_slab = slab[0].repeat((5,5,1))

    atom_positions = np.array(large_slab.get_positions())
    z_sort = np.argsort(atom_positions[:,-1])[::-1]
    atom_positions = atom_positions[z_sort,:]
    
    atom_symbols = np.array(large_slab.get_chemical_symbols())
    atom_symbols = atom_symbols[z_sort]
    
    n_atoms = atom_positions.shape[0]

    atomsperlayer = int(n_atoms / 4)
    layers_pos = [atom_positions[:atomsperlayer,:], atom_positions[atomsperlayer:atomsperlayer * 2, :], atom_positions[atomsperlayer * 2 : atomsperlayer * 3, :]]

    #Checking validity of adsorption site

    hlargerthanz = np.round(layers_pos[0][:,-1] * 10.) / 10. > np.round(site[-1] * 10.) / 10.
    if np.any(hlargerthanz):
        if np.any((np.round(layers_pos[0][hlargerthanz,0] * 10.) / 10. == np.round(site[0] * 10.) / 10.) & (np.round(layers_pos[0][hlargerthanz,1] * 10.) / 10. == np.round(site[1] * 10.) / 10.)):
            return None, None
        else:
            site_xy = np.array([site[0], site[1]])
            dist  =  np.linalg.norm(layers_pos[0][hlargerthanz,:2] - site_xy, axis = 1)
            if np.any(dist < 2):
                return None, None

    large_cell = np.copy(np.array(large_slab.get_cell()))
    x_length = np.linalg.norm(large_cell[0,:])
    y_length = np.linalg.norm(large_cell[1,:])

    CNmax = get_CNmax(slab[1], crystal)

    CN = 0
    symb_in_shell = []

    site_pos = np.array([x for x in site])
    if site_type == 'hollow':
        site_pos[-1] = np.amax(atom_positions[:,-1])

    unique_heights =  np.sort(np.unique(np.round(atom_positions[:,-1] * 10.) / 10.))
    all_dists = np.round(np.linalg.norm(atom_positions - site_pos, axis = 1) * 10.) / 10.
    original_idxs = np.arange(all_dists.shape[0])[np.round(atom_positions[:,-1] * 10.) / 10. == unique_heights[-1]]
    all_dists = all_dists[original_idxs]
    min_dist = np.amin(all_dists)

    #firt shell
    dist_idxs = np.where(all_dists == min_dist)[0]
    shell_idxs = original_idxs[dist_idxs]
    unique_symb = np.unique(atom_symbols)
    atomic_radii = {}
    for symb in unique_symb:
        loc = df_elements['symbol'] == symb
        atomic_radii[symb] = df_elements[loc]['atomic_radius'].values[0] / 100.

    for idx in shell_idxs:
        symb_in_shell.append(atom_symbols[idx])
        atom_ref = atom_positions[idx, :]
        distsfrom_ref = np.round(np.linalg.norm(atom_positions - atom_ref, axis = 1) * 10.) / 10.
        sortedidxs_bydists = np.argsort(distsfrom_ref[distsfrom_ref != 0])
        sorted_dists = distsfrom_ref[distsfrom_ref != 0][sortedidxs_bydists]
        sorted_symb = atom_symbols[sortedidxs_bydists]

        if crystal != 'Hexagonal':
            neigh_idxs = np.where(distsfrom_ref == sorted_dists[0])[0]
            if site_type == 'ontop':
                for element in neigh_idxs:
                    symb_in_shell.append(atom_symbols[element])
            CN += neigh_idxs.shape[0]
        else:
            for symb in unique_symb:
                symb_idx = np.where(sorted_symb == symb)[0][0]
                neigh_idxs = np.where(distsfrom_ref == sorted_dists[symb_idx])[0]
                if site_type == 'ontop':
                    symb_in_shell += [symb] * neigh_idxs.shape[0]
                CN_tmp = neigh_idxs.shape[0]
                if CN_tmp > CNmax:
                    CN_tmp = CNmax
                CN += CN_tmp

    if site_type != 'ontop':
        #second shell
        all_dists = np.round(np.linalg.norm(atom_positions - site_pos, axis = 1) * 10.) / 10.
        original_idxs = np.arange(all_dists.shape[0])[np.round(atom_positions[:,-1] * 10.) / 10. == unique_heights[-2]]
        all_dists = all_dists[original_idxs]
        min_dist = np.amin(all_dists)
        dist_idxs =  np.where(all_dists == min_dist)[0]
        shell_idxs =  original_idxs[dist_idxs]
        for idx in shell_idxs:
            atom_ref = atom_positions[idx, :]
            if np.round(np.linalg.norm(site_pos[:2] - atom_ref[:2]) * 10.) / 10. >= 0.75*atomic_radii[atom_symbols[idx]]:
                continue
            else:
                if plane != '110' or site_type != 'threefold':
                    symb_in_shell.append(atom_symbols[idx])
                distsfrom_ref = np.round(np.linalg.norm(atom_positions - atom_ref, axis = 1) * 10.) / 10.
                sortedidxs_bydists = np.argsort(distsfrom_ref[distsfrom_ref != 0])
                sorted_dists = distsfrom_ref[distsfrom_ref != 0][sortedidxs_bydists]
                sorted_symb = atom_symbols[sortedidxs_bydists]
         
                if crystal != 'Hexagonal':
                    CN += np.where(distsfrom_ref == sorted_dists[0])[0].shape[0]
                else:
                    for symb in unique_symb:
                        symb_idx = np.where(sorted_symb == symb)[0][0]
                        CN_tmp = np.where(distsfrom_ref == sorted_dists[symb_idx])[0].shape[0]
                        if CN_tmp > CNmax:
                            CN_tmp = CNmax
                        CN += CN_tmp
                

    #Calculate GCN
    GCN = CN /CNmax

    #Making some hard-coded corrections to some GCN
    if plane == '10-10' and site_type == 'bridge':
        GCN += 1
        
    #Get Proportions at active sites and features of each element
    Proportions_dict = {}
    outerE = {}
    eng = {}
    for symb in unique_symb:
        Proportions_dict[symb] = np.where(np.array(symb_in_shell) == symb)[0].shape[0]
        if plane == '10-10' and site_type == 'bridge':
            if Proportions_dict[symb] == 4:
                Proportions_dict[symb] += 2
            if Proportions_dict[symb] == 0 and CN == 8:
                Proportions_dict[symb] += 1

        outerE[symb] = get_outerE(symb)
        eng[symb] = df_elements['en_pauling'][df_elements['symbol'] == symb].values[0]
        if np.isnan(eng[symb]):
            print('WARNING: Pauling eng is not available for '+symb)

    #Compute PSI
    n_active = np.sum([Proportions_dict[symb] for symb in Proportions_dict.keys()])
    outer_p = 1.
    eng_p = 1.
    for symb in unique_symb:
        outer_p *= outerE[symb] ** Proportions_dict[symb]
        eng_p *= eng[symb] ** Proportions_dict[symb]

    PSI = (outer_p ** (2./n_active)) / (eng_p ** (1./n_active))

    return PSI, GCN 
    

def get_from_dataset(df, adsorbate):

    not_valid = ['label','Eads', 'Binding site', 'adsorbate', 'Material']

    tags = [data_tag for data_tag in list(df) if data_tag not in not_valid]

    features = tags

    X = df[features].values
    Y = df['Eads']
    label_features = df[['Binding site', 'Material', 'Biaxial Strain', 'label']]

    mm_X, mm_scaler = get_scaled_features(X[:,:])

    ads_idxs = np.where(df['adsorbate'].values == adsorbate)[0]
    target = Y.iloc[ads_idxs].values[np.where((label_features['Material'].iloc[ads_idxs].values == 'Pt') & (label_features['Biaxial Strain'].iloc[ads_idxs].values == 0) & (label_features['Binding site'].iloc[ads_idxs] == 'fcc'))[0]]

    return target, features, mm_scaler

def evaluate_candidates(mm_X_candidates, count_Eads, Model1_Eads, model_data, label_features, desired_range):
    
    Model_Y = model_data['Model'].predict(mm_X_candidates)
    print('Getting Eads for the candidates to confirm counterfactual...')

    idxs2keep = []
    for i in range(Model_Y.shape[0]):
        print()
        print(*[label_features[name].values[i] for name in label_features.columns])
        print('Count Eads:', count_Eads[i], 'Model1 Eads:', Model1_Eads[i], 'Updated Eads:', Model_Y[i])
        print()

        if Model_Y[i] >= desired_range[0] and Model_Y[i] <= desired_range[1]:
            print('Eads predicted by the model lies within Counterfactual ranges. Candidates is confirmed...')
            idxs2keep.append(i)
        else:
            print('Eads predicted by the model lies outside Counterfactual ranges. Cnadidate will be discarded...')

        print()

    return np.array(idxs2keep), Model_Y


def main(df_candidates, model_file, dataset_file, API_key_file, df_count, df_original):
    
    OS = sys.platform
    if OS == 'win32' or OS == 'cywin':
        folder_lim = '\\'
    else:
        folder_lim = '/'

    df_dataset = pd.read_excel(dataset_file)

    API_key = np.loadtxt(API_key_file,dtype='str')

    with open(model_file, 'rb') as F:
        model_data = pickle.load(F)

    adsorbate_dataset = dataset_file.split(folder_lim)[-1].split('_')[1]

    target, features, scaler = get_from_dataset(df_dataset, adsorbate_dataset)

    ads = df_candidates['adsorbate'].values[0]

    final_candidates = {}
    for feature in df_candidates.columns:
        if feature == 'Model Eads':
            name = 'Model1_Eads'
        elif feature == 'Eads':
            name = 'count_Eads'
        else:
            name = feature

        final_candidates[name] = []

    final_candidates['count_GCN'] = []
    final_candidates['count_PSI'] = []
    final_candidates['site'] = []
    final_candidates['site_type'] = []
    final_candidates['facet'] = []
    

    final_Counterfactuals = None
    final_original_Mats = None
    for i, ID in enumerate(df_candidates['MP ID'].values):
        
        print(df_candidates['Material'].values[i])
        slabs, bad_slabs = get_slab(ID, df_candidates['crystal'].values[i], API_key)
        if not slabs:
            print('We go to the next material')
            continue

        print('VALID SLABS:')
        for plane in slabs.keys():
            print(plane)
            ads_sites = get_sites(slabs[plane][0])
            slabs[plane] = {'slab' : slabs[plane],
                            'sites' : ads_sites
                           }
            for site_type in slabs[plane]['sites'].keys():
                if not slabs[plane]['sites'][site_type]:
                    continue
                print('\t', site_type)
                for site in slabs[plane]['sites'][site_type]:
                    print('\t',site)
                    PSI, GCN = get_features(slabs[plane]['slab'], site, df_candidates['crystal'].values[i], site_type, plane)
                    print()
                    if PSI is None:
                        print('Site is not valid\n')
                        print()
                        continue
                    print('GCN - ', 'Count:', df_candidates['GCN'].values[i],'Found:', GCN)
                    print('PSI - ', 'Count:', df_candidates['PSI'].values[i],'Found:', PSI)
                    if df_candidates['GCN'].values[i] * 1.2 >= GCN and df_candidates['GCN'].values[i] * 0.8 <= GCN:
                        if df_candidates['PSI'].values[i] + 15 >= PSI and df_candidates['PSI'].values[i] - 15 <= PSI:
                            for feature in df_candidates.columns:
                                if feature == 'GCN':
                                    final_candidates[feature].append(GCN)
                                    final_candidates['count_GCN'].append(df_candidates[feature].values[i])
                                elif feature == 'PSI':
                                    final_candidates[feature].append(PSI)
                                    final_candidates['count_PSI'].append(df_candidates[feature].values[i])
                                elif feature == 'Eads':
                                    final_candidates['count_Eads'].append(df_candidates[feature].values[i])
                                elif feature == 'Model Eads':
                                    final_candidates['Model1_Eads'].append(df_candidates[feature].values[i])
                                else:
                                    final_candidates[feature].append(df_candidates[feature].values[i])

                            final_candidates['facet'].append(plane)
                            final_candidates['site'].append(site)
                            final_candidates['site_type'].append(site_type)
                            print('Site is valid\n')
                            if final_Counterfactuals is None:
                                final_Counterfactuals = df_count.iloc[i]
                                final_original_Mats = df_original.iloc[i]
                            else:
                                final_Counterfactuals = pd.concat([final_Counterfactuals, df_count.iloc[i]], axis=0, ignore_index=True)
                                final_original_Mats = pd.concat([final_original_Mats, df_original.iloc[i]], axis=1,ignore_index=True)
                                
                        else:
                            print('Site is not valid\n')
                    else:
                        print('Site is not valid\n')

        print()
 
        if not slabs:
            print('We go to the next material')
            continue

    print(final_Counterfactuals)
    print()
    print()
    print(final_original_Mats)
    exit()
    df_f_candidates = pd.DataFrame.from_dict(final_candidates)

    desired_range = [target - 0.1, target + 0.1]

    X_candidates = df_f_candidates[features].values
    mm_X_candidates = scaler.transform(X_candidates)
    
    candidates_labels = df_f_candidates[['Material', 'crystal', 'facet', 'site_type']]
    
    idxs2keep, model_Y = evaluate_candidates(mm_X_candidates, df_f_candidates['count_Eads'], df_f_candidates['Model1_Eads'], model_data, candidates_labels, desired_range)
    
    df_f_candidates['Eads'] = model_Y
    
    df_f_candidates = df_f_candidates.iloc[idxs2keep]
    final_Counterfactuals = final_Counterfactuals.iloc[idxs2keep]
    final_original_Mats = final_original_Mats.iloc[idxs2keep]

    print(df_f_candidates)
    
    filename = 'Final_Candidates_'+ads+'.xlsx'
    
    df_f_candidates.to_excel(filename, index=False)
    final_Counterfactuals.to_excel('Counterfactuals_'+ads+'.xlsx', index=False)
    final_original_Mats.to_excel('Original_Materials_'+ads+'.xlsx', index=False)

if __name__ == '__main__':
    main()
