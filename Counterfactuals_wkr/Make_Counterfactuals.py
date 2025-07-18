from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import pandas as pd
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler
import pickle
import dice_ml
from dice_ml.utils import helpers
from mendeleev import element
from mendeleev.fetch import fetch_table, fetch_ionization_energies
import math
from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase import Atoms

df_elements = fetch_table('elements')
df_ionenergy  = fetch_ionization_energies()

missing_eng_pauling = {'36' : 3.00, #Kr
                       '61' : 1.13, #Pm
                       '63' : 1.2,  #Eu
                       '65' : 1.1,  #Tb
                       '70' : 1.1,  #Yb
                       '86' : 2.2,  #Rn
                       '95' : 1.13, #Am
                       '96' : 1.28, #Cm
                       '97' : 1.3,  #Bk
                       '98' : 1.3,  #Cf
                       '99' : 1.3,  #Es
                       '100' : 1.3, #Fm
                       '101' : 1.3, #Md
                       '102' : 1.3, #No
                       '103' : 1.3, #Lr
                      }
def get_weighted_feature(feature, Material):

    feature2property = {'WEN' : 'en_pauling',
                        'WAR' : 'atomic_radius',
                        'WIE' : 'ionenergy',
                       }

    atoms = Atoms(Material)
    numbers = list(atoms.get_atomic_numbers())

    weighted_feature = 0

    for n in set(numbers):
        loc = df_elements['atomic_number'] == n
        data = df_elements[loc]
        weight = numbers.count(n) / len(numbers)
        value = data[feature2property[feature]].values[0] if feature != 'WIE' else df_ionenergy['IE1'].values[n-1]
        if feature == 'WEN':
            if np.isnan(value):
                value = missing_eng_pauling[str(n)]
        weighted_feature += weight * value

    return weighted_feature
      
def get_composition(Material):
    elements = []
    composition = []
    symb =''
    upper=False
    number=False
    len_Matstr = len(Material.split()[0])
    if len_Matstr == 1:
        elements.append(Material.split()[0])
    else:
        for i, char in enumerate(Material.split()[0]):
            if char.isalpha():
                if not upper:
                    symb += char
                    if char.isupper():
                        upper = True
                        if i == len_Matstr - 1:
                            elements.append(symb)
                        if i == len_Matstr - 2 and not Material.split()[0][-1].isalpha():
                            elements.append(symb)
                    if len(elements) >= 1: 
                        if not number:
                            composition.append(1)
                        else:
                            number = False
                else:
                    if char.islower():
                        symb += char
                        elements.append(symb)
                        symb = ''
                        upper = False
                    else:
                        elements.append(symb)
                        symb = ''
                        symb += char
                        upper = True
                        if not number:
                            composition.append(1)
                        else:
                            number = False
                        if i == len_Matstr - 1:
                            elements.append(symb)
                        if i == len_Matstr - 2 and not Material.split()[0][-1].isalpha():
                            elements.append(symb)
            else:
                composition.append(int(char))
                number = True

    if len(composition) != len(elements):
        composition.append(1)

    return elements, tuple(composition)

def get_scaled_features(X):

    #MinMax Normalization
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(X)
    mm_X = mm_scaler.transform(X)

    scaler = mm_scaler

    return mm_X, scaler

def check_sample(df_row, features, Full_data):
    
    not_needed = ['PSI', 'GCN', 'Biaxial Strain', 'Unit cell volume', 'Eads']

    idxs2check = np.arange(Full_data['Eads'].values.shape[0])
    for i, feature in enumerate(features):
        if feature in not_needed:
            continue
        Full_data_tmp = np.round(Full_data[feature].values[idxs2check] * 100000.) / 100000.
        equal_idxs = np.where(Full_data_tmp == np.round(df_row[feature] * 100000.) / 100000.)[0]
        if equal_idxs.shape[0] == 0:
            print(feature, ': There is no equal sample... Counterfactual is valid (Type 1)')
            return 1, None
        idxs2check = idxs2check[equal_idxs]

    print('Checking if at least PSI and GCN are different...')
    PSI_idxs =  np.where(np.round(Full_data['PSI'].values[idxs2check] * 100000.) / 100000. == np.round(df_row['PSI'] * 100000.) / 100000.)[0]
    if PSI_idxs.shape[0] == 0:
        print('PSI is different... Counterfactual is valid (Type 2)')
        return 2, None
    else:
        idxs2check = idxs2check[PSI_idxs]
        GCN_idxs = np.where(np.round(Full_data['GCN'].values[idxs2check] * 100000.) / 100000. == np.round(df_row['GCN'] * 100000.) / 100000.)[0]
        if GCN_idxs.shape[0] == 0:
            print('GCN is different... Counterfactual is valid (Type 2)')
            return 2, None
        else:
            return 0, idxs2check[GCN_idxs]

def find_candidate(s1_data, s2_data, df_row):

    property2weighted = {'en_pauling' : 'WEN',
                         'atomic_radius' : 'WAR',
                         'ionenergy' : 'WIE',
                        }
    A = np.zeros([3,2])
    b = []
    for i, p in enumerate(s1_data.keys()):
        A[i,:] = np.array([s1_data[p], s2_data[p]])
        b.append(df_row[property2weighted[p]])

    composition, residuals, ranl, s = np.linalg.lstsq(A,b)

    if np.any(composition >= 1) or np.any(composition <= 0):
        return None
    
    else:
        composition = np.round(composition * 10.).astype(int)
        if np.any(composition == 0):
            return None
        else:
            GCD = math.gcd(composition[0], composition[1])
            composition = composition / GCD
            
            return composition.astype(int)

def find_samples(original_Mat, df_row, valid_samples, original_df, API_key):
    print('Original Sample:', original_Mat)
    elements, comp = get_composition(original_Mat)
    print(elements, comp)

    original_df.loc[:,'Material'] = original_Mat

    not_valid_elements = ['H', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Sr', 'Ba', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'B', 'Tl', 'Nh', 'Ge', 'C', 'Pb', 'Fl']

    RE_dict = {'La' : 3,
               'Ce' : 4,
               'Pr' : 5,
               'Nd' : 6,
               'Pm' : 7,
               'Sm' : 8,
               'Eu' : 9,
               'Gd' : 10,
               'Tb' : 11,
               'Dy' : 12,
               }

    symbols_dict = {'1': [],
                    '2': [],
                   }

    for i, symb in enumerate(elements):
        symbols_dict[str(i+1)].append(symb)
        symb_idx = np.where(df_elements['symbol'].values == symb)[0]
        if symb in RE_dict:
            groups = [RE_dict[symb]]
        else:
            groups = [df_elements['group_id'][symb_idx].values[0]]

        if groups[0] == 3:
            groups += [13]
        elif groups[0] == 13:
            groups += [3]

        for group in groups:
            idxs2save = np.where(df_elements['group_id'].values == group)[0]
            symbols2save = df_elements['symbol'].values[idxs2save]
            for s in symbols2save:
                if s not in symbols_dict[str(i+1)] and s not in not_valid_elements:
                    symbols_dict[str(i+1)].append(s)
            for s in RE_dict.keys():
                if RE_dict[s] == group:
                    if s not in symbols_dict[str(i+1)]:
                        symbols_dict[str(i+1)].append(s)
                    break

    for s1 in symbols_dict['1']:
        for s2 in symbols_dict['2']:
            if s1 == s2:
                continue
            s1_idx = np.where(df_elements['symbol'].values == s1)[0]
            s2_idx = np.where(df_elements['symbol'].values == s2)[0]
            properties = ['en_pauling', 'atomic_radius', 'ionenergy']
            s1_data = {}
            s2_data = {}
            for p in properties:
                s1_data[p] = df_elements[p].values[s1_idx][0] if p != 'ionenergy' else df_ionenergy['IE1'].values[s1_idx][0]
                s2_data[p] = df_elements[p].values[s2_idx][0] if p != 'ionenergy' else df_ionenergy['IE1'].values[s2_idx][0]

                if p == 'en_pauling':
                    if np.isnan(s1_data[p]):
                        s1_data[p] = missing_eng_pauling[str(s1_idx[0]+1)]
                    if np.isnan(s2_data[p]):
                        s2_data[p] = missing_eng_pauling[str(s2_idx[0]+1)]

            composition = find_candidate(s1_data, s2_data, df_row)
            
            if composition is not None:

                if np.any(composition == 1):
                    one_idxs = np.where(composition ==  1)[0]
                    if one_idxs.shape[0] == 2:
                        subscripts = ['','']
                    else:
                        subscripts = ['', str(composition[1])] if one_idxs[0] == 0 else [str(composition[0]), '']
                else:
                    subscripts = [str(composition[0]), str(composition[1])]
                
                formula = s1+subscripts[0]+s2+subscripts[1]
                formula_v2 = s2+subscripts[1]+s1+subscripts[0]
                    
                with MPRester(str(API_key)) as mpr:
                    docs = mpr.summary.search(formula=formula, fields=["material_id", 'symmetry', 'structure', 'energy_above_hull', 'volume'])

                if len(docs) > 0:
                    if formula not in valid_samples.keys() and formula_v2 not in valid_samples.keys():
                        if formula != original_Mat and formula_v2 != original_Mat:
                            valid_samples[formula] = [docs, df_row, original_df]

                print()

    return valid_samples

def idxsfromlabels(labels, Y, target):

    final_idxs = []
    for label in np.unique(labels.values):
        original_idxs = np.where(labels.values == label)[0]
        Y_label = Y[original_idxs]
        kept_idx = np.argmin(np.abs(Y_label - target[0]))

        final_idxs.append(original_idxs[kept_idx])

    return np.array(final_idxs)

        

def _make_counterfactuals(model_data, features, X, Y, adsorbate, label_features, target, API_key, scaler=None):

    Full_data = X.copy()
    Full_data_original = scaler.inverse_transform(Full_data.values)
    Full_data.loc[:, 'Eads'] = Y.values[:,None]
    Full_data_original = pd.DataFrame(Full_data_original, columns= features)
    Full_data_original.loc[:, 'Eads'] = Y.values[:, None]

    datafromtest = False
    zero_idxs = np.where(label_features['Biaxial Strain'].values == 0)[0]
    labels_zero = label_features.iloc[zero_idxs]
    unique_label_idxs = idxsfromlabels(labels_zero['label'], Y.values[zero_idxs], target)
    X4Count = X.iloc[zero_idxs].iloc[unique_label_idxs]
    Materials_test = labels_zero['Material'].values[unique_label_idxs]
    Sites_test = labels_zero['Binding site'].values[unique_label_idxs]
    print('Number of materials:', X4Count.shape[0])

    desired_range = [target[0] - 0.1, target[0] + 0.1] if adsorbate == 'H' else [target[0] - 0.15, target[0] + 0.15]

    print('TARGET Eads:', desired_range)
    print()

    d = dice_ml.Data(dataframe=Full_data, continuous_features=features, outcome_name = 'Eads')

    m = dice_ml.Model(model=model_data['Model'], backend='sklearn', model_type='regressor')

    if adsorbate != 'OH':
        exp = dice_ml.Dice(d, m, method='genetic') #kdtree genetic random
    else:
       exp = dice_ml.Dice(d, m, method='random')

    if 'adsorbate' in features:
        features2vary = X4Count.drop(['Unit cell volume', 'outer electrons A', 'outer electrons B', 'adsorbate'], axis=1).columns.tolist()
    else:
        features2vary = X4Count.drop(['Unit cell volume', 'outer electrons A', 'outer electrons B'], axis=1).columns.tolist()


    used_idxs = np.arange(X4Count.values.shape[0])

    counterfactual = exp.generate_counterfactuals(X4Count[:100], total_CFs=4, desired_range=desired_range, features_to_vary=features2vary)

    cfs_lists = counterfactual.visualize_as_dataframe(show_only_changes=True, print_dfs =False)

    print()

    original_df = []
    final_cfs = []
    valid_candidates = {}
    for i, cf_list in enumerate(cfs_lists):
        print()
        print()
        print()
        print('Sample:', Materials_test[used_idxs[i]], Sites_test[used_idxs[i]])
        
        test_real_values = scaler.inverse_transform(cf_list.test_instance_df.drop('Eads', axis=1).values)
        test_df = pd.DataFrame(test_real_values, columns= features)
        test_df.loc[:,'Eads'] = cf_list.test_instance_df['Eads'].values[:,None]
        print('ORIGINAL VALUES')
        print(test_df)
        print()
        try:
           cfs_real_values = scaler.inverse_transform(cf_list.final_cfs_df_sparse.drop('Eads', axis=1).values)
        except AttributeError:
            print('It was not possible to find valid counterfactuals for '+Materials_test[used_idxs[i]]+' '+Sites_test[used_idxs[i]])
            continue

        cfs_df = pd.DataFrame(cfs_real_values, columns=features)
        cfs_df.loc[:,'Eads'] = cf_list.final_cfs_df_sparse['Eads'].values[:,None]
        print('COUNTERFACTUALS')
        print(cfs_df)

        for index, row in cfs_df.iterrows():
            print('Counterfactual '+str(index))
            validity, eq_idxs = check_sample(row, features+['Eads'], Full_data_original)
            if validity == 0:
                print('Counterfactual is equal to', label_features['Material'].values[eq_idxs[0]],label_features['Binding site'].values[eq_idxs[0]])
                continue
            elif validity == 1:
                valid_candidates = find_samples(Materials_test[used_idxs[i]], row, valid_candidates, test_df, API_key)
            else:
                pass

    final_candidates = {}
    for name in valid_candidates.keys():
        if name in label_features['Material'].values:
            print(name+' is already in the dataset and will not be considered...')
        else:
            final_candidates[name] = valid_candidates[name]


    print()
    print()
    print()
    print('CANDIDATES RETRIEVED:')
    print(list(final_candidates.keys()))

    return final_candidates, desired_range
                
def get_candidates_df(Candidates, adsorbate):

    columns = ['Biaxial Strain', 'PSI', 'outer electrons A', 'outer electrons B', 'Unit cell volume', 'WEN', 'WIE', 'WAR', 'GCN', 'Eads', 'Material', 'adsorbate', 'hull', 'MP ID','crystal']

    from_counterfactual = ['Biaxial Strain', 'PSI',  'outer electrons A', 'outer electrons B', 'GCN', 'Eads']
    weighted_features = ['WEN', 'WIE', 'WAR']

    dict4df = {}
    for feature in columns:
        dict4df[feature] = []

    counterfactuals = {}
    original_Mats = {}
    for Material in Candidates.keys():
        n_docs = len(Candidates[Material][0])
        for doc in Candidates[Material][0]:
            print()
            print(Material)
            hull = doc.energy_above_hull
            if hull == 0:
                print('The material is stable')
            else:
                print('The material is not stable')

            crystal_system = doc.symmetry.crystal_system

            if hull == 0 or crystal_system in ['Cubic', 'Hexagonal']:
                str2print = 'But t' if hull != 0 else 'T'

                print(str2print+'he material is '+str(crystal_system))
                print('The material will be saved')
                Save =  True
            elif n_docs == 1:
                print('The material is '+str(crystal_system))
                print('However, the material will be saved since it is the only available')
                Save = True
            else:
                print('The material will be discarded because hull='+str(hull)+' and it is '+str(crystal_system))
                Save = False
           
            print()
            if Save:
                for name in Candidates[Material][1].keys():
                    if name not in counterfactuals.keys():
                        counterfactuals[name] = [Candidates[Material][1][name]]
                    else:
                        counterfactuals[name].append(Candidates[Material][1][name])
                for name in Candidates[Material][2].columns:
                    if name not in original_Mats.keys():
                        original_Mats[name] = [Candidates[Material][2][name].values[0]]
                    else:
                        original_Mats[name].append(Candidates[Material][2][name].values[0])

                for feature in columns:
                    if feature in from_counterfactual:
                        dict4df[feature].append(Candidates[Material][1][feature])
                    elif feature in weighted_features:
                        dict4df[feature].append(get_weighted_feature(feature, Material))

                        print(feature, 'Count:'+str(Candidates[Material][1][feature]), 'Real:'+str(dict4df[feature][-1]))
                    else:
                        if feature == 'Unit cell volume':
                            dict4df[feature].append(doc.volume)
                        if feature == 'Material':
                            dict4df[feature].append(Material)
                        if feature == 'adsorbate':
                            dict4df[feature].append(adsorbate)
                        if feature == 'hull':
                            dict4df[feature].append(hull)
                        if feature == 'MP ID':
                            dict4df[feature].append(doc.material_id)
                        if feature == 'crystal':
                            dict4df[feature].append(crystal_system)
            else:
                continue

    return pd.DataFrame.from_dict(dict4df), pd.DataFrame.from_dict(counterfactuals), pd.DataFrame.from_dict(original_Mats)

def evaluate_candidates(mm_X_candidates, Counterfactual_Y, model_data, label_features, desired_range):

    Model_Y = model_data['Model'].predict(mm_X_candidates)
    print('Getting Eads for the candidates to confirm counterfactual...')

    idxs2keep = []
    for i in range(Model_Y.shape[0]):
        print()
        print(label_features['Material'].values[i], label_features['crystal'].values[i])
        print('Counterfactual Eads:', Counterfactual_Y[i], 'Candidate Eads:', Model_Y[i])

        if Model_Y[i] >= desired_range[0] and Model_Y[i] <= desired_range[1]:
            print('Eads predicted by the model lies within Counterfactual ranges. Candidate is confirmed...')
            idxs2keep.append(i)
        else:
            print('Eads predicted by the model lies outside Counterfactual ranges. Candidate will be discarded...')

    return np.array(idxs2keep), Model_Y


def main(model_file, dataset_file, API_key_file):
    
    OS = sys.platform
    if OS == 'win32' or OS == 'cywin':
        folder_lim = '\\'
    else:
        folder_lim = '/'

    API_key = np.loadtxt(API_key_file,dtype='str')

    df = pd.read_excel(dataset_file)

    #Load model data
    with open(model_file, 'rb') as F:
        model_data = pickle.load(F)

    ads = dataset_file.split(folder_lim)[-1].split('_')[1]

    DATE = '_'.join(datetime.today().strftime('%Y-%m-%d').split())

    not_valid = ['label','Eads', 'Binding site', 'adsorbate', 'Material']

    tags = [data_tag for data_tag in list(df) if data_tag not in not_valid]

    features = tags

    print('Features:')
    print(features)
    print()

    X = df[features].values
    Y = df['Eads']
    label_features = df[['Binding site', 'Material', 'Biaxial Strain', 'label']]

    mm_X, mm_scaler = get_scaled_features(X[:,:])

    scaled_X = mm_X

    scaled_X_df = pd.DataFrame(scaled_X, columns = features)

    print('DATA:')
    print(scaled_X_df)


    ads_idxs = np.where(df['adsorbate'].values == ads)[0]

    target = Y.iloc[ads_idxs].values[np.where((label_features['Material'].iloc[ads_idxs].values == 'Pt') & (label_features['Biaxial Strain'].iloc[ads_idxs].values == 0) & (label_features['Binding site'].iloc[ads_idxs] == 'fcc'))[0]]

    Candidates, desired_range = _make_counterfactuals(model_data, features, scaled_X_df.iloc[ads_idxs], Y.iloc[ads_idxs], ads, label_features.iloc[ads_idxs],  target, API_key, scaler=mm_scaler)

    df_candidates, df_counterfactuals, df_original_Mats = get_candidates_df(Candidates, ads)

    X_candidates = df_candidates[features].values
    Counterfactual_Y = df_candidates['Eads']
    candidates_labels = df_candidates[['Material', 'Biaxial Strain', 'crystal']]
    
    mm_X_candidates = mm_scaler.transform(X_candidates)
 
    idxs2keep, model_Y = evaluate_candidates(mm_X_candidates, Counterfactual_Y, model_data, candidates_labels, desired_range)
     
    df_candidates['Model Eads'] = model_Y
     
    df_candidates = df_candidates.iloc[idxs2keep]
    df_counterfactuals = df_counterfactuals.iloc[idxs2keep]
    df_original_Mats = df_original_Mats.iloc[idxs2keep]
   
    return df_original_Mats, df_counterfactuals, df_candidates 

if __name__ == '__main__':
    main()
