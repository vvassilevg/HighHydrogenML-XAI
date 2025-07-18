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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
import shap

def get_scaled_features(X):

    #MinMax Normalization
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(X)
    mm_X = mm_scaler.transform(X)

    return mm_X, mm_scaler

def rename_features(features):

    names_dict = {'Biaxial Strain' : 'Strain',
                  'outer electrons A' : r'S$_A$',
                  'outer electrons B' : r'S$_B$',
                  'Unit cell volume' : 'Volume',
                 }

    new_names = []
    for feature in features:
        if feature in names_dict.keys():
            new_names.append(names_dict[feature])
        else:
            new_names.append(feature)

    return new_names

def _check_feature_importance(model_data, features, X, Y, adsorbate, label_features, PWD_figures, folder_lim, scaler):

    features4plot = np.array(rename_features(features))

    PWD4save = PWD_figures+folder_lim+'Shap'
    if not os.path.isdir(PWD4save):
        os.makedirs(PWD4save)
    
    perform_shap(model_data, features, X, Y, adsorbate, label_features, PWD4save+folder_lim, scaler)


def perform_shap(model_data, features, X, Y, adsorbate, label_features, PWD, scaler):

    X_shap = shap.utils.sample(X,int(np.round(.15 * X.shape[0])))
    Y_shap = np.copy(Y)
    Materials = label_features['Material'].values
    Sites = label_features['Binding site'].values

    features4plot = np.array(rename_features(features))
    df_X = pd.DataFrame(X[model_data['train_idxs'], :], columns=features4plot)
    Y4shap = Y[model_data['train_idxs']]
    df_X_shap = pd.DataFrame(X_shap, columns=features4plot)

    xticks = np.array([0,0.2,0.4,0.6,0.8,1])
    xticks_arr = np.tile(xticks[:,None], (1, features4plot.shape[0]))


    explainer_model = shap.Explainer(model_data['Model'], df_X_shap)
    shap_values_model = explainer_model(df_X)
    shap_values_test = explainer_model(df_X_shap)

    
    features4dependence = ['GCN']
    features4combined = {'H' : [('PSI', 'WEN')],
                         'O' : [('WIE', 'WEN'), ('WAR', 'PSI')],
                         'OH' : [('WAR', 'PSI'), ('WIE', 'WEN')],
                        }

    for i, feature in enumerate(features4plot):

        if feature in features4dependence:
            #Partial dependence plot
            fig, ax = shap.partial_dependence_plot(
                feature,
                model_data['Model'].predict,
                df_X_shap,
                model_expected_value = False,
                feature_expected_value =  False,
                show= False,
                ice=False,
                ylabel=r'Expected E$_{ads}$ (eV)',
                hist= False,
                )
            
            xticks = np.array([0,0.2,0.4,0.6,0.8,1])
            xticks_arr = np.tile(xticks[:,None], (1, features4plot.shape[0]))
            xtick_labels = np.round(scaler.inverse_transform(xticks_arr)[:,i].ravel() * 10.) / 10.
 
            plt.xticks(xticks, labels=xtick_labels, fontsize=25) 
            plt.yticks(fontsize=16)
            plt.tight_layout()
           
            DATA = ax.get_children()[0].get_data()
            DATA_X_arr = np.tile(DATA[0][:,None], (1, features4plot.shape[0]))
            DATA_X = scaler.inverse_transform(DATA_X_arr)[:,i].ravel()
       
            DATA2save = np.concatenate((DATA_X[:,None], DATA[1].ravel()[:,None]), axis=1)
            file_header = '#'+feature + '\t' + 'Expected Eads (eV)'
            filename = 'PartialDependence_'+feature+'_'+adsorbate+'.txt'
            np.savetxt(filename, DATA2save, header=file_header, fmt='%.8f')
            
            #plt.show()

        shap_values_model_data = scaler.inverse_transform(shap_values_model.data)


        for pairs in features4combined[adsorbate]:
           feature1_idx = np.where(features4plot == pairs[0])[0]
           feature2_idx = np.where(features4plot == pairs[1])[0]
           DATA_x = shap_values_model_data[:,feature1_idx[0]].ravel()[:,None]
           DATA_z = shap_values_model_data[:,feature2_idx[0]].ravel()[:,None]
           DATA_y = shap_values_model[:,pairs[0]].values + shap_values_model[:, pairs[1]].values

           DATA2save = np.concatenate((DATA_x, DATA_y.ravel()[:,None], DATA_z), axis=1)
           file_header ='#'+pairs[0]+'\t'+'Combined SHAP value (eV)'+'\t'+pairs[1]
           filename = 'CombinedSHAP_'+pairs[0]+'vs'+pairs[1]+'_'+adsorbate+'.txt'
           np.savetxt(filename, DATA2save, header = file_header, fmt='%.8f')
  

    
    plt.clf()
    shap.plots.beeswarm(shap_values_model, show=False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(PWD+'Shap_Beeswarm_'+adsorbate+'.svg', dpi=300, format='svg')
    plt.show()


def main():

    OS = sys.platform
    if OS == 'win32' or OS == 'cywin':
        folder_lim = '\\'
    else:
        folder_lim = '/'

    model_file = sys.argv[1]
    dataset_file = sys.argv[2]

    df = pd.read_excel(dataset_file)

    #Load model data
    with open(model_file, 'rb') as F:
        model_data = pickle.load(F)

    adsorbate = dataset_file.split(folder_lim)[-1].split('_')[1]

    DATE = '_'.join(datetime.today().strftime('%Y-%m-%d').split())

    PWD_figures = 'Figures'+folder_lim+'Feature_Importance'

    PWD_figures += folder_lim+DATE+folder_lim+adsorbate+folder_lim+model_file.split(folder_lim)[-1].split('.pickle')[0].split('_')[0]+folder_lim+model_file.split(folder_lim)[-2]

    not_valid = ['Eads', 'Binding site', 'adsorbate', 'Material', 'label']

    tags = [data_tag for data_tag in list(df) if data_tag not in not_valid]

    features = tags

    print('Features:')
    print(features)
    print()

    X = df[features].values
    Y = df['Eads'].values
    label_features = df[['Binding site', 'Material']]

    mm_X, mm_scaler = get_scaled_features(X[:,:])

    scaled_X = mm_X

    _check_feature_importance(model_data, features, scaled_X, Y, adsorbate, label_features, PWD_figures, folder_lim, mm_scaler)


if __name__ == '__main__':
    main()

