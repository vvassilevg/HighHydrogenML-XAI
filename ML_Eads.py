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
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
import pickle
import xgboost

def _cross_validation(data_size, train_size, pure_idxs, test_idxs):

    n_splits = int(np.round(1/train_size)) if train_size <= 0.5 else int(np.round(1/(1-train_size)))

    remain_size = data_size - pure_idxs.shape[0] - test_idxs.shape[0]

    split_size = int(remain_size / n_splits)
    last_split_size = remain_size - (split_size * (n_splits - 1))


    splits = []
    excl_idxs = np.concatenate((pure_idxs, test_idxs))
    for i in range(n_splits):
        print('Split:', i+1)
        n = split_size if i < n_splits - 1 else last_split_size
        remain_idxs = np.delete(np.arange(data_size), excl_idxs)
        rng = np.random.default_rng(seed=876)
        split_idxs = remain_idxs[rng.choice(remain_idxs.shape[0], size=n, replace=False)]
        splits.append(np.copy(split_idxs))
        print('Size of current split:', split_idxs.shape[0])
        print()
        excl_idxs = np.concatenate((np.copy(excl_idxs), np.copy(split_idxs))).ravel()

    for i in range(len(splits)):
        for j in range(len(splits)):
            if i != j:
                assert np.intersect1d(splits[i], splits[j]).shape[0] == 0, 'splits %i and %i have repeated samples' % (i+1, j+1)

    return splits #Splits do not contain Pure Metal samples

def _in_house_cv_sets(trainval_size, pure_idxs, data_size, test_idxs):

    print('Creating sets with cross-validation...')
    n_rand = int(np.random.rand(1)[0] * 100000)


    #We force a 10-fold CV
    train_prop = 0.9

    one_out_type = 'training' if trainval_size <= 0.5 else 'validation'
    cv_splits = _cross_validation(data_size, train_prop, pure_idxs, test_idxs)
    n_train = int(np.round(data_size * trainval_size * train_prop))

    test_sets = []
    train_sets = []
    val_sets = []
    for i in range(len(cv_splits)):
        one_out = cv_splits[i]
        rest_splits = np.concatenate([cv_splits[j] for j in range(len(cv_splits)) if j != i]).ravel()
        if one_out_type == 'training':
            train_idxs_tmp = one_out
            val_idxs_tmp = rest_splits
        else:
            train_idxs_tmp = rest_splits
            val_idxs_tmp = one_out
    
        train_idxs, val_idxs = add_pure_idxs(train_idxs_tmp, val_idxs_tmp, pure_idxs, n_train)
        train_sets.append(train_idxs)
        val_sets.append(val_idxs)
     
    return train_sets, val_sets

def get_initial_samples(system, sites):


    Pures = ['Ag', 'Au', 'Cd', 'Co', 'Cu', 'Fe', 'Hf', 'Ir', 'Mo', 'Nb', 'Ni', 'Os', 'Pd', 'Pt', 'Re', 'Rh', 'Ru','Ta', 'Tc', 'V', 'W', 'Y', 'Zn', 'Zr']
    invalid_ontop = ['Mg', 'Ca', 'Al','Ga', 'Sn', 'Ti', 'Sc', 'Dy', 'In', 'Cr', 'Li', 'Sm', 'Mn', 'Nd', 'Si', 'Pr', 'Tb']
    sites4ontop = ['ontopA', 'ontopB', 'fccAAA', 'hcpAAA', 'hcpAAA_A', 'hcpAAA_B', 'longbridgeA', 'longbridgeB']
    siteswB = ['ontopB', 'longbridgeB']

    all_idxs = np.arange(system.shape[0])

    pure_tokens = np.array([True if len(system[i]) <= 2 else False for i in range(system.shape[0])])

    tmp_train_idxs = all_idxs[pure_tokens]

    weird_idxs = []
    for i in range(system.shape[0]):
        if i in tmp_train_idxs:
            continue

        site2check = sites[i].split('-')[0] if '-d' in sites[i] else sites[i]
        if site2check in sites4ontop:
            upper = 0
            A=''
            B=''
            do_A = True
            for letter in system[i]:
                if letter.isupper():
                    upper += 1
                if upper > 1:
                    do_A = False
                if do_A:
                    A+=letter
                else:
                    B+=letter

            target = B if site2check in siteswB else A 
            token_systems = np.array([True if S in target else False for S in Pures])

            
            if not token_systems.any():
                weird_idxs.append(i)
            
        train_idxs = np.concatenate((tmp_train_idxs, np.array(weird_idxs))).ravel()

    assert train_idxs.shape[0] == np.unique(train_idxs).shape[0], 'Training idxs have repeated samples. Please verify...'

    return train_idxs

def add_pure_idxs(train_idxs, test_idxs_tmp,pure_idxs,n_train):

    n = (pure_idxs.shape[0] + train_idxs.shape[0]) - n_train
    assert n > 0, "The sum of pure idxs an current number of training idxs is lower than the total number of training points needed. Please check..."

    rng = np.random.default_rng()
    rm_idxs = np.arange(train_idxs.shape[0])[rng.choice(train_idxs.shape[0], size = n, replace=False)]
    new_test = np.copy(train_idxs)[rm_idxs]
    up_train_idxs = np.delete(train_idxs, rm_idxs)
    final_train_idxs = np.concatenate((up_train_idxs, pure_idxs)).ravel()
    up_other_idxs = np.concatenate((test_idxs_tmp, new_test)).ravel()

    test_idxs = up_other_idxs 

    return final_train_idxs, test_idxs

def get_scaled_features(X):

    #MinMax Normalization
    mm_scaler = MinMaxScaler()
    mm_scaler.fit(X)
    mm_X = mm_scaler.transform(X)

    scaler = mm_scaler

    return mm_X, scaler

def plot_scatter(best_pred, for_outliers =  False, PWD_figures = None):

    OS = sys.platform
    if OS == 'win32' or OS == 'cywin':
        folder_lim = '\\'
    else:
        folder_lim = '/'

    n_methods = len(best_pred.keys())

    for name in best_pred.keys():
        plt.figure(figsize = (6,8))
        if for_outliers:
            alpha = 0.5
        else:
            alpha = 1
        
        print('MAE:', best_pred[name]['MAE'])
        print('RMSE:', best_pred[name]['RMSE'])
        r2 = metrics.r2_score(best_pred[name]['true_test'], best_pred[name]['test'])
        abs_errors = np.abs(best_pred[name]['true_test'] - best_pred[name]['test'])
        print('R2:', r2)
        error_0 = np.array([np.amin(np.concatenate((best_pred[name]['true_train'], best_pred[name]['true_test'], best_pred[name]['true_val']))), np.amax(np.concatenate((best_pred[name]['true_train'], best_pred[name]['true_test'], best_pred[name]['true_val'])))])
        error_0 += np.array([-3000, 3000])
        plt.plot(error_0, error_0, color='black')
        plt.scatter(best_pred[name]['true_train'], best_pred[name]['train'], color = 'royalblue', label='Train', alpha = alpha)
        plt.scatter(best_pred[name]['true_val'], best_pred[name]['val'], color = 'forestgreen', label='Validation', alpha=alpha)
        plt.scatter(best_pred[name]['true_test'], best_pred[name]['test'], color = 'firebrick', label= 'Test', alpha=alpha)
        min_lim = np.amin(np.concatenate((best_pred[name]['true_train'], best_pred[name]['true_test'], best_pred[name]['true_val'])))
        max_lim = np.amax(np.concatenate((best_pred[name]['true_train'], best_pred[name]['true_test'], best_pred[name]['true_val'])))
        if for_outliers:
            if 'outliers_idxs' in best_pred[name].keys():
                test_set = best_pred[name]['train_test_sets']['test']
                test_outliers = best_pred[name]['outliers_idxs']
                plt.scatter(best_pred[name]['true_test'][test_outliers], best_pred[name]['test'][test_outliers], s=55., color='firebrick', label='Outlier',edgecolors = 'black')
                for i, point in enumerate(test_outliers):
                    x_axis = best_pred[name]['true_test'][point]
                    y_axis = best_pred[name]['test'][point]
                    x_axis = x_axis - (max_lim - min_lim) *0.06 if y_axis > x_axis else x_axis + (max_lim - min_lim)*0.02
                    plt.text(x_axis,  y_axis, str(i+1), horizontalalignment='left', size='medium', color='black')
            std = np.std(abs_errors)
            plt.plot(error_0, error_0 - best_pred[name]['MAE'], linestyle='--',color='black')
            plt.plot(error_0, error_0 + best_pred[name]['MAE'], linestyle='--', color='black')
            plt.plot(error_0, error_0 - best_pred[name]['MAE'] - std*3, linestyle='--', linewidth = 2, color='green')
            plt.plot(error_0, error_0 + best_pred[name]['MAE'] + std*3, linestyle='--', linewidth = 2, color='green')

            outlier_thres_3 = best_pred[name]['MAE'] + std*3
            outliers_idxs_3 = np.where(abs_errors > outlier_thres_3)[0]

            title_prefix = best_pred[name]['adsorbate']+' - '+name
        else:
            title_prefix = best_pred[name]['adsorbate']+' - '+name
        plt.title(title_prefix+'\n RMSE:'+str(np.round(best_pred[name]['RMSE']*1000)/1000.)+' R2:'+str(np.round(r2*1000)/1000.), fontsize=16)
        plt.legend(loc='best', fontsize=14)
        min_lim -= np.abs(min_lim)*0.1
        max_lim += np.abs(max_lim)*0.1
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.xlabel(r'DFT E$_{ads}$ [eV]', fontsize=20)
        plt.ylabel(r'ML E$_{ads}$ [eV]', fontsize=20)
        plt.xticks(fontsize =16)
        plt.yticks(fontsize=16)
        plt.tight_layout()
            
        final_PWD = PWD_figures
        if PWD_figures is not None:
            if not os.path.isdir(final_PWD):
                os.makedirs(final_PWD)
                
            plt.show()
        else:
            plt.show()

        plt.clf()    

def get_grid(name, features, adsorbate, do_best = False):

    if do_best:
        takebest = ['ExtraTrees', 'GPR', 'KRR', 'XGboost']
    else:
        takebest = []

    if name in takebest:
        param_grids = { "KRR_H" : {'alpha' : [0.01],
                             'kernel' : ['laplacian'],
                             'gamma' : [1],
                             'degree': [2],
                             'coef0' : [0.1],
                            },
                       "KRR_OH" : {'alpha' : [0.001],
                             'kernel' : ['laplacian'],
                             'gamma' : [0.1],
                             'degree': [2],
                             'coef0' : [0.1],
                            },
                       "KRR_O" : {'alpha' : [0.1],
                             'kernel' : ['laplacian'],
                             'gamma' : [0.5],
                             'degree': [2],
                             'coef0' : [0.1],
                            },
                    "GPR_OH" : {'kernel': [sklearn.gaussian_process.kernels.ConstantKernel(1.0) * sklearn.gaussian_process.kernels.Matern(length_scale=1.0, nu=0.5)],
                             'alpha' : [0.01],
                             'n_restarts_optimizer' :[0],
                             'normalize_y' : [False],
                            },
                    "GPR_O" : {'kernel': [sklearn.gaussian_process.kernels.ConstantKernel(1.0) * sklearn.gaussian_process.kernels.Matern(length_scale=1.0, nu=1.5)],
                             'alpha' : [0.01],
                             'n_restarts_optimizer' :[10],
                             'normalize_y' : [False],
                            },
                    "GPR_H" : {'kernel': [sklearn.gaussian_process.kernels.ConstantKernel(1.0) * sklearn.gaussian_process.kernels.Matern(length_scale=1.0, nu=1.5)],
                             'alpha' : [0.01],
                             'n_restarts_optimizer' : [0],
                             'normalize_y' : [False],
                            },
                    "ExtraTrees_OH" : {'n_estimators': [1000],
                            'max_depth': [100],
                                           'min_samples_split' : [2],
                                           'min_samples_leaf' : [1],
                                           'max_features' : [None],
                                           'random_state' : [42],
                                }, 
                    "ExtraTrees_O" : {'n_estimators': [50],
                            'max_depth': [None],
                                           'min_samples_split' : [2],
                                           'min_samples_leaf' : [1],
                                           'max_features' : [None],
                                           'random_state' : [42],
                                },
                    "ExtraTrees_H" : {'n_estimators': [100],
                                           'max_depth': [100],
                                           'min_samples_split' : [2],
                                           'min_samples_leaf' : [1],
                                           'max_features' : [None],
                                           'random_state' : [42],
                                },
                        "XGboost_H" : {'eta':[ 0.1],
                                     'max_depth' : [50],
                                     'subsample' : [1],
                                     'gamma' : [0.005],
                                     'colsample_bytree' : [1],
                                     'colsample_bylevel' : [0.5],
                                     'colsample_bynode' : [1],
                                     'lambda' : [0.001],
                                     'min_child_weigth' : [3],
                                     'tree_method' : ['hist'],
                                    },
                        "XGboost_OH" : {'eta':[0.1],
                                     'gamma' : [0.005],
                                     'max_depth' : [10],
                                     'subsample' : [0.5],
                                     'colsample_bytree' : [0.75],
                                     'colsample_bylevel' : [1],
                                     'colsample_bynode' : [1],
                                     'lambda' : [0.1],
                                     'min_child_weight' : [3],
                                     'tree_method' : ['hist'],
                                    },
                        "XGboost_O" : {'eta':[0.1],
                                     'gamma' : [0],
                                     'max_depth' : [50],
                                     'subsample' : [1],
                                     'colsample_bytree' : [1],
                                     'colsample_bylevel' : [1],
                                     'colsample_bynode' : [0.5],
                                     'lambda' : [0.1],
                                     'min_child_weight' : [3],
                                     'tree_method' : ['hist'],
                                    },
                      }
    else:

        param_grids = { "KRR" : {'alpha' : np.logspace(0,4,5) / 10000.,
                                 'kernel' : ['linear', 'polynomial', 'laplacian', 'rbf', 'sigmoid', 'cosine'],
                                 'gamma' : [0.01, 0.05, 0.1, 0.5, 1, 10],
                                 'degree': [2, 3, 5],
                                 'coef0' : [0.1, 1, 5],
                                },
                        "GPR" : {'kernel': [sklearn.gaussian_process.kernels.ConstantKernel(1.0, constant_value_bounds="fixed") * sklearn.gaussian_process.kernels.RBF(1.0, length_scale_bounds="fixed"), sklearn.gaussian_process.kernels.ConstantKernel(1.0) * sklearn.gaussian_process.kernels.RBF(1.0), sklearn.gaussian_process.kernels.ConstantKernel(1.0) * sklearn.gaussian_process.kernels.Matern(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=0.5), sklearn.gaussian_process.kernels.ConstantKernel(1.0) * sklearn.gaussian_process.kernels.Matern(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=1.5)],
                                 'alpha' : np.logspace(0,10,6) / 10000000000.,
                                 'n_restarts_optimizer' : [0, 2,5,10],
                                 'normalize_y' : [False, True],
                                },
                        "Extra Trees" : {'n_estimators': [10, 50, 100, 1000],
                                           'max_depth': [None, 10, 100, 300, 500, 1000],
                                           'min_samples_split' : [2, 3, 4, 5, 10],
                                           'min_samples_leaf' : [1, 2, 5, 10],
                                           'max_features' : [None, 'sqrt', 'log2', len(features) * 0.25, len(features) * 0.5, len(features) * 0.75]
                                        },
                        "XGboost" : {'eta':[ 0.1, 0.3, 0.5, 0.75, 1],
                                     'max_depth' : [6, 10, 50, 100, 500],
                                     'subsample' : [0.25, 0.5, 0.75, 1],
                                     'colsample_bytree' : [0.5, 1],
                                     'colsample_bylevel' : [0.5,1],
                                     'colsample_bynode' : [0.5, 1],
                                     'lambda' :  np.logspace(0,4,5) / 10000.,
                                     'gamma' : [0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1],
                                     'min_child_weight': [1, 2, 3, 5, 10, 20],
                                    },
                      }

    if name in takebest:
        name2use = name+'_'+adsorbate
    else:
        name2use = name

    return param_grids[name2use]


def _GridSearchCV(X,y, features, names, regressors, train_sets, val_sets, init_idxs, test_idxs, adsorbate, PWD_figures=None, do_best = False):

    OS = sys.platform
    if OS == 'win32' or OS == 'cywin':
        folder_lim = '\\'
    else:
        folder_lim = '/'

    cv_idxs = np.sort(np.concatenate((train_sets[0], val_sets[0])))
    X_cv, y_cv = X[cv_idxs, :], y[cv_idxs]
    cv = []
    for i in range(len(train_sets)):
        train_set_tmp = np.intersect1d(cv_idxs,train_sets[i], return_indices=True)[1]
        val_set_tmp = np.intersect1d(cv_idxs,val_sets[i], return_indices=True)[1]

        assert np.concatenate((train_set_tmp, val_set_tmp)).shape[0] == cv_idxs.shape[0], "Total shape of train and validation is not equal to shape of cv_idxs"
        assert np.intersect1d(train_set_tmp, val_set_tmp).shape[0] == 0, "idxs repeat in train and validation"

        cv.append((np.copy(train_set_tmp), np.copy(val_set_tmp)))

    print('X shape:', X_cv.shape)
    print('y shape:', y_cv.shape)
    print('Features:', ' '.join(features))

    methods = [name for name in names]
    n_rand = int(np.random.rand(1)[0] * 100000)

    best_estimators_dict = {}
    METRICS = ['neg_mean_absolute_error', 'neg_root_mean_squared_error']
    i = 0
    best_models_dict = {}
    for name, reg in zip(names, regressors):
        i += 1
        param_grid = get_grid(name, features, adsorbate, do_best = do_best)
        clf = GridSearchCV(reg, param_grid, scoring = METRICS, cv = cv, refit = 'neg_root_mean_squared_error', n_jobs=-1 )
        clf.fit(X_cv,y_cv)
        best_estimators_dict[name] = {'best_estimator' : clf.best_estimator_,
                                      'best_score' : clf.best_score_,
                                      'best_params' : clf.best_params_,
                                      'all_cv_results': clf.cv_results_,
                                      'best_index': clf.best_index_,
                                      'scorer': clf.scorer_,
                                     }

        print()
        print()
        print()
        print(name)
        for key in best_estimators_dict[name].keys():
            if key != 'all_cv_results' and key != 'scorer':
                print(key)
                print(best_estimators_dict[name][key])
                print()

        scores = cross_val_score(sklearn.base.clone(best_estimators_dict[name]['best_estimator']), X_cv, y=y_cv, cv=cv, scoring ='neg_root_mean_squared_error')
        best_models_dict[name] = np.argsort(scores)[-1]

    all_preds = {}
    i = 0
    estimators = []
    final_names = []
    for name in names:
        i += 1
        print(best_estimators_dict[name]['best_estimator'])
        train_idxs = train_sets[best_models_dict[name]]
        val_idxs = val_sets[best_models_dict[name]]

        X_train = X[train_idxs, :]
        y_train = y[train_idxs]
        X_val = X[val_idxs,:]
        y_val = y[val_idxs]
        X_test = X[test_idxs,:]
        y_test = y[test_idxs]

        final_names.append(name)
        estimators.append(best_estimators_dict[name]['best_estimator'])

        new_clf = sklearn.base.clone(best_estimators_dict[name]['best_estimator'])
        new_clf.fit(X_train, y_train)
        test_predict = new_clf.predict(X_test)
        val_predict = new_clf.predict(X_val)
        train_predict = new_clf.predict(X_train)
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, test_predict))
        MAE = metrics.mean_absolute_error(y_test, test_predict)
        all_preds[name] = {'RMSE' : RMSE,
                           'MAE' : MAE,
                           'train' : train_predict,
                           'val' : val_predict,
                           'test' : test_predict,
                           'true_train' : y_train,
                           'true_val' : y_val,
                           'true_test' : y_test,
                           'adsorbate' : adsorbate,
                          }

    #plot_scatter(all_preds, PWD_figures = PWD_figures+folder_lim+'Best_Models')

    return final_names, estimators, best_models_dict

def main():

    OS = sys.platform
    if OS == 'win32' or OS == 'cywin':
        folder_lim = '\\'
    else:
        folder_lim = '/'

    dataset_file = sys.argv[1]
    PWD = sys.argv[2]
    sets_file = sys.argv[3]
    do_HypSearch = True if sys.argv[4] == 'True' else False


    if sets_file != 'None':
        print('Using available sets for training')
        sets_from_file = True

        with open(sets_file, 'rb') as F:
            sets_dict =  pickle.load(F)
    else:
        sets_from_file = False


    DATE = '_'.join(datetime.today().strftime('%Y-%m-%d').split())

    print('Working with '+dataset_file.split(folder_lim)[-1]+'...')
    print()

    df = pd.read_excel(dataset_file)

    print('Sample of dataset..')
    print(df)
    print()

    if PWD == '.':
        PWD_models = 'Models'+folder_lim+DATE
        PWD_figures = 'Figures'+folder_lim+DATE
    else:
        PWD_models = PWD+folder_lim+'Models'+folder_lim+DATE
        PWD_figures = PWD+folder_lim+'Figures'+folder_lim+DATE

    adsorbate = dataset_file.split(folder_lim)[-1].split('_')[1]

    PWD_figures += folder_lim+adsorbate
    PWD_models +=folder_lim+adsorbate

    not_valid = ['Eads', 'Binding site', 'adsorbate', 'Material', 'label']

    tags = [data_tag for data_tag in list(df) if data_tag not in not_valid]

    features = tags

    print('Features:')
    print(features)
    print()


    X = df[features].values
    Y = df['Eads'].values
    first_idxs = np.arange(Y.shape[0])

    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    print('Min - Max Eads:', np.amin(Y), np.amax(Y))
    print()

    mm_X, _ = get_scaled_features(X[:,:])
    scaled_X = np.copy(mm_X)


    names = [#'GPR',
             'ExtraTrees',
             'XGboost',
             'KRR',
            ]
     
    regressors = [#GaussianProcessRegressor(),
                  ExtraTreesRegressor(),
                  xgboost.XGBRegressor(objective="reg:squarederror"), 
                  KernelRidge(), 
                 ]

    idxs4model = {}
    dataset_idxs = first_idxs
    Materials = df['Material'].values[dataset_idxs]
    Sites = df['Binding site'].values[dataset_idxs]

    idxs4model['init_idxs'] = get_initial_samples(Materials, Sites)
    
    training_size = 0.85 if not sets_from_file else float(sets_file.split(folder_lim)[-1].split('.pickle')[0].split('-')[1])
    
    print('Train size:', int(np.round(Y.shape[0] * training_size)), '# Forced traning samples:', idxs4model['init_idxs'].shape[0])
    print()
    
    if idxs4model['init_idxs'].shape[0] >= np.round(Y.shape[0] * training_size):
        print('Required training set size is equal or smaller than the number of forced training entries. Exiting...')
        exit()
    
    
    print('Forced training samples that have been added:', idxs4model['init_idxs'].shape[0])
    idxs4model['remain_idxs'] = np.delete(np.arange(Y.shape[0]), idxs4model['init_idxs'])
    
    rng = np.random.default_rng()
    if sets_from_file:
        idxs4model['test_idxs'] = sets_dict['test_idxs']
        test_size = idxs4model['test_idxs'].shape[0]
    else:
        test_size = Y.shape[0] - int(np.round(Y.shape[0] *  training_size))
        idxs4model['test_idxs'] = idxs4model['remain_idxs'][rng.choice(idxs4model['remain_idxs'].shape[0], size = test_size, replace=False)]
    
    
    print()
    print()


    if sets_from_file:
        idxs4model['train_sets'] = sets_dict['train_sets']
        idxs4model['val_sets'] = sets_dict['val_sets']
    else:
        idxs4model['train_sets'], idxs4model['val_sets'] = _in_house_cv_sets(training_size,idxs4model['init_idxs'], Y.shape[0], idxs4model['test_idxs'])

    n_models = len(idxs4model['train_sets'])

    Model_outputs = {}
    scores = {}
    train_test_sets = {}
    
    do_best = False if do_HypSearch else True
    
    Model_outputs['final_names'], Model_outputs['opt_regressors'], Model_outputs['best_model_idxs'] = _GridSearchCV(scaled_X, Y, features, names, regressors,idxs4model['train_sets'], idxs4model['val_sets'], idxs4model['init_idxs'], idxs4model['test_idxs'], adsorbate, PWD_figures=PWD_figures+folder_lim+str(training_size)+folder_lim, do_best = do_best)
    
    if not sets_from_file:
        sets_dict =  {'train_sets' : idxs4model['train_sets'],
                     'val_sets' : idxs4model['val_sets'],
                     'test_idxs' : idxs4model['test_idxs'],
                     'best_model_idxs' : Model_outputs['best_model_idxs'],
                    }

        with open(adsorbate+'_sets-'+str(training_size)+'.pickle', 'wb') as F:
            pickle.dump(sets_dict, F)
    
    train_sizes = [training_size, 1]
    
    size2Analyze = training_size
    
    
    for name in Model_outputs['final_names']:
        scores[name] = {}
        for size in train_sizes:
            scores[name][size] = {'MAE' :[], 'RMSE': [], 'R2' :[], 'test' : [], 'true_test' : [], 'train' : [], 'true_train' : [], 'val' : [], 'true_val': [], 'X_train' : [], 'X_test' : [], 'X_val' :[],  'Model' : [], 'train_idxs' :[], 'test_idxs': [], 'val_idxs' :[]}
    
    for size in train_sizes:
    
        if size == 1:
            for name, reg in zip(Model_outputs['final_names'], Model_outputs['opt_regressors']):
                est = sklearn.clone(reg)
                est.fit(scaled_X, Y)
                train_predictions = est.predict(scaled_X)
                scores[name][size]['train'].append(train_predictions)
                scores[name][size]['true_train'].append(Y)
                scores[name][size]['X_train'].append(scaled_X)
                scores[name][size]['Model'].append(est)
                scores[name][size]['train_idxs'].append(np.arange(Y.shape[0]))
        else:
            train_sets = sets_dict['train_sets'] 
            val_sets = sets_dict['val_sets']
            n_models = len(train_sets)
            print('Training:', size)
            train_test_sets[size] = {} 
            for i in range(n_models):
                train_idxs = train_sets[i]
                val_idxs = val_sets[i]
                test_idxs = sets_dict['test_idxs']
      
                train_test_sets[size][i] = {'train' : train_idxs, 'val' : val_idxs, 'test' : test_idxs}
         
                assert np.intersect1d(train_idxs,val_idxs).shape[0] == 0, "idxs repeat in train and validation"
         
                X_train = scaled_X[train_idxs, :]
                y_train = Y[train_idxs]
                X_val = scaled_X[val_idxs, : ]
                y_val = Y[val_idxs]
                X_test = scaled_X[test_idxs, :]
                y_test = Y[test_idxs]
         
                print('train X:', X_train.shape)
                print('train Y:', y_train.shape)
                print('val X:', X_val.shape)
                print('val Y:', y_val.shape)
                print('test X:', X_test.shape)
                print('test y:', y_test.shape)
                print()
                
                
                for name, reg in zip(Model_outputs['final_names'], Model_outputs['opt_regressors']):
                    est = sklearn.clone(reg)
                    est.fit(X_train, y_train)
                    val_predictions= est.predict(X_val)
                    y_pred = est.predict(X_test)
                    train_predictions = est.predict(X_train)
                    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                    MAE = metrics.mean_absolute_error(y_test, y_pred)
                    R2 = metrics.r2_score(y_test, y_pred)
                    scores[name][size]['MAE'].append(MAE)
                    scores[name][size]['RMSE'].append(RMSE)
                    scores[name][size]['R2'].append(R2)
                    scores[name][size]['test'].append(y_pred)
                    scores[name][size]['true_test'].append(y_test)
                    scores[name][size]['train'].append(train_predictions)
                    scores[name][size]['true_train'].append(y_train)
                    scores[name][size]['true_val'].append(y_val)
                    scores[name][size]['val'].append(val_predictions)
                    scores[name][size]['X_train'].append(X_train)
                    scores[name][size]['X_test'].append(X_test)
                    scores[name][size]['X_val'].append(X_val)
                    scores[name][size]['Model'].append(est)
                    scores[name][size]['train_idxs'].append(train_idxs)
                    scores[name][size]['test_idxs'].append(test_idxs)
                    scores[name][size]['val_idxs'].append(val_idxs)

    best_pred = {}

    if 1 in train_sizes:
        save_models = True
        if not os.path.isdir(PWD_models):
            os.makedirs(PWD_models)
    else:
        save_models = False
    for name in Model_outputs['final_names']:
        if save_models:
            model_path = PWD_models+folder_lim+name
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
        print()
        print('== '+name+' performance ==')
        print("\t%5s -- \t%5s (%5s) (%5s) \t%5s (%5s) (%5s)\t%5s (%5s) (%5s)\t %5s"%("Train", "MAE", "Std", "Best",  "RMSE","Std", "Best", "R2", "Std", "Best", "Best Task"))
        for size in train_sizes:
            scores_dict = scores[name][size]
            if size != 1:
                best_task = np.argmin(scores_dict['RMSE'])
                if save_models:
                    model_final_path = model_path+folder_lim+str(size)
                    if not os.path.isdir(model_final_path):
                        os.makedirs(model_final_path)
                    for i in range(len(scores_dict['RMSE'])):
                        model_name = name+'_'+adsorbate+'_'+str(size)+'_Task'+str(i+1)
                        if i == best_task:
                            model_name += '_BEST'
                        model_name +='.pickle'
                        model = {'RMSE' : scores_dict['RMSE'][i],
                                          'MAE' : scores_dict['MAE'][i],
                                          'true_train': scores_dict['true_train'][i],
                                          'train' : scores_dict['train'][i],
                                          'true_test' : scores_dict['true_test'][i],
                                          'test' : scores_dict['test'][i],
                                          'true_val' : scores_dict['true_val'][i],
                                          'val' : scores_dict['val'][i],
                                          'adsorbate' : adsorbate,
                                          'X_train' : scores_dict['X_train'][i],
                                          'X_test' : scores_dict['X_test'][i],
                                          'X_val' : scores_dict['X_val'][i],
                                          'Model' : scores_dict['Model'][i],
                                          'train_idxs' : scores_dict['train_idxs'][i],
                                          'test_idxs' : scores_dict['test_idxs'][i],
                                          'val_idxs' : scores_dict['val_idxs'][i],
                                         }
                        with open(model_final_path+folder_lim+model_name, 'wb') as F:
                            pickle.dump(model, F)
    
                if size == size2Analyze:
                    best_pred[name] = {'RMSE' : scores_dict['RMSE'][best_task],
                                          'MAE' : scores_dict['MAE'][best_task],
                                          'true_train': scores_dict['true_train'][best_task],
                                          'train' : scores_dict['train'][best_task],
                                          'true_test' : scores_dict['true_test'][best_task],
                                          'test' : scores_dict['test'][best_task],
                                          'true_val' : scores_dict['true_val'][best_task],
                                          'val' : scores_dict['val'][best_task],
                                          'adsorbate' : adsorbate,
                                          'X_train' : scores_dict['X_train'][best_task],
                                          'X_test' : scores_dict['X_test'][best_task],
                                          'X_val' : scores_dict['X_val'][best_task],
                                         }
                n_train = int(np.round(Y.shape[0] * size))
                print("\t%i    -- \t%2.3f (%2.3f) (%2.3f) \t%2.3f (%2.3f) (%2.3f)\t%2.3f (%2.3f) (%2.3f)\t %i" % (n_train, np.mean(scores_dict['MAE']), np.std(scores_dict['MAE']),scores_dict['MAE'][best_task], np.mean(scores_dict['RMSE']), np.std(scores_dict['RMSE']), scores_dict['RMSE'][best_task], np.mean(scores_dict['R2']), np.std(scores_dict['R2']), scores_dict['R2'][best_task], best_task+1))
            else:
                model_final_path = model_path+folder_lim+str(size)
                if not os.path.isdir(model_final_path):
                    os.makedirs(model_final_path)
                model_name = name+'_'+adsorbate+'_'+str(size)+'.pickle'
                model = {'true_train': scores_dict['true_train'][0],
                                  'train' : scores_dict['train'][0],
                                  'adsorbate' : adsorbate,
                                  'X_train' : scores_dict['X_train'][0],
                                  'Model' : scores_dict['Model'][0],
                                  'train_idxs' : scores_dict['train_idxs'][0],
                                 }
                with open(model_final_path+folder_lim+model_name, 'wb') as F:
                    pickle.dump(model, F)
    
        print()

    plot_scatter(best_pred, PWD_figures=PWD_figures+folder_lim+str(training_size), for_outliers =  True)
        

if __name__ == '__main__':
    main()
