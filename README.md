# HighHydrogenML-XAI
Python routines developed within the Marie Skłodowska-Curie Actions (MSCA) Postdoctoral Fellowship HighHydrogenML (grant agreement 101105610) for the developed explainable artificial intelligence materials discovery and design strategy. Preprint available at https://chemrxiv.org/engage/chemrxiv/article-details/6876617523be8e43d6556edc

Before using **XAI_Discovery.py**, the user needs to change the **counterfactual_explanations.py** file from the original dice_ml package with the one provided in this repository. For locating the path to the dice_ml package one can run the command _pip show dice_ml_.


**ML_Eads.py**

Contains the routines for training ML models for the prediction of adsorption energies. Four different models can be trained: Extremely randomized trees, Kernel Ridge Regression, XGBoost, and Gaussian Process Regression. Also, parity plots can be obtained from the code.

Arguments (in order):

_dataset_file_ : The file (xlsx format) containing the dataset (for instance, see Examples/H_adsorption/Dataset_H_All.xlsx)

_PWD_ : Path to the folder where the files/figures created within the code will be saved.
 
_sets_file_ : The file (pickle format) containing the training, validation, and test splits, if available (for instance, see Examples/H_adsorption/H_sets-0.85.pickle). In case no such file is available, write None.

_do_HypSearch_ : Whether hyperparameter optimization will be carried out or not. Options: True, False

**Analyze_MLModels.py**

Contains the routines to perform the feature analysis by SHAP aanalysis. It outputs beeswarm plots and source data for doing partial dependence plots and scatter plots with combined SHAP values.

Arguments (in order):

_model_file_ : The file (pickle format) containing the model and all relevant information of it. These models are those obtained from the **ML_Eads.py** routine (for instance, see Examples/H_adsorption/ExtraTrees_H_1.pickle)

_dataset_file_ : The file (xlsx format) containing the dataset (for instance, see Examples/H_adsorption/Dataset_H_All.xlsx)

**XAI_Discovery.py**

Contains the routines for counterfactual generation and candidates retrieval. It outputs three xlsx files containing, original samples, counterfactuals, and potential candidates for DFT validation.

Arguments (in order):

_model_file_ : The file (pickle format) containing the model and all relevant information of it. These models are those obtained from the **ML_Eads.py** routine (for instance, see Examples/H_adsorption/ExtraTrees_H_1.pickle)

_dataset_file_ : The file (xlsx format) containing the dataset (for instance, see Examples/H_adsorption/Dataset_H_All.xlsx)

_MaterialsProject_APIkey_ : txt file containing the API key to use Materials project for the retrieval of candidates. Such key is personal for each person's account (see the Materials Project web page, https://next-gen.materialsproject.org/api , for more information)
