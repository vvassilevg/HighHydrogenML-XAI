from Counterfactuals_wkr import Create_Count_Candidates, Make_Counterfactuals
import sys


def main():

    model_file = sys.argv[1]
    dataset_file = sys.argv[2]
    API_key_file = sys.argv[3]

    df_original, df_count, df_candidates =  Make_Counterfactuals.main(model_file, dataset_file, API_key_file)


    Create_Count_Candidates.main(df_candidates, model_file, dataset_file, API_key_file, df_count, df_original)


if __name__ == '__main__':
    main()

