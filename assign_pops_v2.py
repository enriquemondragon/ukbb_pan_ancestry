#source('https://github.com/macarthur-lab/gnomad_hail/blob/master/utils/generic.py')

'''
Forked on 11/07/2025 from https://github.com/atgu/ukbb_pan_ancestry by enriquemondragon
'''

import pandas as pd
import numpy as np
from collections import defaultdict, namedtuple, OrderedDict
from sklearn.ensemble import RandomForestClassifier
from typing import *
import random
# Additional
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree
import argparse


def assign_population_pcs(
        pop_pc_pd: pd.DataFrame,
        path_out: str,
        num_pcs: int,
        pcs_col: str = 'scores',
        known_col: str = 'super_pop', # known_pop
        fit: RandomForestClassifier = None,
        seed: int = 42,
        prop_train: float = 0.8,
        n_estimators: int = 100,
        min_prob: float = 0.9,
        output_col: str = 'pop',
        missing_label: str = 'oth'
) -> Tuple[pd.DataFrame, RandomForestClassifier]:
    """
    This function uses a random forest model to assign population labels based on the results of PCA.
    Default values for model and assignment parameters are those used in gnomAD.
    :param Table pop_pc_pd: Pandas dataframe containing population PCs as well as a column with population labels
    :param str path_out: Output path for saving tree structure
    :param str known_col: Column storing the known population labels
    :param str pcs_col: Columns storing the PCs
    :param RandomForestClassifier fit: fit from a previously trained random forest model (i.e., the output from a previous RandomForestClassifier() call)
    :param int num_pcs: number of population PCs on which to train the model
    :param int seed: Random seed
    :param float prop_train: Proportion of known data used for training
    :param int n_estimators: Number of trees to use in the RF model
    :param float min_prob: Minimum probability of belonging to a given population for the population to be set (otherwise set to `None`)
    :param str output_col: Output column storing the assigned population
    :param str missing_label: Label for samples for which the assignment probability is smaller than `min_prob`
    :return: Dataframe containing sample IDs and imputed population labels, trained random forest model
    :rtype: DataFrame, RandomForestClassifier
    """

    # Expand PC column
    #pop_pc_pd = expand_pd_array_col(pop_pc_pd, pcs_col, num_pcs, 'PC')
    pc_cols = ['PC{}'.format(i + 1) for i in range(num_pcs)]
    print(pc_cols)
    #pop_pc_pd[pc_cols] = pd.DataFrame(pop_pc_pd[pcs_col].values.tolist())[list(range(num_pcs))]
    train_data = pop_pc_pd.loc[~pop_pc_pd[known_col].isnull()]

    N = len(train_data)
    print(train_data.shape)

    # Split training data into subsamples for fitting and evaluating
    if not fit:
        random.seed(seed)
        train_subsample_ridx = random.sample(list(range(0, N)), int(N * prop_train))
        train_fit = train_data.iloc[train_subsample_ridx]
        fit_samples = [x for x in train_fit['sample']]
        evaluate_fit = train_data.loc[~train_data['sample'].isin(fit_samples)]
        # Train RF
        # .as_matrix() # 11/07/25 - deprecated
        # training_set_known_labels = train_fit[known_col]
        # training_set_pcs = train_fit[pc_cols].as_matrix()
        # evaluation_set_pcs = evaluate_fit[pc_cols].as_matrix()
        print(train_fit)
        training_set_known_labels = train_fit[[known_col]].to_numpy()
        training_set_pcs = train_fit[pc_cols].to_numpy()
        evaluation_set_pcs = evaluate_fit[pc_cols].to_numpy()

        print('\n----------------------------------------------------')
        print('Fitting random forest...')
        pop_clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
        pop_clf.fit(training_set_pcs, training_set_known_labels)
        print('Random forest feature importances are as follows: {}'.format(pop_clf.feature_importances_))

        # Evaluate RF
        predictions = pop_clf.predict(evaluation_set_pcs)
        error_rate = 1 - sum(evaluate_fit[known_col] == predictions) / float(len(predictions))
        accuracy = accuracy_score(evaluate_fit[known_col], predictions)
        print("Accuracy:", accuracy)

        print('Estimated error rate for RF model is {}'.format(error_rate))
    else:
        pop_clf = fit

    # Classify data
    print('Classifying data')
    # .as_matrix() # 11/07/25 - deprecated
    # pop_pc_pd[output_col] = pop_clf.predict(pop_pc_pd[pc_cols].as_matrix())
    # probs = pop_clf.predict_proba(pop_pc_pd[pc_cols].as_matrix())
    pop_pc_pd[output_col] = pop_clf.predict(pop_pc_pd[pc_cols].to_numpy())
    probs = pop_clf.predict_proba(pop_pc_pd[pc_cols].to_numpy())
    probs = pd.DataFrame(probs, columns=[f'prob_{p}' for p in pop_clf.classes_])
    print('probs shape ' + str(probs.shape))
    print('pop_pc_pd shape ' + str(pop_pc_pd.shape))
    #print(probs.iloc[:3,])
    #print(pop_pc_pd.iloc[:3,])
    ##pop_pc_pd = pd.concat([pop_pc_pd, probs], axis=1, ignore_index=True)
    print(pop_pc_pd.shape)
    probs['max'] = probs.max(axis=1)
    pop_pc_pd.loc[probs['max'] < min_prob, output_col] = missing_label
    #pop_pc_pd = pop_pc_pd.drop(pc_cols, axis='columns')
    print(pop_pc_pd.shape)

    # Visualize tree
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

    tree.plot_tree(pop_clf.estimators_[0], 
           feature_names=pc_cols, 
           class_names=pop_pc_pd[known_col].unique(), 
           filled=True,        
           rounded=True) 

    print('Saving tree figure...')
    fig_path = path_out + '/pop_clf_tree.pdf'
    fig.savefig(fig_path)

    return pop_pc_pd, pop_clf


def main():

    parser = argparse.ArgumentParser(description='\n============ Estimate ancestry using ukbb-pan-ancestry approach ============\n', usage='%(prog)s  [--input] [--ref_data] [--pop_ref_data] [--out_dir]')
    parser.add_argument('-i', '--input', type=str, required=True, help='Target input data', dest='TARGET')
    parser.add_argument('-r', '--ref_data', type=str, required=True, help='Reference data', dest='REF_1KGP')
    parser.add_argument('-pop', '--pop_ref_data', type=str, required=True, help='targets file', dest='REF_INFO')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='Output directory', dest='OUT')
    parser.add_argument('-pc_n', '--pc_number', type=int, required=True, help='Number of PCs used', default=10, dest='PC_N')
    parser.add_argument('-p', '--probability', type=float, required=True, help='Minimum probability for random forest',default=0.5, dest='MIN_PROB')
    
    args = parser.parse_args()
    
    REF_1KGP = args.REF_1KGP
    REF_INFO = args.REF_INFO
    TARGET = args.TARGET
    OUT = args.OUT
    PC_N = args.PC_N
    MIN_PROB = args.MIN_PROB

    ref = pd.read_table(REF_1KGP, header=0, sep='\t')
    ref_info = pd.read_table(REF_INFO, header=0, sep='\t')
    ref_info = ref_info[['sample', 'super_pop']]
    target = pd.read_table(TARGET, header=0, sep='\t')

    ref_merge = pd.merge(left=ref, right=ref_info, left_on='IID', right_on='sample', how='inner')

    target_ref = pd.concat([ref_merge, target])

    # Rename PLINK2 PC names 
    # NB: Assumes PLINK2 input
    target_ref = target_ref.rename(columns={f'PC{i}_AVG': f'PC{i}' for i in range(1, PC_N+1)})

    print(target_ref)
    print(target_ref['super_pop'].value_counts())

    # Ancestry inference
    pcs_df, clf = assign_population_pcs(pop_pc_pd=target_ref, num_pcs=PC_N, min_prob=MIN_PROB, path_out = OUT)
    
    print('\n----------------------------------------------------')
    print('Estimating ancestry...')
    
    target_pops = pcs_df.loc[pcs_df['super_pop'].isnull()]
    print(target_pops['pop'].value_counts())
    cols = ['IID', 'pop'] + [f'PC{i}' for i in range(1, PC_N+1)]
    print("cols:",target_pops)
    target_pops_df = target_pops[cols]
    print(target_pops_df)
    
    out_path = OUT + '/results.tsv.gz'
    target_pops_df.to_csv(out_path, sep='\t', index=False, compression='gzip')
    print('saved!')

if __name__ == "__main__":
    main()