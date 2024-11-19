import os
import numpy as np
from sklearn.model_selection import KFold

from boostlr_website.src.BoostingLRWrapper import BoostingLRWrapper
from boostlr_website.src.utils import *

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":

    start_jvm()

    dataset = "boostlr_website/datasets/sushi_mini.xarff"

    # Extract the base name (without extension) and directory
    base_name = os.path.basename(dataset).replace(".xarff", "")
    # directory = os.path.dirname(dataset)

    # Create the full paths and the desired string
    train_base_name = f"{base_name}_train"
    test_base_name = f"{base_name}_test"

    # train_dataset = os.path.join(directory, f"{train_base_name}.xarff")
    # test_dataset = os.path.join(directory, f"{test_base_name}.xarff")

    folds = os.path.join("boostlr_website/folds")
    if not os.path.exists(folds):
        os.makedirs(folds)

    results = os.path.join("boostlr_website/results")
    if not os.path.exists(results):
        os.makedirs(results)


    # Load the dataset and get attribute information
    df, attribute_info = load_xarff(dataset)


    print(f"Running dataset {base_name}")

    function_sets = [
        (kendalls_tau, kendalls_tau),
        (kendalls_tau, ndcg),
        (ndcg, kendalls_tau),
        (ndcg, ndcg),
    ]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    with open(f"boostlr_website/results/{base_name}_compare.txt", "a") as file:
        file.write(f"Exp     algo             score            mean      std     \n")
    
    for func_index, (dist_algo, dist_score) in enumerate(function_sets):

        num_of_iters = 50
        curr_seed = 7

        print(f"\nRunning with function set {func_index + 1}: (dist_algo={dist_algo.__name__}, dist_score={dist_score.__name__})")

        total_kt_scores = []
        # Loop over the folds
        for fold, (train_index, test_index) in enumerate(kf.split(df)):
            print(f"Processing fold {fold + 1}...")

            # Split data into training and test sets for this fold
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]

            # Save the split datasets into XARFF files
            train_dataset_file = f"{folds}/train_fold_{fold + 1}.xarff"
            test_dataset_file = f"{folds}/test_fold_{fold + 1}.xarff"

            save_to_xarff(train_data, train_dataset_file, relation_name=f"train_fold_{fold + 1}", attribute_info=attribute_info)
            save_to_xarff(test_data, test_dataset_file, relation_name=f"test_fold_{fold + 1}", attribute_info=attribute_info)

            # Load the dataset as Instances for Weka
            train_instances = load_dataset_as_Instances(train_dataset_file)
        #     test_instances = load_dataset_as_Instances(test_dataset_file)

        #     # Initialize and train the BoostingLR model
        #     model = BoostingLRWrapper(max_iterations=num_of_iters, seed=curr_seed, dist_algo=dist_algo, dist_score=dist_score)
        #     model.fit(train_instances)

        #     curr_seed += 1

        #     avg_kt = model.score(test_instances)
        #     total_kt_scores.append(avg_kt)
        #     print(f"Fold {fold + 1} {dist_score.__name__}: {avg_kt * 100:.2f}%\n")

        # # Calculate the overall average Kendall's Tau across all folds
        # mean_kt = np.mean(total_kt_scores)
        # std_kt = np.std(total_kt_scores)
        # print(f"{mean_kt * 100:.2f}% (mean)")
        # print(f"{std_kt * 100:.2f}% (std)")

        # with open(f"results/{base_name}_compare.txt", "a") as file:

        #     score_space = "     " if dist_algo.__name__ == "kendalls_tau" else "             "
        #     mean_space = "     " if dist_score.__name__ == "kendalls_tau" else "             "
        #     exp_space = "       "
        #     std_space = " " * (10 - len(f"{mean_kt * 100:.2f}%"))
            
        #     file.write(f"{func_index + 1}{exp_space}{dist_algo.__name__}{score_space}{dist_score.__name__}{mean_space}{mean_kt * 100:.2f}%{std_space}{std_kt * 100:.2f}%\n")

    stop_jvm()