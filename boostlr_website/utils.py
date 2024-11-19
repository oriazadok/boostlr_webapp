from sklearn.model_selection import train_test_split
from boostlr_website.constants import *

from sklearn.ranking.utils import *
from sklearn.ranking.BoostingLRWrapper import BoostingLRWrapper

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_boostlr(dataset_path, dist_algo, dist_score):

    # Extract the base name (without extension)
    base_name = os.path.basename(dataset_path).replace(".xarff", "")

    # Create the full paths and the desired string
    train_base_name = f"tmp/{base_name}_train"
    test_base_name = f"tmp/{base_name}_test"

    predictions_base_name = f"predictions/{base_name}_predictions.csv"

    train_dataset = os.path.join(ROOT_DIR, f"{train_base_name}.xarff")
    test_dataset = os.path.join(ROOT_DIR, f"{test_base_name}.xarff")

    predictions_path = os.path.join(ROOT_DIR, predictions_base_name)

    # Load the dataset and get attribute information
    df, attribute_info = load_xarff(dataset_path)


    # Split the data into training and test sets using Pandas
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Save the split datasets to XARFF files with the original attribute info
    save_to_xarff(train_data, train_dataset, relation_name=train_base_name, attribute_info=attribute_info)
    save_to_xarff(test_data, test_dataset, relation_name=test_base_name, attribute_info=attribute_info)

    print("Training and test datasets saved to XARFF files.")

    # Load dataset as Instances
    train_data_Instances = load_dataset_as_Instances(train_dataset)
    test_data_Instances = load_dataset_as_Instances(test_dataset)

    # Initialize model
    model = BoostingLRWrapper(max_iterations=50, seed=7, dist_algo=dist_algo, dist_score=dist_score)

    # Train and score the model
    model.fit(train_data_Instances)

    predictions = model.predict(test_data_Instances)

    labels = get_labels(attribute_info)

    create_preds_test_file(predictions_path, test_data, predictions, labels)

    score = model.score(test_data_Instances)

    return score, predictions_base_name

def run_boostlr_with_two_datasets(train_path, test_path, dist_algo, dist_score):

    base_name = os.path.basename(test_path).replace(".xarff", "")
    predictions_base_name = f"predictions/{base_name}_predictions.csv"
    predictions_path = os.path.join(ROOT_DIR, predictions_base_name)

    # Load datasets
    train_instances = load_dataset_as_Instances(train_path)
    test_instances = load_dataset_as_Instances(test_path)

    # Initialize model
    model = BoostingLRWrapper(max_iterations=50, seed=7, dist_algo=dist_algo, dist_score=dist_score)

    # Train the model on the training dataset
    model.fit(train_instances)
    print("Model trained on training dataset")

    predictions = model.predict(test_instances)
    test_data_df, attribute_info = load_xarff(test_path)
    labels = get_labels(attribute_info)

    create_preds_test_file(predictions_path, test_data_df, predictions, labels)

    # Score the model using the test dataset
    score = model.score(test_instances)
    print("Model scored on test dataset")

    return score, predictions_base_name

def get_labels(attribute_info):
    rankings_str = attribute_info['L']
    labels_str = re.search(r"\{(.+?)\}", rankings_str).group(1)

    # Split the string by commas to get the list of labels
    labels = labels_str.split(',')

    return labels

def create_preds_test_file(predictions_path, test_data, predictions, labels):

    # List to store the formatted predicted rankings
    predicted_rankings = []
    
    # Iterate over each row of predictions
    for prediction in predictions:
        # Get the sorted indices in ascending order (lowest value first)
        sorted_indices = sorted(range(len(prediction)), key=lambda x: prediction[x])

        # Map the sorted indices to the corresponding labels
        ranked_labels = [labels[idx] for idx in sorted_indices]

        # Create the ranking string (e.g., "g > i > a > d > h > c > f > j > e > b")
        ranking_str = ">".join(ranked_labels)

        # Append the ranking string to the list
        predicted_rankings.append(ranking_str)

    # Add the predicted rankings as a new column in the test DataFrame
    test_data['Predicted_Ranking'] = predicted_rankings

    # Save the test data with predictions as a CSV file
    test_data.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

def creates_dirs():
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    
    if not os.path.exists(PREDICTIONS_FOLDER):
        os.makedirs(PREDICTIONS_FOLDER)