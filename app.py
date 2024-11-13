import os
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from src.BoostingLRWrapper import BoostingLRWrapper
from src.utils import *

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = os.path.join(BASE_DIR, 'datasets')
ALLOWED_EXTENSIONS = {'xarff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/algo', methods=['GET', 'POST'])
def algo():
    # Get the list of available datasets
    datasets = [f for f in os.listdir(DATASETS_FOLDER) if allowed_file(f)]

    result = None  # Initialize result variable

    if request.method == 'POST':
        # Get the dataset choice from the dropdown
        dataset_choice = request.form.get('dataset_choice')

        # Get the uploaded files (single or double upload)
        uploaded_file = request.files.get('uploaded_file')
        uploaded_file1 = request.files.get('uploaded_file1')
        uploaded_file2 = request.files.get('uploaded_file2')

        # Get the metric selections
        dist_algo_choice = request.form.get('dist_algo')
        dist_score_choice = request.form.get('dist_score')

        # Determine which dataset(s) to use
        if dataset_choice and dataset_choice != '':
            dataset_path = os.path.join(DATASETS_FOLDER, dataset_choice)
            print("Selected dataset from list:", dataset_path)

        elif uploaded_file and allowed_file(uploaded_file.filename):
            filename = uploaded_file.filename
            dataset_path = os.path.join(DATASETS_FOLDER, filename)
            uploaded_file.save(dataset_path)
            print("Uploaded single dataset:", dataset_path)

        elif uploaded_file1 and allowed_file(uploaded_file1.filename) and uploaded_file2 and allowed_file(uploaded_file2.filename):
            filename1 = uploaded_file1.filename
            filename2 = uploaded_file2.filename
            dataset_path1 = os.path.join(DATASETS_FOLDER, filename1)
            dataset_path2 = os.path.join(DATASETS_FOLDER, filename2)
            uploaded_file1.save(dataset_path1)
            uploaded_file2.save(dataset_path2)
            print("Uploaded two datasets:", dataset_path1, dataset_path2)
        else:
            return render_template('algo.html', datasets=datasets, error="Please select a dataset or upload a file.")

        # Map user choices to functions
        dist_algo = kendalls_tau if dist_algo_choice == 'kendalltau' else ndcg
        dist_score = kendalls_tau if dist_score_choice == 'kendalltau' else ndcg

        # Run the BoostLR algorithm
        if dataset_choice or uploaded_file:
            result = run_boostlr(dataset_path, dist_algo, dist_score)
        elif uploaded_file1 and uploaded_file2:
            result = run_boostlr_with_two_datasets(dataset_path1, dataset_path2, dist_algo, dist_score)

    # Render input.html with the score result if available
    return render_template('algo.html', datasets=datasets, score=result)


def run_boostlr(dataset_path, dist_algo, dist_score):

    # Extract the base name (without extension) and directory
    base_name = os.path.basename(dataset_path).replace(".xarff", "")
    root_directory = os.path.dirname(os.path.dirname(dataset_path))

    # Create the full paths and the desired string
    train_base_name = f"tmp/{base_name}_train"
    test_base_name = f"tmp/{base_name}_test"

    predictions_base_name = f"predictions/{base_name}_predictions.csv"

    train_dataset = os.path.join(root_directory, f"{train_base_name}.xarff")
    test_dataset = os.path.join(root_directory, f"{test_base_name}.xarff")

    predictions_path = os.path.join(root_directory, predictions_base_name)

    # Load the dataset and get attribute information
    df, attribute_info = load_xarff(dataset_path)


    # Split the data into training and test sets using Pandas
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Save the split datasets to XARFF files with the original attribute info
    save_to_xarff(train_data, train_dataset, relation_name=train_base_name, attribute_info=attribute_info)
    save_to_xarff(test_data, test_dataset, relation_name=test_base_name, attribute_info=attribute_info)

    print("Training and test datasets saved to XARFF files.")

    print("train_dataset: ", train_dataset)
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

    return score

def run_boostlr_with_two_datasets(train_path, test_path, dist_algo, dist_score):
    # Load datasets
    train_instances = load_dataset_as_Instances(train_path)
    test_instances = load_dataset_as_Instances(test_path)

    # Initialize model
    model = BoostingLRWrapper(max_iterations=50, seed=7, dist_algo=dist_algo, dist_score=dist_score)

    # Train the model on the training dataset
    model.fit(train_instances)
    print("Model trained on training dataset")

    # Score the model using the test dataset
    score = model.score(test_instances)
    print("Model scored on test dataset")

    return score


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
    tmp = os.path.join("tmp")
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    
    predictions = os.path.join("predictions")
    if not os.path.exists(predictions):
        os.makedirs(predictions)


if __name__ == '__main__':
    creates_dirs()
    start_jvm()  # Start JVM when the Flask app starts
    try:
        app.run(debug=True)
    finally:
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()  # Shutdown JVM when the app exits
