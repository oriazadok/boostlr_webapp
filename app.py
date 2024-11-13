import os
from flask import Flask, render_template, request
from src.BoostingLRWrapper import BoostingLRWrapper
from src.utils import start_jvm, load_dataset_as_Instances, kendalls_tau, ndcg

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_FOLDER = os.path.join(BASE_DIR, 'datasets')
ALLOWED_EXTENSIONS = {'xarff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        print("Form submitted successfully!")

    # Get the list of available datasets
    datasets = [f for f in os.listdir(DATASETS_FOLDER) if allowed_file(f)]

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
            # User selected a dataset from the list
            dataset_path = os.path.join(DATASETS_FOLDER, dataset_choice)
            print("Selected dataset from list:", dataset_path)

        elif uploaded_file and allowed_file(uploaded_file.filename):
            # User uploaded a single dataset
            filename = uploaded_file.filename
            dataset_path = os.path.join(DATASETS_FOLDER, filename)
            uploaded_file.save(dataset_path)
            print("Uploaded single dataset:", dataset_path)

        elif uploaded_file1 and allowed_file(uploaded_file1.filename) and uploaded_file2 and allowed_file(uploaded_file2.filename):
            # User uploaded two datasets
            filename1 = uploaded_file1.filename
            filename2 = uploaded_file2.filename
            dataset_path1 = os.path.join(DATASETS_FOLDER, filename1)
            dataset_path2 = os.path.join(DATASETS_FOLDER, filename2)
            uploaded_file1.save(dataset_path1)
            uploaded_file2.save(dataset_path2)
            print("Uploaded first dataset:", dataset_path1)
            print("Uploaded second dataset:", dataset_path2)

        else:
            # If neither a file nor a selection was made, return an error message
            return render_template('input.html', datasets=datasets, error="Please select a dataset or upload a file.")

        # Map user choices to the actual functions
        dist_algo = kendalls_tau if dist_algo_choice == 'kendalltau' else ndcg
        dist_score = kendalls_tau if dist_score_choice == 'kendalltau' else ndcg

        # Run the BoostLR algorithm based on the selected option
        if dataset_choice or uploaded_file:
            # Use a single dataset
            result = run_boostlr(dataset_path, dist_algo, dist_score)
        elif uploaded_file1 and uploaded_file2:
            # Use two datasets (train and test)
            result = run_boostlr_with_two_datasets(dataset_path1, dataset_path2, dist_algo, dist_score)

        return render_template('results.html', score=result)

    return render_template('input.html', datasets=datasets)

def run_boostlr(dataset_path, dist_algo, dist_score):
    # Load dataset
    instances = load_dataset_as_Instances(dataset_path)

    # Initialize model
    model = BoostingLRWrapper(max_iterations=50, seed=7, dist_algo=dist_algo, dist_score=dist_score)

    # Train and score the model
    model.fit(instances)
    
    score = model.score(instances)

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


if __name__ == '__main__':
    start_jvm()  # Start JVM when the Flask app starts
    try:
        app.run(debug=True)
    finally:
        import jpype
        if jpype.isJVMStarted():
            jpype.shutdownJVM()  # Shutdown JVM when the app exits
