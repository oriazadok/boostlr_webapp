import os
from flask import render_template, request, send_from_directory

from boostlr_website import app
from boostlr_website.constants import *
from boostlr_website.utils import *


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/algo', methods=['GET', 'POST'])
def algo():
    datasets = sorted([f for f in os.listdir(DATASETS_FOLDER) if allowed_file(f)])

    if request.method == 'POST':
        # Get dataset choices and uploaded files
        dataset_choice = request.form.get('dataset_choice')

        uploaded_file = request.files.get('uploaded_file')
        
        uploaded_file1 = request.files.get('uploaded_file1')
        uploaded_file2 = request.files.get('uploaded_file2')

        dist_algo_choice = request.form.get('dist_algo')
        dist_score_choice = request.form.get('dist_score')

        # Determine which dataset(s) to use
        if dataset_choice and dataset_choice != '':
            dataset_path = os.path.join(DATASETS_FOLDER, dataset_choice)
        elif uploaded_file and allowed_file(uploaded_file.filename):
            filename = uploaded_file.filename
            dataset_path = os.path.join(DATASETS_FOLDER, filename)
            uploaded_file.save(dataset_path)
        elif uploaded_file1 and allowed_file(uploaded_file1.filename) and uploaded_file2 and allowed_file(uploaded_file2.filename):
            # Handle double upload case
            filename1 = uploaded_file1.filename
            filename2 = uploaded_file2.filename
            dataset_path1 = os.path.join(DATASETS_FOLDER, filename1)
            dataset_path2 = os.path.join(DATASETS_FOLDER, filename2)
            uploaded_file1.save(dataset_path1)
            uploaded_file2.save(dataset_path2)
        else:
            # If no valid input is provided, show an error
            return render_template('algo.html', datasets=datasets, score=None, error="Please select a dataset or upload a file.")

        # Map user choices to functions
        dist_algo = kendalls_tau if dist_algo_choice == 'kendalltau' else ndcg
        dist_score = kendalls_tau if dist_score_choice == 'kendalltau' else ndcg

        # Run the BoostLR algorithm based on the input
        if dataset_choice or uploaded_file:
            result, predictions_filename = run_boostlr(dataset_path, dist_algo, dist_score)
        elif uploaded_file1 and uploaded_file2:
            result, predictions_filename = run_boostlr_with_two_datasets(dataset_path1, dataset_path2, dist_algo, dist_score)

        return render_template('algo.html', datasets=datasets, score=result, predictions_filename=predictions_filename)

    return render_template('algo.html', datasets=datasets, score=None, predictions_filename=None)


@app.route('/download/<filename>')
def download_predictions(filename):
    return send_from_directory(PREDICTIONS_FOLDER, filename, as_attachment=True)

