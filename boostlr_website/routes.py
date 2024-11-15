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
    datasets = [f for f in os.listdir(DATASETS_FOLDER) if allowed_file(f)]
    result = None
    predictions_filename = None

    if request.method == 'POST':
        dataset_choice = request.form.get('dataset_choice')
        uploaded_file = request.files.get('uploaded_file')

        dist_algo_choice = request.form.get('dist_algo')
        dist_score_choice = request.form.get('dist_score')

        if dataset_choice and dataset_choice != '':
            dataset_path = os.path.join(DATASETS_FOLDER, dataset_choice)
        elif uploaded_file and allowed_file(uploaded_file.filename):
            filename = uploaded_file.filename
            dataset_path = os.path.join(DATASETS_FOLDER, filename)
            uploaded_file.save(dataset_path)
        else:
            return render_template('algo.html', datasets=datasets, error="Please select a dataset or upload a file.")

        dist_algo = kendalls_tau if dist_algo_choice == 'kendalltau' else ndcg
        dist_score = kendalls_tau if dist_score_choice == 'kendalltau' else ndcg

        result, predictions_filename = run_boostlr(dataset_path, dist_algo, dist_score)

    return render_template('algo.html', datasets=datasets, score=result, predictions_filename=predictions_filename)

@app.route('/download/<filename>')
def download_predictions(filename):
    return send_from_directory(PREDICTIONS_FOLDER, filename, as_attachment=True)

