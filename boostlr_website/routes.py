import os
import redis
from flask import render_template, request, jsonify, session, send_from_directory
from boostlr_website import app
from boostlr_website.constants import *
from boostlr_website.utils import allowed_file
from boostlr_website.tasks import run_boostlr_task

# Initialize Redis client
redis_client = redis.Redis(host='redis', port=6379, db=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/algo', methods=['GET', 'POST'])
def algo():
    datasets = sorted([f for f in os.listdir(DATASETS_FOLDER) if allowed_file(f)])

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
            return render_template('algo.html', datasets=datasets, error="Invalid input.")

        # Start Celery task
        task = run_boostlr_task.delay(dataset_path, dist_algo_choice, dist_score_choice)

        # Store task_id in Redis
        redis_client.set('latest_task_id', task.id)

        return jsonify({"task_id": task.id})

    # Check if there is a running task
    latest_task_id = redis_client.get('latest_task_id')
    if latest_task_id:
        latest_task_id = latest_task_id.decode('utf-8')
        task = run_boostlr_task.AsyncResult(latest_task_id)

        if task.state == 'SUCCESS':
            result = task.result['result']
            predictions_filename = task.result['predictions_filename']
            return render_template('algo.html', datasets=datasets, score=result, predictions_filename=predictions_filename)

        elif task.state == 'PENDING':
            return render_template('algo.html', datasets=datasets, message="Task is still running, please wait...")

        elif task.state == 'FAILURE':
            return render_template('algo.html', datasets=datasets, error="Task failed. Please try again.")

    # Default render for GET request
    return render_template('algo.html', datasets=datasets, score=None, predictions_filename=None)


@app.route('/status/<task_id>')
def check_status(task_id):
    if task_id == 'latest':
        task_id = redis_client.get('latest_task_id')
        if not task_id:
            return jsonify({"status": "No task found"}), 404
        task_id = task_id.decode('utf-8')

    task = run_boostlr_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return jsonify({"status": "SUCCESS", "result": task.result})
    elif task.state == 'PENDING':
        return jsonify({"status": "PENDING"})
    else:
        return jsonify({"status": "FAILURE"}), 500


@app.route('/download/<filename>')
def download_predictions(filename):
    return send_from_directory(PREDICTIONS_FOLDER, filename, as_attachment=True)
