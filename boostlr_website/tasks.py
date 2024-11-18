from celery import Celery
from boostlr_website.utils import run_boostlr, kendalls_tau, ndcg
from boostlr_website.src.utils import start_jvm

app = Celery('tasks', broker='redis://redis:6379/0', backend='redis://redis:6379/0')

@app.task
def run_boostlr_task(dataset_path, dist_algo_choice, dist_score_choice):
    start_jvm()
    dist_algo = kendalls_tau if dist_algo_choice == 'kendalltau' else ndcg
    dist_score = kendalls_tau if dist_score_choice == 'kendalltau' else ndcg
    result, predictions_filename = run_boostlr(dataset_path, dist_algo, dist_score)
    return {"result": result, "predictions_filename": predictions_filename}
