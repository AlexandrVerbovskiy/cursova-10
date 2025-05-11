import threading
import queue
from random_forest import random_forest_worker

if __name__ == '__main__':
    task_queue = queue.Queue()
    worker_thread = threading.Thread(target=random_forest_worker, args=(task_queue,),
                                     kwargs={'props': {"new_model_id": 16}})
    worker_thread.start()
    task_queue.put("Train Random Forest Model")
