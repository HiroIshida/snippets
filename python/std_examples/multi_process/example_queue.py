import multiprocessing
import numpy as np
# asked at:
# https://stackoverflow.com/questions/71322856

def run_mp(images, f_work, n_worker):

    def f_worker(worker_index, data, q, result_q):
        print("worker {} started".format(worker_index))
        while True:
            image_idx = q.get() # Blocking get
            if image_idx is None: # Sentinel?
                break # We are done!
            print('processing image idx {}'.format(image_idx))
            image_out = f_work(data, image_idx)
            result_q.put((image_out, image_idx))
        print("worker {} finished".format(worker_index))
        return

    q = multiprocessing.Queue()
    for img_idx in range(len(images)):
        q.put(img_idx)

    # Add sentinels:
    for _ in range(n_worker):
        q.put(None)

    result_q = multiprocessing.Queue()

    processes = list()
    for i in range(n_worker):
        process = multiprocessing.Process(target=f_worker, args=(i, images, q, result_q))
        # We do not need daemon processes now:
        #process.daemon = True
        process.start()
        processes.append(process)

    # If we are interested in the results, we must process the result queue
    # before joining the processes. We are expecting 20 results, so:
    results = [result_q.get() for _ in range(20)]
    print(results)

    for process in processes:
        process.join()


images = [np.random.randn(100, 100) for _ in range(20)]

f = lambda image_list, idx: image_list[idx] + np.random.randn()
run_mp(images, f, 2)
