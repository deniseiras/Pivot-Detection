import time
from src.label_image_func import *
from src.debugutils import DebugUtils
import os

debug = DebugUtils.get_instance()


# TODO : open all files first - not working - for adaptation in reading blocks
# File "/home/denis/miniconda3/envs/pivotDetection/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 307, in __init__
#     'Tensor. (%s)' % (fetch, str(e)))
# ValueError: Fetch argument <tf.Tensor 'truediv:0' shape=(1, 224, 224, 3) dtype=float32> cannot be interpreted as a Tensor. (Tensor Tensor("truediv:0", shape=(1, 224, 224, 3), dtype=float32) is not an element of this graph.)
# python-BaseException
def test_dir(exp):

    # default values
    if exp["architecture"] == 'inception_v3':
        input_layer = "Mul"
        exp["input_height"] = 299 if exp["input_height"] is None else exp["input_height"]
        exp["input_width"] = 299 if exp["input_width"] is None else exp["input_width"]
        exp["input_mean"] = 0 if exp["input_mean"] is None else exp["input_mean"]
        exp["input_std"] = 255 if exp["input_std"] is None else exp["input_std"]
    else:
        input_layer = "input"
        exp["input_height"] = 224 if exp["input_height"] is None else exp["input_height"]
        exp["input_width"] = 224 if exp["input_width"] is None else exp["input_width"]
        exp["input_mean"] = 128 if exp["input_mean"] is None else exp["input_mean"]
        exp["input_std"] = 128 if exp["input_std"] is None else exp["input_std"]

    output_layer = "final_result"
    model_file = exp["model_file"]
    label_file = exp["label_file"]
    graph = load_graph(model_file)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    files_tested = 0
    for root, dirs, files in os.walk(exp["test_dir"]):
        all_results = [None] * len(files)
        # tensors = [None] * len(files)
        total_time = 0
        for file in files:
            # if file.endswith(".jpg"):
            file_name = os.path.join(root, file)
            normalized = read_tensor_from_image_file_no_test(file_name,
                                                    input_height=exp["input_height"],
                                                    input_width=exp["input_width"],
                                                    input_mean=exp["input_mean"],
                                                    input_std=exp["input_std"])
            millis_init = time.time()
            with tf.Session() as new_sess:
                t = new_sess.run(normalized)
                labels_results = run_session_graph(graph, input_operation, label_file, output_operation, t)
                all_results[files_tested] = labels_results
            millis_end = time.time()
            time_exec = millis_end - millis_init
            total_time += time_exec
            # tensors[files_opened] = normalized
            files_tested += 1

    if files_tested == 0:
        debug.msg("No files tested")
    else:
        # millis_init = time.time()
        # files_tested = 0
        # for normalized in tensors:
        #     with tf.Session() as new_sess:
        #         t = new_sess.run(normalized)
        #     labels_results = run_session_graph(graph, input_operation, label_file, output_operation, t)
        #     all_results[files_tested] = labels_results
        #     files_tested = files_tested + 1
        #
        # millis_end = time.time()
        # time_exec = millis_end - millis_init
        #
        avg_time_tested = total_time / files_tested
        debug.msg("\n{0} files tested in {1} seconds. Average time = {2} miliseconds".format(files_tested,
                                                                                             total_time / 1000,
                                                                                             avg_time_tested))
        debug.flush_file()

    return all_results, files_tested, total_time


def run_session_graph(graph, input_operation, label_file, output_operation, t):
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
    results = np.squeeze(results)
    top_k = results.argsort()[-10:][::-1]
    labels = load_labels(label_file)
    labels_results = {}
    for i in top_k:
        labels_results[labels[i]] = results[i]
        # debug.msg(labels[i], ' {0:.4f}%'.format(results[i]*100))
    return labels_results


def test(exp, file_name):
    # debug.msg("\n ===> TESTING file {} ".format(file_name))
    millis_init = time.time()

    # default values
    if exp["architecture"] == 'inception_v3':
        input_layer = "Mul"
        exp["input_height"] = 299 if exp["input_height"] is None else exp["input_height"]
        exp["input_width"] = 299 if exp["input_width"] is None else exp["input_width"]
        exp["input_mean"] = 0 if exp["input_mean"] is None else exp["input_mean"]
        exp["input_std"] = 255 if exp["input_std"] is None else exp["input_std"]
    else:
        input_layer = "input"
        exp["input_height"] = 224 if exp["input_height"] is None else exp["input_height"]
        exp["input_width"] = 224 if exp["input_width"] is None else exp["input_width"]
        exp["input_mean"] = 128 if exp["input_mean"] is None else exp["input_mean"]
        exp["input_std"] = 128 if exp["input_std"] is None else exp["input_std"]

    output_layer = "final_result"
    model_file = exp["model_file"]
    label_file = exp["label_file"]

    # TODO - do once
    graph = load_graph(model_file)

    t = read_tensor_from_image_file(file_name,
                                    input_height=exp["input_height"],
                                    input_width=exp["input_width"],
                                    input_mean=exp["input_mean"],
                                    input_std=exp["input_std"])

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name);
    labels_results = run_session_graph(graph, input_operation, label_file, output_operation, t)
    millis_end = time.time()
    time_exec = millis_end - millis_init
    return labels_results, time_exec
