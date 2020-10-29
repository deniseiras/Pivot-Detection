import os, shutil, time
from src.retrain_func import mainRetrain, parseArguments
from src.debugutils import DebugUtils

debug = DebugUtils.get_instance()

def train(pars):

    FLAGS = parseArguments()
    FLAGS.model_dir = pars["model_dir"]
    FLAGS.summaries_dir = pars["summaries_dir"]
    FLAGS.bottleneck_dir = pars["bottlenecks_dir"]
    FLAGS.output_graph = pars["model_file"]
    FLAGS.output_labels = pars["label_file"]
    FLAGS.architecture = pars["architecture"]
    FLAGS.image_dir = pars["train_dir"]
    FLAGS.how_many_training_steps = pars["train_steps"]
    FLAGS.learning_rate = pars["learning_rate"] # - já usa GradientDescentOptimizer
    FLAGS.print_misclassified_test_images = True
    # TODO tests changing the values below
    FLAGS.validation_percentage = pars["validation_percentage"]
    FLAGS.testing_percentage = pars["testing_percentage"]
    FLAGS.random_brightness = pars["random_brightness"]
    FLAGS.random_scale = pars["random_scale"]
    FLAGS.random_crop = pars["random_crop"]
    FLAGS.flip_left_right = pars["flip_left_right"]

    if os.path.isdir(FLAGS.summaries_dir):
        try:
            shutil.rmtree(FLAGS.summaries_dir, ignore_errors=False)
        except:
            # TODO - Denis - not erasing - check retrain_func.py - prepare
            print("Could not erase {}".format(FLAGS.summaries_dir))
            # exit(-1)

    print(">>> Tensor face_trainer started ... ")
    millis_init = time.time()
    test_accuracy = mainRetrain(FLAGS, [])
    millis_end = time.time()
    time_exec = millis_end - millis_init
    print(">>> Tensor face_trainer end ")
    debug.msg("Train concluded in {} seconds".format(time_exec))
    debug.flush_file()
    return test_accuracy

# --model_dir=models/ --summaries_dir=training_summaries/"${ARCHITECTURE}" --bottleneck_dir=$FOTOSDIR/bottlenecks
# --output_graph=$FOTOSDIR/retrained_graph.pb --output_labels=$FOTOSDIR/retrained_labels.txt --architecture="${ARCHITECTURE}" --image_dir=$FOTOSDIR
# --how_many_training_steps=500