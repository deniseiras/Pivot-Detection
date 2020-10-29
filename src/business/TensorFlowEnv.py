
class TensorFlowEnv:

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.tf_files_dir = self.root_dir + 'experiments/'
        self.summaries_dir = self.root_dir + 'training_summaries/'
        self.model_dir = self.root_dir + 'models/'
