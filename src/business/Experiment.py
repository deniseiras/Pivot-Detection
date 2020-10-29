# from sqlalchemy import Integer, Float, String, Boolean, Column, ForeignKey
# from sqlalchemy.orm import relationship
# from sqlalchemy_dao import Model


class Experiment:

    def __init__(self, name, tf_env):
        self.name = name
        self.tf_env = tf_env
        self.exp_root_dir = self.tf_env.tf_files_dir + self.name + '/'
        self.log_dir = self.exp_root_dir + 'log/'
        self.trains_dir = self.exp_root_dir + 'trains/'
        self.tests_dir = self.exp_root_dir + 'to_test/'
        self.samples_dir = self.exp_root_dir + 'samples/'
