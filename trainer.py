
from logger import Auditor

class GeneralTrainer(object):
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.environ_string = kwargs['environ_string']
        self.max_episode_count = kwargs['max_episode_count']
        self.seed = kwargs['seed']
        self.instance_name = kwargs['environ_string'] + ('_seed_%d' %  self.seed)
        self.episode_count = 0
        self.auditor = Auditor(self.instance_name)
