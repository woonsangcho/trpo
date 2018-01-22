from abc import ABCMeta, abstractmethod

class Environment(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_environment(self):
        pass

    @abstractmethod
    def reset_environment(self):
        pass

    @abstractmethod
    def get_state_shape(self):
        pass

    @abstractmethod
    def get_num_action(self):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def set_seed(self, seed):
        pass

    @abstractmethod
    def perform_action(self, action):
        pass

    @abstractmethod
    def get_action_bound(self):
        pass

    @abstractmethod
    def help_message(self):
        pass

