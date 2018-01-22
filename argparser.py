
import argparse

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=None)
        self._add_arguments()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments(self):
        self.parser.add_argument('--environ-string', default="HalfCheetah-v1", help='Environment id')
        self.parser.add_argument('--max-episode-count', default=15000, type=int, help='max number of training episodes') # 2000
        self.parser.add_argument('--seed', default=1989, type=int, help='seed')







