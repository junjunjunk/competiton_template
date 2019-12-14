from abc import ABCMeta, abstractmethod
import argparse
import inspect
from logs.time_keeper import stop_watch
import pandas as pd
from pathlib import Path
import pickle
import re
import time


class Feature(metaclass=ABCMeta):
    """
    Base class
    Inherit this class and, implement create_features method.
    """
    prefix = ''
    suffix = ''
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'{self.name}_train.pkl'
        self.test_path = Path(self.dir) / f'{self.name}_test.pkl'
    
    @stop_watch(self.name)
    def run(self):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.train.to_pickle(str(self.train_path))
        self.test.to_pickle(str(self.test_path))

    def load(self):
        self.train = pd.read_pickle(str(self.train_path))
        self.test = pd.read_pickle(str(self.test_path))


"""
command line tool
"""
def get_arguments():
    """
    -f [--force] overwrite mode
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    """
    namespaceを渡してそこに含まれる特徴量が保存済みかどうか確認
    存在しない場合は計算
    """
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()