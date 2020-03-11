from abc import ABCMeta, abstractmethod
import argparse
import inspect
import pandas as pd
from pathlib import Path
import pickle
from contextlib import contextmanager

import csv

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class Feature(metaclass=ABCMeta):
    """
    Base class
    Inherit this class and, implement create_features method.
    """
    prefix = ''
    suffix = '' # ex) period, version, train, test
    dir = '.'
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.df = pd.DataFrame()
        self.path = Path(self.dir) / f'{self.name}.pkl'
    
    # @stop_watch(self.name)
    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''
            self.df.columns = prefix + self.df.columns + suffix
        return self
    
    @abstractmethod
    def create_features(self):
        raise NotImplementedError
    
    def save(self):
        self.df.to_pickle(str(self.path))

    def load(self):
        self.df = pd.read_pickle(str(self.path))

    def create_details(self,file_name,col_name,description):
        file_path = dir + file_name

        if not os.path.isfile(file_path):
            with open(file_path,"w"):pass

        with open(file_path,"r+") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

            col = [line for line in lines if line.split(',')[0] == col_name ]
            if len(col)!=0:
                return

            writer = csv.writer()
            writer.writerow([col_name,description])
"""
command line tool
"""
def get_arguments():
    """
    -f [--force, --overwrite] overwrite mode
    """
    parser = argparse.ArgumentParser(description='Process features.')
    parser.add_argument('-f', '--force', '--overwrite',action='store_true', help='Overwrite existing files')

    return parser.parse_args()


def get_features(namespace):
    """
    Featureを継承したクラスをインスタンス化して返すiterator
    """
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    """
    namespaceを渡してそこに含まれる特徴量が保存済みかどうか確認
    [1] 存在しない場合 -> 計算
    [2] 存在する場合 -> 処理をスキップ
    [3] overwriteモード -> 計算
    """
    for f in get_features(namespace):
        if f.df.path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save().create_details()

