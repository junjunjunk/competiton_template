import json
from logs import logger
import logging
import datetime
import lightgbm as lgb
import sys
import pandas as pd 

class LightGBM:
    def __init__(self,params,save_name):
        self.train = None
        self.valid = None
        self.test = None
        self.params = params
        self.model = None
        self.save_name = save_name
    
    def set_data(self,train=None,valid=None,test=None):
        use_features, target = self.params['features'], self.params['target']
        cate_cols = self.params['cate_cols'] if self.params['cate_cols'] is not None else ""

        if train is not None:
            self.train = lgb.Dataset(train[use_features], train[target], categorical_feature=cate_cols)
        if valid is not None:
            self.valid = lgb.Dataset(valid[use_features], valid[target], categorical_feature=cate_cols)
        if test is not None:
            self.test = lgb.Dataset(test[use_features], test[target], categorical_feature=cate_cols)

        return True

    def fit(self,callbacks=None):
        watchlist = [self.train, self.valid]
        self.model = lgb.train(self.params['model_params'], self.train,valid_sets=watchlist, callbacks=callbacks)
        return model
    
    def save_importance(self):
        if model is not None:
            ax = lgb.plot_importance(self.model)
            ax.figure.tight_layout()
            ax.figure.savefig(self.save_name+'.png')
            return True

        return False

    def save_predict(self,add_info=None): 
        predict = pd.DataFrame({params['target']:self.model.predict(test)})
        predict = pd.concat([predict,add_info],axis=1)
        predict.to_csv('../data/output/'+save_name+'.csv',index=False)
        return True

if __name__ == '__main__':
    args = sys.argv
    with open(args[1]) as f:
        params = json.load(f)

    now = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')
    save_name = params['model']+'_'+now

    # make logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    sc = logging.StreamHandler()
    logger.addHandler(sc)

    fh = logging.FileHandler('./logs/logfile/'+save_name+'.log')
    logger.addHandler(fh)

    callbacks = [logger.log_evaluation(logger, period=10)]

    logger.info("[Features]")
    for feature in params['features']:
        logger.info(feature)

    logger.info("[Params]")
    for k, v in params['model_params'].items():
        logger.info("{}: {}".format(k,v))

    # make data


    # train model
    lgbm = LightGBM(params,save_name)
    lgbm.set_data(train,valid,test)
    lgbm.fit()
    lgbm.save_importance()
    lgbm.save_predict()

    



    