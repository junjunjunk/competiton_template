import lightgbm as lgb
import logging

from logs.logger import log_evaluation

def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lgbm_params):

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    logging.debug(lgbm_params)

    # ロガーの作成
    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=30)]

    model = lgb.train(
        params = lgbm_params, 
	train_set = lgb_train,
        valid_sets = lgb_eval,

        # ログ
        callbacks=callbacks
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return model,y_pred
