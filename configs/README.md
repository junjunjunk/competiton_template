jsonファイルで各種実験設定を管理
「使用する特徴量」
「使用するパラメータ」
「評価(CV)の設定」

命名はmodels・logディレクトリ内のフォルダと対応させる
publicスコアをディレクトリのpredixにつける
{
    "features": [
        "age",
        "embarked",
        "family_size",
        "fare",
        "pclass",
        "sex"
    ],
    "lgbm_params": {
        "learning_rate": 0.1,
        "num_leaves": 8,
        "boosting_type": "gbdt",
        "colsample_bytree": 0.65,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "objective": "multiclass",
        "num_class": 2
    },
    "loss": "multi_logloss",
    "target_name": "Survived",
    "ID_name": "PassengerId"
}