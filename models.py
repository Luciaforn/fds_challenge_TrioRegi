from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42

cat_model = CatBoostClassifier(
    random_seed=SEED,
    iterations=300, 
    learning_rate=0.059240505706248246, 
    depth=4,                             
    l2_leaf_reg=9.891914312064191,      
    verbose=0,
    allow_writing_files=False 
)

xgb_model = XGBClassifier(
    random_state=SEED,
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
    n_estimators=400, 
    learning_rate=0.03332571528034287,
    max_depth=4,
    reg_lambda=4.383753444005586,
    subsample=0.8,
    colsample_bytree=1.0
)

lgbm_model = LGBMClassifier(
    random_state=SEED,
    objective='binary',
    n_estimators=500, 
    verbosity=0,
    learning_rate=0.02774827931875224,
    num_leaves=29,
    reg_lambda=0.5976366516509378,
    subsample=0.9231993799100238,
    colsample_bytree=0.8114266596517326
)