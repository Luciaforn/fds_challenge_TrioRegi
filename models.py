from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=6, 
    l2_leaf_reg=5, 
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False
)

xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=SEED,
    tree_method='hist',
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
    
    colsample_bytree=1.0,
    learning_rate=0.03,
    max_depth=4, 
    n_estimators=400, # Userà tutti i 400 estimators
    reg_lambda=1, 
    subsample=0.8
)

lgbm_model = LGBMClassifier(
    objective='binary',
    random_state=SEED,
    n_estimators=500, # Userà tutti i 500 estimators
    learning_rate=0.05,
    num_leaves=31, 
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0, 
    verbosity=0
)

