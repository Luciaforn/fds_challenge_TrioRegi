from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42

cat_model = CatBoostClassifier(
    iterations=300, # Puoi tenere 300, Optuna non l'ha testato
    learning_rate=0.059240505706248246, # <-- Nuovo
    depth=4,                             # <-- Nuovo
    l2_leaf_reg=9.891914312064191,      # <-- Nuovo
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False # Aggiungi questo per sicurezza
)

xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=SEED,
    tree_method='hist',
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
    n_estimators=400, # Manteniamo 400 estimators come da tuo codice
    learning_rate=0.03332571528034287,
    max_depth=4,
    reg_lambda=4.383753444005586,
    subsample=0.8,
    colsample_bytree=1.0
)

lgbm_model = LGBMClassifier(
    objective='binary',
    random_state=SEED,
    n_estimators=500, # Manteniamo 500 estimators come da tuo codice
    verbosity=0,
    learning_rate=0.02774827931875224,
    num_leaves=29,
    reg_lambda=0.5976366516509378,
    subsample=0.9231993799100238,
    colsample_bytree=0.8114266596517326
)