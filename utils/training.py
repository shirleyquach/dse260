from hpsklearn import HyperoptEstimator, random_forest_classifier, sgd_classifier, svc, \
    gradient_boosting_classifier, k_neighbors_classifier, logistic_regression, ada_boost_classifier, \
    xgboost_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from hyperopt import tpe
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from collections import OrderedDict
import xgboost as xgb
import optuna

import logging


def train_model_fast(file_path, train_file, y, threshold):
    with open(file_path + train_file, "rb") as fp:
        X = pickle.load(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifiers = [RandomForestClassifier(),
                   GradientBoostingClassifier(),
                   KNeighborsClassifier(),
                   SGDClassifier(),
                   SVC(probability=True),
                   AdaBoostClassifier(),
                   LogisticRegression()
                   ]
    clf_dict = OrderedDict()
    for clf in classifiers:
        # fit this model
        model = clf
        model.fit(X_train, y_train)
        cc = CalibratedClassifierCV(model, method='sigmoid', n_jobs=-1, cv='prefit')
        cc.fit(X_train, y_train)

        # evaluate this model
        y_pred = cc.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')

        # if this model is bad, skip it
        if f1 < threshold:
            continue

        # store predictions of this model
        f1_class = f1_score(y_test, y_pred, average=None)
        accuracy = cc.score(X_test, y_test)
        predictions = cc.predict_proba(X)[:, 1]
        model_name = type(clf).__name__
        clf_dict[model_name] = {'model': cc,
                                'accuracy': accuracy,
                                'f1': f1,
                                'f1_0': f1_class[0],
                                'f1_1': f1_class[1],
                                'predictions': predictions}

    return clf_dict


def train_model_opt(file_path, train_file, y, threshold):
    with open(file_path + train_file, "rb") as fp:
        X = pickle.load(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifiers = [#random_forest_classifier,
                   #gradient_boosting_classifier,
                   #k_neighbors_classifier,
                   #gaussian_process_classifier,
                   xgboost_classification
                   ]
    clf_dict = OrderedDict()
    for clf in classifiers:

        model = HyperoptEstimator(classifier=clf('this_clf'),
                                  max_evals=5,
                                  n_jobs=16,
                                  algo=tpe.suggest,
                                  preprocessing=[],
                                  verbose=False
                                  )
        # find optimized model
        model.fit(X_train, y_train, n_folds=10, cv_shuffle=True)
        model = model.best_model()['learner']
        model.fit(X_train, y_train)

        # calibrate prob for this model
        # cc = CalibratedClassifierCV(model, method='sigmoid', n_jobs=-1, cv='prefit')
        # cc.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')

        if f1 < threshold:
            continue
        # store predictions of this model
        f1_class = f1_score(y_test, y_pred, average=None)
        accuracy = model.score(X_test, y_test)
        predictions = model.predict_proba(X)[:, 1]
        clf_dict[clf.__name__] = {'model': model,
                                  'accuracy': accuracy,
                                  'f1': f1,
                                  'f1_0': f1_class[0],
                                  'f1_1': f1_class[1],
                                  'predictions': predictions}

    return clf_dict


def train_model_eval_ensemble(X, y, tol):
    loggers_to_shut_up = [
        "hyperopt.tpe",
        "hyperopt.fmin",
        "hyperopt.pyll.base",
    ]
    for logger in loggers_to_shut_up:
        logging.getLogger(logger).setLevel(logging.ERROR)

    start_size = X.shape[1]
    rf = RandomForestClassifier()

    sfs = SequentialFeatureSelector(rf, n_features_to_select='auto', tol=tol, scoring='f1_micro', n_jobs=-1, cv=5)
    sfs.fit(X, y)
    X = sfs.transform(X)

    print('Models removed:', start_size - X.shape[1])

    rf = HyperoptEstimator(classifier=random_forest_classifier('this_clf'),
                           max_evals=10,
                           n_jobs=8,
                           algo=tpe.suggest,
                           preprocessing=[],
                           verbose=False
                           )
    rf.fit(X, y, n_folds=5, cv_shuffle=True)
    rf = rf.best_model()['learner']
    rf.fit(X, y)

    cc = CalibratedClassifierCV(rf, method='sigmoid', n_jobs=-1, cv='prefit')
    cc.fit(X, y)

    return sfs.get_support(), cc


def train_model_search(file_path, train_file, y, best_model, best_accuracy, best_file, max_evals):
    loggers_to_shut_up = [
        "hyperopt.tpe",
        "hyperopt.fmin",
        "hyperopt.pyll.base",
    ]
    for logger in loggers_to_shut_up:
        logging.getLogger(logger).setLevel(logging.ERROR)

    with open(file_path + train_file, "rb") as fp:
        X = pickle.load(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(train_file)

    # hyperopt models
    classifiers = [random_forest_classifier,
        # gradient_boosting_classifier,
        # k_neighbors_classifier,
        # gaussian_process_classifier,
        # xgboost_classification
    ]
    for clf in classifiers:
        model = HyperoptEstimator(classifier=clf('clf', n_jobs=-1),
                                  n_jobs=16,
                                  max_evals=max_evals,
                                  algo=tpe.suggest,
                                  fit_increment=5,
                                  preprocessing=[])
        model.fit(X_train, y_train, n_folds=10, cv_shuffle=True)

        accuracy = model.score(X_test, y_test)

        if accuracy > best_accuracy:
            model = model.best_model()['learner']
            model.fit(X_train, y_train)
            best_model = model
            best_accuracy = accuracy
            best_file = train_file

    return best_model, best_accuracy, best_file


def train_model_search_opt(file_path, train_file, y, best_model, best_accuracy, best_file, max_evals):
    def objective(trial):

        d_train = xgb.DMatrix(X_train, label=y_train)

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            # "tree_method":'gpu_hist',
            # "gpu_id": 0,
            "n_jobs": -1,
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.4, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 5900, 200),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
            "colsample_bylevel": trial.suggest_int("colsample_bylevel", 0.5, 1),
        }

        if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 11)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
        history = xgb.cv(param, d_train, num_boost_round=100, callbacks=[pruning_callback])

        mean_auc = history["test-auc-mean"][-1]
        return mean_auc

    with open(file_path + train_file, "rb") as fp:
        X = pickle.load(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(train_file)

    # hyperopt models
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(pruner=pruner, direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=max_evals, n_jobs=-1)
    trial = study.best_trial

    value = trial.value

    if value > best_accuracy:
        xgb_params =study.best_params
        model = xgb.XGBClassifier(**xgb_params)
        model = model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        best_model = model
        best_accuracy = accuracy
        best_file = train_file

    return best_model, best_accuracy, best_file