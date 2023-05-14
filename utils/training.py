from hpsklearn import HyperoptEstimator, random_forest_classifier, sgd_classifier, svc, \
    gradient_boosting_classifier, k_neighbors_classifier, logistic_regression, ada_boost_classifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from hyperopt import tpe
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

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
    clf_dict = {}
    for clf in classifiers:
        # fit this model
        model = clf
        cc = CalibratedClassifierCV(model, method='sigmoid', n_jobs=-1)
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

    classifiers = [random_forest_classifier,
                   gradient_boosting_classifier,
                   k_neighbors_classifier,
                   sgd_classifier,
                   ada_boost_classifier
                   ]
    clf_dict = {}
    for clf in classifiers:

        model = HyperoptEstimator(classifier=clf('this_clf'),
                                  max_evals=5,
                                  n_jobs=16,
                                  algo=tpe.suggest(verbose=False),
                                  preprocessing=[],
                                  verbose=False
                                  )
        # find optimized model
        model.fit(X_train, y_train, n_folds=5, cv_shuffle=True)
        model = model.best_model()['learner']

        # calibrate prob for this model
        cc = CalibratedClassifierCV(model, method='sigmoid', n_jobs=-1)
        cc.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')

        if f1 < threshold:
            continue
        # store predictions of this model
        f1_class = f1_score(y_test, y_pred, average=None)
        accuracy = cc.score(X_test, y_test)
        predictions = cc.predict_proba(X)[:, 1]
        clf_dict[clf.__name__] = {'model': cc,
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

    rf = RandomForestClassifier()

    sfs = SequentialFeatureSelector(rf, n_features_to_select='auto', tol=tol, scoring='f1_micro', n_jobs=-1, cv=5)
    sfs.fit(X, y)
    X = sfs.transform(X)

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
