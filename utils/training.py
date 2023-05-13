from hpsklearn import HyperoptEstimator, random_forest_classifier, xgboost_classification, \
    sgd_classifier, svc, gradient_boosting_classifier, k_neighbors_classifier, gradient_boosting_regressor, \
    linear_regression, elastic_net, logistic_regression, xgboost_regression, random_forest_regressor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from hyperopt import tpe
from hyperopt import hp
import pickle


def model_training(train_file, y, threshold):

    with open(train_file, "rb") as fp:
        X = pickle.load(fp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifiers = [random_forest_classifier,
                   gradient_boosting_classifier,
                   k_neighbors_classifier,
                   sgd_classifier,
                   svc,
                   xgboost_classification]
    clf_dict = {}
    for clf in classifiers:

        model = HyperoptEstimator(classifier=clf('this_clf'),
                                  max_evals=5,
                                  algo=tpe.suggest,
                                  preprocessing=[]
                                  )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')

        if f1 >= threshold:
            # store predictions of this model
            f1_class = f1_score(y_test, y_pred, average=None)
            accuracy = model.score(X_test, y_test)
            predictions = model.predict(X)
            clf_dict[clf] = {'model': model,
                             'accuracy': accuracy,
                             'f1': f1,
                             'f1_0': f1_class[0],
                             'f1_1': f1_class[1],
                             'predictions': predictions}

    return clf_dict
