import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from tqdm import tqdm
from hpsklearn import HyperoptEstimator, random_forest_classifier, xgboost_classification, \
    sgd_classifier, svc, gradient_boosting_classifier, k_neighbors_classifier, gradient_boosting_regressor, \
    linear_regression, elastic_net, logistic_regression, xgboost_regression, random_forest_regressor
from hyperopt import hp

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, load_ground_truth, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import fit_feature_reduction_algorithm_pca_ica, fit_feature_reduction_algorithm_final_layer, \
    use_feature_reduction_algorithm_pca_ica, fit_feature_reduction_algorithm_pca_model_ica_opt
    # fit_feature_reduction_algorithm_pca_model_ica
    # fit_feature_reduction_algorithm,
    # use_feature_reduction_algorithm,

from sklearn.preprocessing import StandardScaler
# from archs import Net2, Net3, Net4, Net5, Net6, Net7, Net2r, Net3r, Net4r, Net5r, Net6r, Net7r, Net2s, Net3s, Net4s, Net5s, Net6s, Net7s
import torch

import time
from datetime import datetime

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.scale_parameters_filepath = scale_parameters_filepath
        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round
        self.model_skew = {
            "__all__": metaparameters["infer_cyber_model_skew"],
        }

        self.input_features = metaparameters["train_input_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp, cls=NpEncoder)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 1):  # changes the number of loops on trainer
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)
        del model_repr
        del model_repr_list

        check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        # Flatten models
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = None
        X = None
        y = []
        # layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)
        # layer_transform = fit_feature_reduction_algorithm_pca_ica(flat_models, self.weight_table_params, self.input_features)
        # layer_transform = fit_feature_reduction_algorithm_final_layer(flat_models, self.weight_table_params, self.input_features)

        layer_pca_components = [25, 30, 100, 200]
        arch_pca_components = [100, 200, 300] # arch pca components must be less than the number of samples in each architechture
        #arch_pca_components = [10, 15, 25] # use these for testing
        dataset_pca_components = [2, 4, 6]
        ica_components = [2, 4, 6]
        kernels = ['poly', 'linear', 'rbf', 'sigmoid', 'cosine']
        fc = fit_feature_reduction_algorithm_pca_model_ica_opt(file_path=self.learned_parameters_dirpath,
                                                               model_dict=flat_models,
                                                               layer_pca_components=layer_pca_components,
                                                               arch_pca_components=arch_pca_components,
                                                               dataset_pca_components=dataset_pca_components,
                                                               ica_components=ica_components,
                                                               kernels=kernels)
        print('Files Generated: ', fc)
        '''
        for kernel in tqdm(kernels, desc='kernels'):
            print('Generating: ', kernel)
            for ica_component in ica_components:
                for layer_pca_component in layer_pca_components:
                    for arch_pca_component in arch_pca_components:
                        for dataset_pca_component in dataset_pca_components:
                            try:
                                X = fit_feature_reduction_algorithm_pca_model_ica(model_dict=flat_models,
                                                                                  layer_pca_component=layer_pca_component,
                                                                                  arch_pca_component=arch_pca_component,
                                                                                  dataset_pca_component=dataset_pca_component,
                                                                                  ica_component=ica_component,
                                                                                  kernel=kernel)
                                # print("Feature reduction applied. Creating feature file...")
                                with open(self.learned_parameters_dirpath + f'2023-05-05_train_num_lpca_{layer_pca_component}_apca_{arch_pca_component}_dpca_{dataset_pca_component}_ica_{ica_component}_kernel_{kernel}.pkl', "wb") as fp:
                                    pickle.dump(X, fp)
                            except Exception as er1:
                                er1  # print(f"Failed - l_PCA: {layer_pca_component}, a_PCA: {arch_pca_component}, d_PCA: {dataset_pca_component},ICA: {ica_component}, kernel:{kernel}\n{er1}")
        '''
        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            # logger.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])  # change to use model_layer_map
                model_index += 1
        with open(self.learned_parameters_dirpath + f'2023-05-06_target_num_pca_ica.pkl', "wb") as fp:
            pickle.dump(y, fp)

            '''
            model_feats = use_feature_reduction_algorithm(
                layer_transform[model_arch], model
            )
            if X is None:
                X = model_feats
                continue

            X = np.vstack((X, model_feats * self.model_skew["__all__"]))
            '''
        '''   
        if X is None:
            # stack transformed features
            for model_arch, layers in layer_transform.items():
                arch_feats = None
                for layer, layer_attributes in tqdm(layers.items()):
                    layer_feats = layer_transform[model_arch][layer].pop('ICA_feat')
                    if arch_feats is None:
                        arch_feats = layer_feats
                        continue
                    # horizontal stack each layer
                    arch_feats = np.hstack((arch_feats, layer_feats * self.model_skew["__all__"]))
                # vertical stack samples from each architecture
                if X is None:
                    X = arch_feats
                    continue
                X = np.vstack((X, arch_feats))
                # delete ICA features before storage
                # store layer_transform
            with open(self.learned_parameters_dirpath + 'layer_transform.bin', "wb") as fp:
                pickle.dump(layer_transform, fp)
            del layer_transform
        '''

        print("Training detector model...")
        # model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        '''
        clf = hp.pchoice('my_name',
                         [(0.2, gradient_boosting_regressor('my_name.gradient_boosting_regressor')),
                          (0.2, linear_regression('my_name.linear_regression')),
                          (0.2, elastic_net('my_name.elastic_net')),
                          (0.2, logistic_regression('my_name.logistic_regression')),
                          (0.2, xgboost_regression('my_name.xgboost_regression')),
                          (0.2, random_forest_regressor('my_name.random_forest_regressor'))
                          ]
                         )
        
        clf = hp.pchoice('my_name',
                         [(1.0, k_neighbors_classifier('my_name.random_forest_classifier'))]
                         )
        
        clf = hp.pchoice('my_name',
                         [(0.3, random_forest_classifier('my_name.random_forest_classifier')),
                          (0.3, gradient_boosting_classifier('my_name.gradient_boosting_classifier')),
                          (0.4, sgd_classifier('my_name.sgd_classifier'))
                          #(0.2, svc('my_name.svc')),
                          #(0.4, xgboost_classification('my_name.xgboost_classification'))
                          ]
                         )
        model = HyperoptEstimator(classifier=clf, n_jobs=16, max_evals=100, preprocessing=[])
        model.fit(X, y)
        print(model.score(X, y))
        print(model.best_model())

        logger.info("Saving model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)
        '''
        self.write_metaparameters()

    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """

        # Setup scaler
        scaler = StandardScaler()

        scale_params = np.load(self.scale_parameters_filepath)

        scaler.mean_ = scale_params[0]
        scaler.scale_ = scale_params[1]

        # Inference on models
        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
                feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()

                pred = torch.argmax(model(feature_vector).detach()).item()

                ground_tuth_filepath = examples_dir_entry.path + ".json"

                with open(ground_tuth_filepath, 'r') as ground_truth_file:
                    ground_truth =  ground_truth_file.readline()

                print("Model: {}, Ground Truth: {}, Prediction: {}".format(examples_dir_entry.name, ground_truth, str(pred)))

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        start_time = time.time()

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)
        '''
        # List all available model and limit to the number provided
        model_path_list = sorted(
            [
                join(round_training_dataset_dirpath, 'models', model)
                for model in listdir(join(round_training_dataset_dirpath, 'models'))
            ]
        )
        logger.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, _ = load_models_dirpath(model_path_list)
        logger.info("Loaded models. Flattenning...")
        '''
        with open(self.models_padding_dict_filepath, "rb") as fp:
            models_padding_dict = pickle.load(fp)
        '''
        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        # Flatten model
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        del model_repr
        logger.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)
        '''
        # List all available test model and limit to the number provided
        test_model_path_list = sorted(
            [
                join(round_training_dataset_dirpath, 'test_models', model)
                for model in listdir(join(round_training_dataset_dirpath, 'test_models'))
            ]
        )

        with open(self.learned_parameters_dirpath + 'layer_transform.bin', "rb") as fp:
            layer_transform = pickle.load(fp)

        results = []
        with open(self.model_filepath, "rb") as fp:
            detector_model = pickle.load(fp)

        print(f"Running inference on %d models...", len(test_model_path_list))
        for test_model in tqdm(test_model_path_list):
            test_model_filepath = test_model + '/model.pt'
            model, model_repr, model_class = load_model(test_model_filepath)
            model_repr = pad_model(model_repr, model_class, models_padding_dict)
            flat_model = flatten_model(model_repr, model_layer_map[model_class])

            # Inferences on examples to demonstrate how it is done for a round
            # This is not needed for the random forest classifier
            # self.inference_on_example_data(model, examples_dirpath)

            X = (
                use_feature_reduction_algorithm_pca_ica(layer_transform[model_class], flat_model)
                * self.model_skew["__all__"]
            )

            probability = str(detector_model.predict(X)[0])

            # with open(result_filepath, "w") as fp:
            #    fp.write(probability)

            ground_truth = load_ground_truth(test_model)

            # logger.info("Trojan probability: %s", probability)
            results.append([test_model[-11:], probability, ground_truth])

        # log the results
        run_time = str(time.time() - start_time)
        test_count = len(test_model_path_list)
        st_str = datetime.fromtimestamp(start_time).strftime('%Y%m%d_%H:%M:%S')
        fn = 'results_' + st_str + '.csv'
        dm_str = str(detector_model)

        # calculate metrics
        results = np.asarray(results)
        y_true = results[:, 2].astype(int)
        y_pred = results[:, 1].astype(float).round().astype(int)  # convert probability to class
        target_names = ['clean', 'trojan']
        c_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True) # save dictionary of report
        print(classification_report(y_true, y_pred, target_names=target_names))
        test_log = [run_time, test_count, dm_str, fn, c_report]

        np.savetxt('./results/results_' + st_str + '.csv', results, delimiter=", ", fmt='% s')
        np.savetxt('./results/result_info' + st_str + '.csv', test_log, delimiter=", ", fmt='% s')
