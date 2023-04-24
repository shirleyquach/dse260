import importlib

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA


def feature_reduction(model, weight_table, max_features):
    outputs = {}
    tf = int(max_features / len(model))  # min features per layer
    add_feat = max_features - tf * len(model)  # additional features to add
    sm = sum([l.shape[0] for l in model.values()])  # sum of the shape
    for (layer, weights) in model.items():
        # wt_i = np.round(weights.shape[0] / sm * 100).astype(np.int32)  # based on what percentage of the weights this layer has
        # out_f = int(weight_table[wt_i] * tf)
        wt_percent = (weights.shape[0] / sm)  # percentage of weights in this layer
        out_f = int(np.round(wt_percent * add_feat)) + tf  # every layer min feature plus additional
        if layer == list(model.keys())[-1]:
            out_f = max_features - sum(outputs.values())
        assert out_f > 0
        outputs[layer] = out_f
    return outputs

def init_feature_reduction(output_feats):
    fr_algo = "sklearn.decomposition.FastICA"
    fr_algo_mod = ".".join(fr_algo.split(".")[:-1])
    fr_algo_class = fr_algo.split(".")[-1]
    mod = importlib.import_module(fr_algo_mod)
    fr_class = getattr(mod, fr_algo_class)
    return fr_class(n_components=output_feats)


def init_weight_table(random_seed, mean, std, scaler):
    rnd = np.random.RandomState(seed=random_seed)
    return np.sort(rnd.normal(mean, std, 100)) * scaler


def fit_feature_reduction_algorithm(model_dict, weight_table_params, input_features):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)

    for (model_arch, models) in model_dict.items():
        layers_output = feature_reduction(models[0], weight_table, input_features)
        layer_transform[model_arch] = {}
        for (layers, output) in tqdm(layers_output.items()):
            layer_transform[model_arch][layers] = init_feature_reduction(output)
            s = np.stack([model[layers] for model in models])
            layer_transform[model_arch][layers].fit(s)

    return layer_transform

def fit_feature_reduction_algorithm_pca_ica(model_dict, weight_table_params, input_features):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)

    for (model_arch, models) in model_dict.items():
        layers_output = feature_reduction(models[0], weight_table, input_features)

        layer_transform[model_arch] = {}
        arch = layer_transform[model_arch]

        for (layers, output) in tqdm(layers_output.items()):
            arch[layers] = {}
            layer_dict = arch[layers]

            layer_dict['ICA'] = init_feature_reduction(output)
            s = np.stack([model[layers] for model in models])
            pca = PCA(whiten=True)
            layer_dict['PCA'] = pca.fit(s)  # store PCA fit
            s = pca.transform(s)
            layer_dict['ICA'].fit(s)  # store ICA fit
            layer_dict['ICA_feat'] = layer_dict['ICA'].transform(s)  # store the transformed features

            # remove layer
            for model in models:
                del model[layers]
    breakpoint()
    return layer_transform


def use_feature_reduction_algorithm(layer_transform, model):
    out_model = np.array([[]])
    for (layer, weights) in model.items():
        out_model = np.hstack((out_model, layer_transform[layer].transform([weights])))
    return out_model


def use_feature_reduction_algorithm_pca_ica(layer_transform, model):
    out_model = np.array([[]])
    for (layer, weights) in model.items():
        pca_weights = layer_transform[layer]['PCA'].transform([weights])
        out_model = np.hstack((out_model, layer_transform[layer]['ICA'].transform(pca_weights)))
    return out_model
