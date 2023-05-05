import importlib

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA, FastICA, KernelPCA


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
        #assert out_f > 0
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
        for (layers, output) in tqdm(layers_output.items()):
            layer_transform[model_arch][layers] = {}
            layer_transform[model_arch][layers]['ICA'] = init_feature_reduction(output)
            s = np.stack([model[layers] for model in models])
            pca = PCA(whiten=True)
            layer_transform[model_arch][layers]['PCA'] = pca.fit(s)  # store PCA fit
            s = pca.transform(s)
            layer_transform[model_arch][layers]['ICA'].fit(s)  # store ICA fit
            layer_transform[model_arch][layers]['ICA_feat'] = layer_transform[model_arch][layers]['ICA'].transform(s)  # store the transformed features

            # remove layer
            # for model in models:
            #     del model[layers]
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

def fit_feature_reduction_algorithm_final_layer(model_dict, weight_table_params, input_features):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)

    for (model_arch, models) in model_dict.items():
        layers_output = feature_reduction(models[0], weight_table, input_features)
        layer_transform[model_arch] = {}

        n = len(layers_output)
        i = 0
        for (layers, output) in tqdm(layers_output.items()):
            i += 1
            if i == n:
                layer_transform[model_arch][layers] = {}
                layer_transform[model_arch][layers]['ICA'] = init_feature_reduction(input_features)
                s = np.stack([model[layers] for model in models])
                pca = PCA(n_components=30,whiten=True)
                layer_transform[model_arch][layers]['PCA'] = pca.fit(s)  # store PCA fit
                s = pca.transform(s)
                layer_transform[model_arch][layers]['ICA'].fit(s)  # store ICA fit
                layer_transform[model_arch][layers]['ICA_feat'] = layer_transform[model_arch][layers]['ICA'].transform(s)  # store the transformed features

            # remove layer
            for model in models:
                del model[layers]
    return layer_transform

def fit_feature_reduction_algorithm_pca_model_ica(model_dict, pca_component, ica_component, weight_params, input_features, kernel):
    layer_transform = {}
    weight_table = init_weight_table(**weight_table_params)
    model_transform = None
    for (model_arch, models) in model_dict.items():
        # layers_output = feature_reduction(models[0], weight_table, input_features)
        layer_transform[model_arch] = {}
        for (layers, output) in models.items():
            layer_transform[model_arch][layers] = {}
            s = np.stack([model[layers] for model in models])
            pca = KernelPCA(n_components=pca_component, kernel=kernel)
            # pca = KernelPCA(n_components=pca_component, whiten=True)
            layer_transform[model_arch][layers]['PCA'] = pca.fit(s)  # store PCA fit
            layer_transform[model_arch][layers]['PCA_feat'] = pca.transform(s)  # store the PCA transformed features

            # remove layer
            #for model in models:
            #    del model[layers]
        # perform ica at model level
        s = np.hstack([layer_transform[model_arch][l]['PCA_feat'] for l in layer_transform[model_arch]])
        for l in layer_transform[model_arch]:
            del layer_transform[model_arch][l]['PCA_feat']
        ica = FastICA(n_components=ica_component)
        s = ica.fit_transform(s)
        if model_transform is None:
            model_transform = s
            continue
        model_transform = np.vstack((model_transform, s))
    return model_transform
