import importlib

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA, FastICA, KernelPCA
import pickle


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
        # assert out_f > 0
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


def fit_feature_reduction_algorithm_pca_model_ica(model_dict, layer_pca_component, arch_pca_component, dataset_pca_component, ica_component, kernel):
    arch_transform = None

    # iterate through each arch
    for (model_arch, models) in model_dict.items():
        layer_transform = None

        # feature reduction of each layer in this arch
        pca = KernelPCA(n_components=layer_pca_component, kernel=kernel)
        for layers in models[0].keys():
            s = np.stack([model[layers] for model in models])  # s = this layer from each model
            # layer_transform[model_arch][layers]['PCA'] = pca.fit(s)  # store PCA fit - commented out because currently not storing
            s = pca.fit_transform(s)  # store the PCA transformed features
            if layer_transform is None:
                layer_transform = s
                continue
            layer_transform = np.hstack((layer_transform, s))

        # perform pca at model arch level
        # s is stack pca features of each model in this arch
        pca = KernelPCA(n_components=arch_pca_component, kernel=kernel)
        layer_transform = pca.fit_transform(layer_transform)
        if arch_transform is None:
            arch_transform = layer_transform
            continue
        arch_transform = np.vstack((arch_transform, layer_transform))

    # pca for the entire dataset
    pca = KernelPCA(n_components=dataset_pca_component, kernel=kernel)
    data_transform = pca.fit_transform(arch_transform)

    # ica for the entire dataset
    ica = FastICA(n_components=ica_component)
    data_transform = ica.fit_transform(data_transform)

    return data_transform


def fit_feature_reduction_algorithm_pca_model_ica_opt(file_path, model_dict, layer_pca_components, arch_pca_components, dataset_pca_components, ica_components, kernels):
    arch_transform = None
    file_count = 0
    # try each layer pca
    for kernel in tqdm(kernels, desc='kernels'):
        print('Generating: ', kernel)
        #try:
        for layer_pca_component in layer_pca_components:
            # iterate through each arch
            try:
                for (model_arch, models) in model_dict.items():
                    layer_transform = None

                    # collect layer transform for this arch
                    pca = KernelPCA(n_components=layer_pca_component, kernel=kernel)
                    for layers in models[0].keys():
                        s = np.stack([model[layers] for model in models])  # s = this layer from each model
                        # layer_transform[model_arch][layers]['PCA'] = pca.fit(s)  # store PCA fit - commented out because currently not storing
                        s = pca.fit_transform(s)  # store the PCA transformed features
                        if layer_transform is None:
                            layer_transform = s
                            continue
                        layer_transform = np.hstack((layer_transform, s))
            except Exception as es1:
                print(f"Failed on: {model_arch}, {kernel}, {layer_pca_component}")
                continue

            # using this layer_transform, try each arch pca_component
            for arch_pca_component in arch_pca_components:
                arch_transform = None
                # perform pca at arch level
                pca = KernelPCA(n_components=arch_pca_component, kernel=kernel)
                all_layer_transform = pca.fit_transform(layer_transform)
                if arch_transform is None:
                    arch_transform = all_layer_transform
                    continue
                arch_transform = np.vstack((arch_transform, layer_transform))

                for dataset_pca_component in dataset_pca_components:
                    # pca for the entire dataset
                    pca = KernelPCA(n_components=dataset_pca_component, kernel=kernel)
                    data_transform = pca.fit_transform(arch_transform)

                    for ica_component in ica_components:
                        # ica for the entire dataset
                        ica = FastICA(n_components=ica_component)
                        ica_transform = ica.fit_transform(data_transform)

                        with open(file_path + f'2023-05-06_train_num_lpca_{layer_pca_component}_apca_{arch_pca_component}_dpca_{dataset_pca_component}_ica_{ica_component}_kernel_{kernel}.pkl',"wb") as fp:
                            pickle.dump(ica_transform, fp)
                        file_count += 1
        # except Exception as er1:
        #    print(er1)  # print(f"Failed - l_PCA: {layer_pca_component}, a_PCA: {arch_pca_component}, d_PCA: {dataset_pca_component},ICA: {ica_component}, kernel:{kernel}\n{er1}")

    return file_count
