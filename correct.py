import torch
import numpy as np
import math


class LabelCorrecter():
    def __init__(self, m, p):
        self.m = m
        self.p = p

    def _get_cosine_similarity(x, y):
        return np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y, axis))

    def _get_cosine_similarity_for_matrix(x, y, axis):
        y = np.reshape(y, (*(y.shape), 1))
        x_dim = x.ndim
        y_dim = y.ndim
        return np.squeeze(np.dot(x, y) / np.dot(np.linalg.norm(x, axis = x_dim - 1, keepdims = True), np.linalg.norm(y, axis = y_dim - 2, keepdims = True)), -1)

    def _get_similarity_matrix(features):
        r = range(len(features))
        # norms = np.sqrt(np.sum(features**2, axis = 1))
        return np.array([(get_cosine_similarity(features[i], features[j]) for i in r) for j in r])

    def _get_rho(similarity_matrix):
        similarity_array = similarity_matrix.flatten()
        threshold = np.sort(similarity_array)[int(len(similarity_array)*0.6)]
        
        return np.sum(np.sign(similarity_matrix - threshold), axis = 1)

    def _get_eta(similarity_matrix, rho_array):
        rho_order_idx = np.argsort(rho_array)[::-1]
        mask = np.zeros(len(rho_array), dtype = np.bool)

        eta = np.zeros(len(rho_array))

        for i in rho_order_idx:
            if rho_array[i] == np.max(rho_array):
                eta[i] = np.min(similarity_matrix[i])
            else:
                eta[i] = np.max(similarity_matrix[i][mask])

            mask[i] = True

        return eta

    def decide_features_prototype_for_class(features):
        # features : (num_samples, size_features,)
        featurs = np.random.choice(features, size = self.m, replace = False)
        # features : (m, size_features,)
        similarity_matrix = self._get_similarity_matrix(features)
        rho = _get_rho(similarity_matrix)
        eta = _get_eta(similarity_matrix, rho)

        features = features[eta < 0.95]
        rho = rho[eta < 0.95]

        features_for_sort = np.array([features, rho])
        features_for_sort = features_for_sort[:, features_for_sort[1, :].argsort()]

        return features_for_sort[0][::-1][:p]

    def save_prototype(features_all):
        # features_all : (num_class, num_samples, size_features,)
        # maybe not numpy array
        self.prototype = np.array([decide_features_prototype_for_class(f) for f in features_all])

    def get_modified_labels(features_input):
        features_prototype = self.prototype
        # input : (num_input, size_feature,)
        # prototype : (num_class, num_prototype, size_feature,)

        sims = get_cosine_similarity_for_matrix(features_input, features_prototype)

        # sims : (num_input, num_class, num_prototype,)

        sims = np.mean(sims, axis = 2)

        # sims : (num_input, num_class,)

        labels = np.argmax(sims, axis = 1)

        # labels : (num_input,)

        return labels