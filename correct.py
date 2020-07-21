import torch
import numpy as np
import math


class LabelCorrecter():
    """
    ノイズの入ったラベルを上書きするためのモジュール．
    n : 各クラスごとの全体のデータ数．
    m : 各クラスごとのサンプリングするデータ数．
    p : 各クラスごとのプロトタイプ数．
    c : クラス数．
    """
    def __init__(self, m : int, p):
        """
        Args
            m : similarity matrixを作るためのサンプリングデータの数．m << n．
            p : 各クラスごとに抽出するプロトタイプの数．p << m．
        """
        self.m = m
        self.p = p

    def _get_cosine_similarity(self, x : np.array, y : np.array) -> float:
        """
        2つのベクトルのcos類似度を返す．
        Args
            x : cos類似度を得るためのベクトル．(d,)
            y : cos類似度を得るためのベクトル．(d,)
        Returns
            xとyのcos類似度．
        """
        return np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y))

    def _get_cosine_similarity_for_matrix(self, x, y, axis):
        """
        2つのベクトル群のcos類似度を返す．
        Args
            x : cos類似度を得るためのベクトルの集合．size : (n_x, d,)
            y : cos類似度を得るためのsize : (c, n_y, d,)
        Returns
            xとyのcos類似度．
            size : (n_x, c, n_y,)
        """
        y = np.reshape(y, (*(y.shape), 1))
        x_dim = x.ndim
        y_dim = y.ndim
        return np.squeeze(np.dot(x, y) / np.dot(np.linalg.norm(x, axis = x_dim - 1, keepdims = True), np.linalg.norm(y, axis = y_dim - 2, keepdims = True)), -1)

    def _get_similarity_matrix(self, features : np.array) -> np.array:
        """
        sample同士のsimilarity matrixを得る．
        Args
            features : m個のデータの特徴量行列．size : (m, d,)
        Returns
            cosine similarity行列．size : (m, m,)
            [i][j]にfeatures[i]とfeatures[j]のsimilarityを保持する．
        """
        r = range(len(features))
        # norms = np.sqrt(np.sum(features**2, axis = 1))
        return np.array([(self._get_cosine_similarity(features[i], features[j]) for i in r) for j in r])

    def _get_rho(self, similarity_matrix : np.array) -> np.array:
        """
        各sampleについてどれほどcosine similarity似た特徴量を持つsampleが多いかを示すrhoを算出する．
        rhoが大きいほど似た特徴量のsampleが多い．
        すなわちrhoの高さは，大きなクラスタで代表となりうることを示すと同時に，そのsampleの唯一性の低さも示す．

        Args
            similarity_matrix : m*mのcosine similarityの行列．size : (m, m,)
            [i][j]にi番目のsampleとj番目のsampleのsimilarityが保持されている．
        Returns
            rhoのarray．size : (m,)
            [i]にi番目のsampleのrhoが入っている．
        """
        similarity_array = similarity_matrix.flatten()
        threshold = np.sort(similarity_array)[int(len(similarity_array)*0.6)]
        
        return np.sum(np.sign(similarity_matrix - threshold), axis = 1)

    def _get_eta(self, similarity_matrix : np.array, rho_array : np.array) -> np.array:
        """
        各sampleの重要度を示すetaを算出する．
        etaは．rhoが最大のsampleに対しては，自身と最もsimilarityが低いsampleとのsimilarityを返す．
        それ以外のsampleに対しては，自分よりrhoが高いsamplesの中からsimilarityが最も高いsampleとのsimilarityを返す．
        etaが小さいほど，自分を含むsample群のクラスタで代表的な存在になっているといえる．

        Args
            similarity_matrix : m*mのcosine similarityの行列．size : (m, m,)
            [i][j]にi番目のsampleとj番目のsampleのsimilarityが保持されている．

            rho_array : rhoのarray．size : (m,)
            [i]にi番目のsampleのrhoを保持する．

        Returns
            etaのarray．size : (m,)
            [i]にi番目のsampleのetaを保持する．
        """
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

    def decide_features_prototype_for_class(self, features : np.array) -> np.array:
        """
        あるクラスの特徴量からプロトタイプを決める．
        Args
            features : クラス内の全sampleの特徴量ベクトル．size : (n, d,)
        Returns
            そのクラスのp個のプロトタイプの特徴量．
        """
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

        return features_for_sort[0][::-1][:self.p]

    def save_prototype(self, features_all : np.array) -> np.array:
        """
        すべてのsampleの特徴量から各クラスのプロトタイプを決める．
        プロトタイプはクラス変数として保持する．

        Args
            features_all : すべてのクラスの特徴量．(c, n, d,)
        Returns
            None
        """
        self.prototype = np.array([decide_features_prototype_for_class(f) for f in features_all])

    def get_modified_labels(self, features_input : np.array) -> np.array:
        """
        プロトタイプを既に保持している状態で全てのデータのラベルを上書きする．
        Args
            features_input : すべての入力データの特徴量．size : (n, d,)
        Returns
            すべてのデータのラベル．size : (n,)
        """
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