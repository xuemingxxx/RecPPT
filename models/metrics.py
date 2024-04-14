from abc import ABCMeta, abstractmethod

import numpy as np


class Metrics(metaclass=ABCMeta):

    @abstractmethod
    def __str__(self):
        """ return display name """
        pass


class Accuracy(Metrics):

    def __str__(self):
        return "acc"

    def __call__(self, output: list, label: list) -> float:
        """get accuracy
        Args:
            output : model prediction, dim: [total data size]
            label : label, dim: [total data size]
        Returns:
            float: accuracy
        """
        total = len(output)
        label_array = np.array(label)
        output_array = np.array(output)

        assert len(label_array) == len(output_array)
        match = np.sum(label_array == output_array)
        return match / total


class nDCG(Metrics):

    def __str__(self):
        return "nDCG"

    def __call__(self, output: list, label: list) -> float:
        """get normalized Discounted Cumulative Gain

        Args:
            output : model prediction, dim: [total data size, top k]
            label : label, dim: [total data size]
        Returns:
            float: ndcg
        """
        output = np.array(output)
        label = np.array(label).reshape(-1, 1)
        assert len(output) == len(label)
        
        hits = output == label
        k = np.array(output).shape[-1]
        dcg_weight = 1 / np.log2(np.arange(2, k + 2))

        idcg = 1.
        dcg = np.sum(hits.astype(float) * dcg_weight, axis=-1, keepdims=True)
        return np.mean(dcg / idcg)


class RecallAtK(Metrics):

    def __str__(self):
        return "recallAtK"

    def __call__(self, output: list, label: list) -> float:
        """get recall at k
        Args:
            output : model prediction, dim: [total data size, top k]
            label : label, dim: [total data size]
        Returns:
            float: recall@k
        """
        n = len(label)
        hits = np.array(output) == np.array(label).reshape(-1, 1)
        n_hits = np.sum(hits)
        return n_hits / n


class PrecisionAtK(Metrics):

    def __str__(self):
        return "precisionAtK"

    def __call__(self, output: list, label: list) -> float:
        """get precision at k
        Args:
            output : model prediction, dim: [total data size, top k]
            label : label, dim: [total data size]
        Returns:
            float: precision@k
        """
        k = np.array(output).shape[-1]
        hits = np.array(output) == np.array(label).reshape(-1, 1)
        precision = np.sum(hits, axis=-1) / k
        return np.mean(precision)
