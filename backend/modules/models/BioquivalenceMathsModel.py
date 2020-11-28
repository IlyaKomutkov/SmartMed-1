import numpy as np

from sklearn.metrics import auc
from scipy import stats


class BioquivalenceMathsModel:

    def get_auc(x: np.array, y: np.array) -> float:
        return auc(x, y)

    def get_log_array(x: np.array) -> np.array:
        return np.log(x)

    def get_kstest(x: np.array) -> tuple:
    	return stats.kstest(x, 'norm')

    def get_shapiro(x: np.array) -> tuple:
    	return stats.shapiro(x)

    def get_f(x: np.array, y: np.array) -> tuple:
    	return stats.f_oneway(x, y)
