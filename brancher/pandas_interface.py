import pandas as pd
import numpy as np
from brancher.utilities import is_tensor
from brancher.utilities import map_iterable
from collections.abc import Iterable


def pandas_dict2list(dic):
    indices, values = zip(*dic.items())
    sorted_indices = np.argsort(indices)
    return np.array(values)[sorted_indices]


def pandas_frame2dict(dataframe):
    if isinstance(dataframe, pd.core.frame.DataFrame):
        return {key: pandas_dict2list(val) for key, val in dataframe.to_dict().items()}
    elif isinstance(dataframe, dict):
        return dataframe
    else:
        raise ValueError("The input should be either a dictionary or a Pandas dataframe")


def pandas_frame2value(dataframe, index):
    if isinstance(dataframe, pd.core.frame.DataFrame):
        values = np.array([np.ndarray.tolist(x) if isinstance(x, np.ndarray) else x
                           for x in dataframe[index].values]) #TODO: Ugly, this should be improved
        return values
    else:
        return dataframe


def reformat_value(value, index):
    if is_tensor(value):
        if np.prod(value[index, :].shape) == 1:
            return float(value[index, :].cpu().detach().numpy())
        elif value.shape[1] == 1:
            return value[index, :].cpu().detach().numpy()[0, :]
        else:
            return value.cpu().detach().numpy()
    elif isinstance(value, Iterable):
        return map_iterable(lambda x: reformat_value(x, index), value)
    else:
        return value


def reformat_sample_to_pandas(sample):
    number_samples = list(sample.values())[0].shape[0]
    data = [[reformat_value(value, index)
             for index in range(number_samples)]
            for variable, value in sample.items()]
    index = [key.name for key in sample.keys()]
    column = range(number_samples)
    frame = pd.DataFrame(data, index=index, columns=column).transpose()
    return frame


def reformat_model_summary(summary_data, var_names, feature_list):
    return pd.DataFrame(summary_data, index=var_names, columns=feature_list).transpose()
