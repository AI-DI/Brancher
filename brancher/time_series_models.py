from brancher.variables import ProbabilisticModel
from brancher.variables import Variable
from brancher.pandas_interface import pandas_frame2dict
from brancher.pandas_interface import reformat_temporal_sample_to_pandas_timeseries

from brancher.utilities import reformat_sampler_input


class TimeSeriesModel(ProbabilisticModel):

    def __init__(self, temporal_variables, time_stamps):
        assert isinstance(temporal_variables, list) and all([isinstance(var, Variable) for var in
                                                             temporal_variables]), "The input temporal_variable should be a list of Brancher variables"
        assert isinstance(temporal_variables, list) and all([isinstance(t, (int, float, str)) for t in
                                                             time_stamps]), "The input time_stamps should be a list of either floats, integers or strings"
        self.time_stamps = time_stamps
        self.temporal_variables = temporal_variables
        super().__init__(temporal_variables)

    def _get_time_stamp(self, var):
        assert var in self.temporal_variables
        var_index = self.temporal_variables.index(var)
        return self.time_stamps[var_index]

    def get_timeseries_sample(self, number_samples, input_values={}, mode="prior"):
        if mode == "prior":
            sampler = self._get_sample
        elif mode == "posterior":
            sampler = self._get_posterior_sample
        else:
            raise ValueError("the mode of the sampler should be either the 'prior' or 'posterior' string")

        reformatted_input_values = reformat_sampler_input(pandas_frame2dict(input_values),
                                                          number_samples=number_samples)
        raw_sample = sampler(number_samples, input_values=reformatted_input_values)
        temporal_raw_sample = {var: (self._get_time_stamp(var), value)
                               for var, value in raw_sample.items() if var in self.temporal_variables}
        temporal_sample = reformat_temporal_sample_to_pandas_timeseries(temporal_raw_sample)
        return temporal_sample


class LatentTimeSeriesModel(TimeSeriesModel):

    def __init__(self, temporal_variables, observation_variables, time_stamps):
        assert isinstance(observation_variables, list) and all([isinstance(var, Variable) for var in
                                                                temporal_variables]), "The input observation_variable should be a list of Brancher variables"
        self.observation_variables = observation_variables
        super().__init__(temporal_variables, time_stamps)

