import pandas as pd


class StatFeatures:

    def __init__(self, data: pd.Series, count_intervals: int = 50):
        self.__data = data
        self.__count_intervals = count_intervals
        self.__min_rr = self.__data.min()
        self.__max_rr = self.__data.max()
        self.__interval = (self.__max_rr - self.__min_rr) / self.__count_intervals

        self.__mode = self.__calc_point_mode()
        self.__mode_amplitude = self.__calc_mode_amplitude()

    def __calc_point_mode(self):
        distribution = list()
        for num_interval in range(self.__count_intervals):
            min_tresh = num_interval * self.__interval + self.__min_rr
            max_tresh = (num_interval + 1) * self.__interval + self.__min_rr

            count = len(self.__data.loc[(self.__data > min_tresh) & (self.__data < max_tresh)])
            distribution.append(count)

        distribution = pd.Series(distribution)
        mode_index_interval = distribution.argmax()

        min_tresh = mode_index_interval * self.__interval + self.__min_rr
        max_tresh = (mode_index_interval + 1) * self.__interval + self.__min_rr
        current_interval_mode = distribution[mode_index_interval]
        self.__start_mode_interval = min_tresh
        if len(distribution.loc[distribution == 0]) == len(distribution):
            return 0

        count_main_interval = len(self.__data.loc[(self.__data > min_tresh) & (self.__data < max_tresh)])
        if mode_index_interval == 0:
            next_max_tresh = (mode_index_interval + 2) * self.__interval + self.__min_rr
            count_next_interval = len(self.__data.loc[(self.__data > max_tresh) & (self.__data < next_max_tresh)])
            next_interval_mode = distribution[mode_index_interval + 1]
            importance_main_interval = count_main_interval / (count_main_interval + count_next_interval)
            importance_next_interval = count_next_interval / (count_main_interval + count_next_interval)
            point_mode = min_tresh + (importance_main_interval * current_interval_mode + importance_next_interval * next_interval_mode)
            print(f'Next: {point_mode}')
            return point_mode

        if mode_index_interval == len(distribution) - 1:
            pred_min_tresh = (mode_index_interval - 1) * self.__interval + self.__min_rr
            count_pred_interval = len(self.__data.loc[(self.__data > pred_min_tresh) & (self.__data < min_tresh)])
            pred_interval_mode = distribution[mode_index_interval - 1]
            importance_main_interval = count_main_interval / (count_main_interval + count_pred_interval)
            importance_next_interval = count_pred_interval / (count_main_interval + count_pred_interval)
            point_mode = min_tresh + (importance_main_interval * current_interval_mode + importance_next_interval * pred_interval_mode)
            print(f'Pred: {point_mode}')
            return point_mode

        pred_interval_mode = distribution[mode_index_interval - 1]
        next_interval_mode = distribution[mode_index_interval + 1]

        point_mode = min_tresh + self.__interval * (current_interval_mode - pred_interval_mode) \
                     / ((current_interval_mode - pred_interval_mode) + (current_interval_mode + next_interval_mode))

        print(f'Main: {point_mode}')
        return point_mode

    def __calc_mode_amplitude(self):
        rr_intervals = self.__data.loc[(self.__data > self.__start_mode_interval) &
                                       (self.__data < (self.__start_mode_interval + self.__interval))]

        return len(rr_intervals) / len(self.__data)

    def get_tension_index(self):
        if self.__mode == 0:
            return 0

        dRR = (self.__max_rr - self.__min_rr) / 1_000
        mode_s = self.__mode / 1000
        tens_index = 100 * self.__mode_amplitude / (2 * dRR * mode_s)

        return tens_index

    def get_mode(self):
        return self.__mode

    def get_mode_amplitude(self):
        return self.__mode_amplitude
