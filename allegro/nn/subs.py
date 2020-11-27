# subs.py

import numpy as np
import torch as T
import time
import sys


version_config = {
    "0.1.0": {
        "X_cols": range(0,11),
        "Y_cols": [11],
        "X_size": 10000,
        "Y_size": 100,
        "NN_set": ["SubsNN"]
    },
    "0.1.1": {
        "X_cols": [0, 1, 2, 3, 4, 6, 9],
        "Y_cols": [11],
        "X_size": 10000,
        "Y_size": 100,
        "NN_set": ["SubsNN2"]
    }
}


class SubsDS(T.utils.data.Dataset):
    def __init__(self, file, rows=None, version="0.1.1"):
        """Subscriptions Dataset Class

        Load subscriptions data, divided in input and guess data
        The loaded dataset must be already normalized
        subscription_id|steps|hour_day|hour_tz|week_day|month_day|app_type|price|status|autorenewal|plan|period|duration_min|user_cancelled|
        ---------------|-----|--------|-------|--------|---------|--------|-----|------|-----------|----|------|------------|--------------|
                 724154|    3|       6|      1|       3|        4|       1| 2.00|     1|          1|   2|     0|           5|             1|
                 687518|    5|      15|      2|       1|       28|       1| 7.88|     0|          1|   2|     0|       61922|             7|


        0.5                 0.391304347826087   0.5 0.8571428571428571  0.2258064516129032  1   0.02022653721682848 1 1 0.3333333333333333  0   0.1428571428571428
        0.6666666666666666  0.9130434782608695  1   0.5714285714285714  0.5483870967741935  1   0.03923948220064725 1 1 1                   0   1
        0.5                 0.6086956521739131  0.5 0.1428571428571428  0.7419354838709677  1   0.0796925566343042  1 1 0.3333333333333333  0   0.1428571428571428

        Args:
                    src_file ([type]): [description]
                    num_rows ([type], optional): [description]. Defaults to None.
        """
        tmp_x = np.loadtxt(
            file,
            max_rows=rows,
            usecols=version_config[version]['X_cols'],
            delimiter=",",
            skiprows=0,
            dtype=np.float32)
        
        tmp_y = np.loadtxt(
            file,
            max_rows=rows,
            usecols=version_config[version]['Y_cols'],
            delimiter=",",
            skiprows=0,
            dtype=np.float32).reshape(-1, 1)

        self.x_data = T.tensor(tmp_x, dtype=T.float32)
        self.y_data = T.tensor(tmp_y, dtype=T.float32)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx]
        incom = self.y_data[idx]
        sample = {'subs': preds, 'duration': incom}
        return sample
