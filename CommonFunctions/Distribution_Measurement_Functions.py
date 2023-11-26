# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to measurement of 
distribution distances
jsb92, 2023/08/01
"""

import numpy as np

class dist_measure():
    def __init__(self):
        self = self
        return

    def bhattacharyya_distance(self, x, distribution1, distribution2):
        """ Estimate Bhattacharyya Distance (between General Distributions)

        Args:
            distribution1: a sample distribution 1
            distribution2: a sample distribution 2

        Returns:
            Bhattacharyya distance
        """
        if len(distribution1) != len(distribution2):
            raise Exception("Distribution 1 does not equal length of Distribution 2")
        DB = np.trapz(np.sqrt(distribution1*distribution2), x=x)
        BD = -np.log(DB)
        if BD < 0:
            BD = 0
        return BD