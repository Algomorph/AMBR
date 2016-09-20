#  ================================================================
#  Created by Gregory Kramida on 9/20/16.
#  Copyright (c) 2016 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

from lstm.network_construction import build_network


class Model(object):
    """
    Class that represents an LSTM network padded by embedding & classification layers.
    """

    def __init__(self, parameters, use_dropout=True, weighted_cost=False, random_seed=2016, decay_constant=0.0):
        self.parameters = parameters
        self.noise_flag, self.x, self.masks, self.cost_weights, self.y, \
        self.compute_sample_classification_probability, self.classify_sequence, self.cost, \
        self.classify_timestep, self.get_network_state = \
            build_network(parameters, use_dropout, weighted_cost, random_seed)

