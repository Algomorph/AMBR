#  ================================================================
#  Created by Gregory Kramida on 7/13/16.
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
import caffe


if __name__ == "__main__":
    caffe.set_mode_cpu()
    model_file = '/media/algomorph/Data/AMBR_data/ml_models/VGG_ILSVRC_16_layers_deploy.prototxt'
    pretrained_file = '/media/algomorph/Data/AMBR_data/ml_models/VGG_ILSVRC_16_layers.caffemodel'
    net = caffe.Net(model_file, pretrained_file, caffe.TEST)