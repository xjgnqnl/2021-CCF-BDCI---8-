#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import random
from ..registry import PIPELINES
from .bone_pairs import bone_25
"""pipeline ops for Activity Net.
"""


@PIPELINES.register()
class AutoPadding(object):
    """
    Sample or Padding frame skeleton feature.
    Args:
        window_size: int, temporal size of skeleton feature.
        random_pad: bool, whether do random padding when frame length < window size. Default: False.
    """
    def __init__(self, window_size, random_pad=False):
        self.window_size = window_size
        self.random_pad = random_pad

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        return T

    
    def get_frame_list(self, data):#clear data function
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[:, i, :, :])
            if tmp > 0:
                T = i + 1
                break
        
        frame_list = []
        for i in range(T):
            frame_data = data[: , i , : , 0]#[C , V]
            y_data = frame_data[1 , :]#[V] 25个点的y坐标
            score_data = frame_data[2 , :]#[V] 25个点的置信分数
            score_data_lower = score_data[[9 , 10 , 11 , 12 , 13 , 14 , 19 , 22]]



            number_t = np.sum(y_data < 1e-3)#total number y+
            number_total = np.sum(y_data < -(1e-3))#total number y-
            number_boundary = np.sum(y_data < -0.15)#total number y-boundary

            score_t = np.sum(score_data_lower > -1)#total number lower
            score_boundary = np.sum(score_data_lower < 0.35)#bad number

            ratio_score = float(score_boundary) / score_t

            if number_t == 25:
                if number_total == number_boundary:#全部位于-0.15以下
                    continue
                elif ratio_score >= 0.4:
                    continue
            frame_list.append(i)
        return frame_list

    

    def __call__(self, results):
        data = results['data']

        C, T, V, M = data.shape
        T = self.get_frame_num(data)

        frame_list = self.get_frame_list(data)
        T_list = [i for i in range(T)]
        noise_list = list(set(T_list) - set(frame_list))
        data[: , noise_list , : , :] = 0
        '''
        if len(frame_noise) >0:
            data_clear = data[: , frame_noise , : , :]
        else:
            data_clear = data[: , :T , : , :]
        '''
        data_clear = data[: , :T , : , :]
        #added zero-8
        for i in range(T):
            index = []
            frame_data = data_clear[: , i , : , 0]#[c , v]
            frame_sum = np.sum(frame_data , axis = 0)#[v]
            for j in range(V):
                if frame_sum[j] > 1e-3 or frame_sum[j] < -(1e-3) :
                    index.append(j)
            if len(index) > 0:
                data_clear[: , i , index , :] = data_clear[: , i , index , :] - data_clear[:, i, 8:9, :]
        #added zero-8
        c , t , v , m = data_clear.shape
        if t == self.window_size:
            data_pad = data_clear[:, :self.window_size, :, :]
    
        #elif T < self.window_size:
        #    begin = random.randint(0, self.window_size -
        #                           T) if self.random_pad else 0
        #    data_pad = np.zeros((C, self.window_size, V, M))
        #    data_pad[:, begin:begin + T, :, :] = data[:, :T, :, :]
        elif t < self.window_size:
            begin = random.randint(0, self.window_size -
                                   t) if self.random_pad else 0
            data_pad = np.zeros((c, self.window_size, v, m))
            data_pad[:, begin:begin + t, :, :] = data_clear[:, :t, :, :]
            #added ---- pad the null frames with the previous frames
            '''
            rest = self.window_size - t
            num = int(np.ceil(rest / t))
            pad = np.concatenate([data_clear[:, :t, :, :] for _ in range(num)] , 1)[:,:rest]
            data_pad[:, (begin + t): , : , :] = pad
            '''
            #added ---- pad the null frames with the previous frames
        else:
            if self.random_pad:
                index = np.random.choice(t, self.window_size,
                                         replace=False).astype('int64')
            else:
                #index = np.linspace(0, t, self.window_size).astype("int64")
                index = np.linspace(0, t-1, self.window_size).astype("int64")
            data_pad = data_clear[:, index, :, :]

        results['data'] = data_pad
        return results

    


@PIPELINES.register()
class SkeletonNorm(object):
    """
    Normalize skeleton feature.
    Args:
        aixs: dimensions of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default: 2.
    """
    def __init__(self, model , axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze
        self.model = model
        print(self.model)

    def __call__(self, results):
        data = results['data']

        # Centralization
        #data = data - data[:, :, 8:9, :]
        data_bone = np.zeros_like(data)#added
        data_motion = np.zeros_like(data)#added
        data_motion[:, :-1] = data[:, 1:] - data[:, :-1]
        data_motion[:, -1] = 0
        for v1, v2 in bone_25:#added
            data_bone[:, :, v1] = data[:, :, v1] - data[:, :, v2]#added
        data_bone = data_bone[:self.axis, :, :, :]  #added
        data = data[:self.axis, :, :, :]  # get (x,y) from (x,y, acc)
        data_motion = data_motion[:self.axis, :, :, :]
        C, T, V, M = data.shape

        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
            data_bone = data_bone.reshape((C, T, V)) #added
            data_motion = data_motion.reshape((C, T, V)) #added
        data_concate = np.concatenate((data, data_bone , data_motion) , axis = 0)
        #results['data'] = data.astype('float32')
        #results['data'] = data_concate.astype('float32')
        if self.model == 'joint':
            results['data'] = data.astype('float32')
        elif self.model == 'bone':
            results['data'] = data_bone.astype('float32')
        elif self.model == 'concate':
            results['data'] = data_concate.astype('float32')
        elif self.model == 'joint-motion':
            results['data'] = data_motion.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class Iden(object):
    """
    Wrapper Pipeline
    """
    def __init__(self, label_expand=True):
        self.label_expand = label_expand

    def __call__(self, results):
        data = results['data']
        results['data'] = data.astype('float32')

        if 'label' in results and self.label_expand:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results
