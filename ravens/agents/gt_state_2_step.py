# coding=utf-8
# Copyright 2022 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ground-truth state 2-step Agent."""

import time

import numpy as np
from ravens.agents.gt_state import GtState6DAgent
from ravens.agents.gt_state import GtStateAgent
from ravens.models import mdn_utils
from ravens.models.gt_state import MlpModel
from ravens.utils import utils
import tensorflow as tf


class GtState2StepAgent(GtStateAgent):
  """Agent which uses ground-truth state information -- useful as a baseline."""

  def __init__(self, name, task):
    super(GtState2StepAgent, self).__init__(name, task)

    # Set up model.
    self.pick_model = None
    self.place_model = None

    self.pick_optim = tf.keras.optimizers.Adam(learning_rate=2e-4)
    self.place_optim = tf.keras.optimizers.Adam(learning_rate=2e-4)
    
    # self.metric = tf.keras.metrics.Mean(name='metric')
    # self.val_metric = tf.keras.metrics.Mean(name='val_metric')

  def init_model(self, dataset):
    """Initialize models, including normalization parameters."""
    self.set_max_obs_vector_length(dataset)

    _, _, info = dataset.random_sample()
    obs_vector = self.info_to_gt_obs(info)

    # Setup pick model
    obs_dim = obs_vector.shape[0]
    act_dim = 3
    self.pick_model = MlpModel(
        self.batch_size, obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

    obs_train_parameters=self.get_train_parameters(dataset)
    
    self.pick_model.set_normalization_parameters(obs_train_parameters)

    # Setup pick-conditioned place model
    obs_dim = obs_vector.shape[0] + act_dim
    act_dim = 3
    self.place_model = MlpModel(
        self.batch_size, obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

    obs_train_parameters=self.get_train_parameters(dataset)

    self.place_model.set_normalization_parameters(obs_train_parameters)
    return self.get_train_step()

  def get_train_step(self):
    pick_model=self.pick_model
    place_model=self.place_model

    @tf.function
    def train_step(self, batch_obs, batch_act,
                   loss_criterion):
      with tf.GradientTape() as tape:
        prediction = pick_model(batch_obs)
        loss0 = loss_criterion(batch_act[:, 0:3], prediction)
        grad = tape.gradient(loss0, pick_model.trainable_variables)
        self.pick_optim.apply_gradients(
            zip(grad, pick_model.trainable_variables))
      with tf.GradientTape() as tape:
        # batch_obs = tf.concat((batch_obs, batch_act[:,0:3] +
        #                        tf.random.normal(shape=batch_act[:,0:3].shape,
        #                                         stddev=0.001)), axis=1)
        batch_obs = tf.concat((batch_obs, batch_act[:, 0:3]), axis=1)
        prediction = place_model(batch_obs)
        loss1 = loss_criterion(batch_act[:, 3:], prediction)
        grad = tape.gradient(loss1, place_model.trainable_variables)
        self.place_optim.apply_gradients(
            zip(grad, place_model.trainable_variables))
      return loss0 + loss1
    return train_step

  def act(self, obs, info):
    """Run inference and return best action."""
    act = {'camera_config': self.camera_config, 'primitive': None}

    # Get observations and run pick prediction
    gt_obs = self.info_to_gt_obs(info)
    pick_prediction = self.pick_model(gt_obs[None, Ellipsis])
    
    pick_prediction=self.prediction_unbatch(pick_prediction)

    # Get observations and run place prediction
    obs_with_pick = np.hstack((gt_obs, pick_prediction))

    # since the pick at train time is always 0.0,
    # the predictions are unstable if not exactly 0
    obs_with_pick[-1] = 0.0

    place_prediction = self.place_model(obs_with_pick[None, Ellipsis])
    
    place_prediction=self.prediction_unbatch(place_prediction)


    prediction = np.hstack((pick_prediction, place_prediction))

    # just go exactly to objects, from observations
    # p0_position = np.hstack((gt_obs[3:5], 0.02))
    # p0_rotation = utils.eulerXYZ_to_quatXYZW(
    #     (0, 0, -gt_obs[5]*self.theta_scale))
    # p1_position = np.hstack((gt_obs[0:2], 0.02))
    # p1_rotation = utils.eulerXYZ_to_quatXYZW(
    #     (0, 0, -gt_obs[2]*self.theta_scale))

    # just go exactly to objects, predicted
    p0_position = np.hstack((prediction[0:2], 0.02))
    p0_rotation = utils.eulerXYZ_to_quatXYZW(
        (0, 0, -prediction[2] * self.theta_scale))
    p1_position = np.hstack((prediction[3:5], 0.02))
    p1_rotation = utils.eulerXYZ_to_quatXYZW(
        (0, 0, -prediction[5] * self.theta_scale))

    # Select task-specific motion primitive.
    act['primitive'] = 'pick_place'
    if self.task == 'sweeping':
      act['primitive'] = 'sweep'
    elif self.task == 'pushing':
      act['primitive'] = 'push'

    params = {
        'pose0': (np.asarray(p0_position), np.asarray(p0_rotation)),
        'pose1': (np.asarray(p1_position), np.asarray(p1_rotation))
    }
    act['params'] = params
    return act



class GtState3Step6DAgent(GtState6DAgent):
  """Agent which uses ground-truth state information -- useful as a baseline."""

  def __init__(self, name, task):
    super().__init__(name, task)

    # Set up model.
    self.pick_model = None
    self.place_se2_model = None
    self.place_rpz_model = None

    self.pick_optim = tf.keras.optimizers.Adam(learning_rate=2e-4)
    self.place_se2_optim = tf.keras.optimizers.Adam(learning_rate=2e-4)
    self.place_rpz_optim = tf.keras.optimizers.Adam(learning_rate=2e-4)

    self.metric = tf.keras.metrics.Mean(name='metric')
    self.val_metric = tf.keras.metrics.Mean(name='val_metric')

  def init_model(self, dataset):
    """Initialize models, including normalization parameters."""
    self.set_max_obs_vector_length(dataset)

    _, _, info = dataset.random_sample()
    obs_vector = self.info_to_gt_obs(info)

    # Setup pick model
    obs_dim = obs_vector.shape[0]
    act_dim = 3
    self.pick_model = MlpModel(
        self.batch_size, obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

    obs_train_parameters=self.get_train_parameters(dataset)

    self.pick_model.set_normalization_parameters(obs_train_parameters)

    # Setup pick-conditioned place se2 model
    obs_dim = obs_vector.shape[0] + act_dim
    act_dim = 3
    self.place_se2_model = MlpModel(
        self.batch_size, obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

    obs_train_parameters=self.get_train_parameters(dataset)
    
    self.place_se2_model.set_normalization_parameters(obs_train_parameters)

    # Setup pick-conditioned place rpz model
    obs_dim = obs_vector.shape[0] + act_dim + 3
    act_dim = 3
    self.place_rpz_model = MlpModel(
        self.batch_size, obs_dim, act_dim, 'relu', self.use_mdn, dropout=0.1)

    obs_train_parameters=self.get_train_parameters(dataset)

    self.place_rpz_model.set_normalization_parameters(obs_train_parameters)
    return self.get_train_step()
  
  def get_train_step(self):
    pick_model=self.pick_model
    place_se2_model=self.place_se2_model
    place_rpz_model=self.place_rpz_model

    @tf.function
    def train_step(self,batch_obs,batch_act, loss_criterion):
      with tf.GradientTape() as tape:
        prediction = pick_model(batch_obs)
        loss0 = loss_criterion(batch_act[:, 0:3], prediction)
        grad = tape.gradient(loss0, pick_model.trainable_variables)
        self.pick_optim.apply_gradients(
            zip(grad, pick_model.trainable_variables))
      with tf.GradientTape() as tape:
        batch_obs = tf.concat((batch_obs, batch_act[:, 0:3]), axis=1)
        prediction = place_se2_model(batch_obs)
        loss1 = loss_criterion(batch_act[:, 3:6], prediction)
        grad = tape.gradient(loss1, place_se2_model.trainable_variables)
        self.place_se2_optim.apply_gradients(
            zip(grad, place_se2_model.trainable_variables))
      with tf.GradientTape() as tape:
        batch_obs = tf.concat((batch_obs, batch_act[:, 3:6]), axis=1)
        prediction = place_rpz_model(batch_obs)
        loss2 = loss_criterion(batch_act[:, 6:], prediction)
        grad = tape.gradient(loss2, place_rpz_model.trainable_variables)
        self.place_rpz_optim.apply_gradients(
            zip(grad, place_rpz_model.trainable_variables))
      return loss0 + loss1 + loss2
    return train_step

  def act(self, obs, info):
    """Run inference and return best action."""
    act = {'camera_config': self.camera_config, 'primitive': None}

    # Get observations and run pick prediction
    gt_obs = self.info_to_gt_obs(info)
    pick_prediction = self.pick_model(gt_obs[None, Ellipsis])

    pick_prediction=self.prediction_unbatch(pick_prediction)

    # Get observations and run place prediction
    obs_with_pick = np.hstack((gt_obs, pick_prediction)).astype(np.float32)

    # since the pick at train time is always 0.0,
    # the predictions are unstable if not exactly 0
    obs_with_pick[-1] = 0.0

    place_se2_prediction = self.place_se2_model(obs_with_pick[None, Ellipsis])
    
    place_se2_prediction=self.prediction_unbatch(place_se2_prediction)

    # Get observations and run rpz prediction
    obs_with_pick_place_se2 = np.hstack(
        (obs_with_pick, place_se2_prediction)).astype(np.float32)

    place_rpz_prediction = self.place_rpz_model(obs_with_pick_place_se2[None,
                                                                        Ellipsis])
    place_rpz_prediction=self.prediction_unbatch(place_rpz_prediction)

    p0_position = np.hstack((pick_prediction[0:2], 0.02))
    p0_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, 0))

    p1_position = np.hstack(
        (place_se2_prediction[0:2], place_rpz_prediction[2]))
    p1_rotation = utils.eulerXYZ_to_quatXYZW(
        (place_rpz_prediction[0] * self.theta_scale,
         place_rpz_prediction[1] * self.theta_scale,
         -place_se2_prediction[2] * self.theta_scale))

    # Select task-specific motion primitive.
    act['primitive'] = 'pick_place_6dof'

    params = {
        'pose0': (p0_position, p0_rotation),
        'pose1': (p1_position, p1_rotation)
    }
    act['params'] = params
    return act
