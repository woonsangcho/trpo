#!/usr/bin/env bash

import tensorflow as tf
from env_mujoco import Mujoco
from argparser import ArgParser
from keras import backend as K
import numpy as np
import random
from trpotrainer import TRPOTrainer

if __name__ == '__main__':
    parser = ArgParser()
    args = parser.parse_args()
    tf.reset_default_graph()

    seed = args.seed * 1958
    def set_global_seeds(i):
        tf.set_random_seed(i)
        np.random.seed(i)
        random.seed(i)
    set_global_seeds(seed)

    args.env = Mujoco(**vars(args))
    args.env.set_seed(seed)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    trainer = TRPOTrainer(**vars(args))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        K.set_session(sess)
        trainer.train(session = sess)
