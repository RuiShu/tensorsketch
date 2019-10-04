import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
import tqdm
import numpy as np
import tensorflow as tf
import tensorsketch as ts
from tensorsketch import SpectralNorm

model = ts.Sequential(
    ts.Conv2d(32, 3, 1),
    ts.ReLU(),
    ts.Conv2d(32, 3, 1),
    ts.ReLU(),
    ts.Conv2d(64, 3, 2),
    ts.ReLU(),
    ts.Conv2d(64, 3, 1),
    ts.ReLU(),
    ts.Conv2d(128, 3, 2),
    ts.ReLU(),
    ts.Conv2d(128, 3, 1),
    ts.ReLU(),
    ts.Conv2d(256, 3, 2),
    ts.ReLU(),
    ts.Conv2d(256, 3, 1),
    ts.ReLU(),
    ts.Conv2d(512, 3, 2),
    ts.ReLU(),
    ts.Conv2d(512, 3, 1),
    ts.ReLU(),
    ts.Flatten(),
    ts.Dense(500),
    ts.ReLU(),
    ts.Dense(10)
).build((1, 64, 64, 3))

model.apply(SpectralNorm.add, targets=(ts.Dense, ts.Conv2d))
opt = tf.keras.optimizers.Adam()
print(model.read(model.WITH_VARS))

@tf.function
def call():
    with tf.GradientTape() as tape:
        x = tf.random.normal((128, 64, 64, 3))
        loss = tf.reduce_sum(tf.square(model(x)))

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

max_iter = 10000
for _ in tqdm.tqdm(range(max_iter)):
    call()
