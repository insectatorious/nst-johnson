import tensorflow as tf

from models.transformer import Transformer

inputs = tf.random.uniform([1, 224, 224, 3])
model: tf.keras.Model = Transformer()

model.build(input_shape=(None, 224, 224, 3))
model.call(tf.keras.layers.Input(shape=(224, 224, 3)))
model.summary()

y = model.predict(inputs)
print(y.shape)
