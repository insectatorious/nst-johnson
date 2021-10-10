import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError

from models.transformer import ResidualBlock, Transformer


class TestResidualBlock(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.num_of_filters = 3
    self.input_shape = (8, 8, self.num_of_filters)
    self.inputs = tf.random.uniform([128, *self.input_shape])

  def test_num_of_filters(self):
    block: ResidualBlock = ResidualBlock(num_of_filters=32)
    assert block.num_of_filters == 32

  def test_all_layers_connected(self):
    layer = ResidualBlock(input_shape=self.input_shape,
                          num_of_filters=self.num_of_filters)

    model = Sequential([layer])
    model.compile(loss=MeanSquaredError())

    conv1_weights = layer.conv_1.get_weights()
    conv2_weights = layer.conv_2.get_weights()

    model.fit(x=self.inputs, y=self.inputs, epochs=2, verbose=0)

    self.assertNotAllClose(conv1_weights,
                           layer.conv_1.get_weights())
    self.assertNotAllClose(conv2_weights,
                           layer.conv_2.get_weights())

  @pytest.mark.skip("Need to figure out how to properly test this")
  def test_identity(self):
    num_of_filters = 3
    input_shape = (8, 8, 3)
    inputs = tf.random.uniform([128, *input_shape])

    model = Sequential([ResidualBlock(input_shape=input_shape,
                                      num_of_filters=num_of_filters)])
    model.compile(loss=MeanSquaredError())

    conv1 = model.layers[0].conv_1.get_weights()
    model.fit(x=inputs, y=inputs, epochs=10)

    self.assertAllClose(inputs[:0], model.predict(inputs[:0]))

  @pytest.mark.skip("Will add later. May not be needed.")
  def test_mnist_classify(self):
    model = tf.keras.models.Sequential([
      ResidualBlock(input_shape=(28, 28, 1)),
      ResidualBlock(),

    ])


class TestTransformer(tf.test.TestCase):
  def setUp(self):
    self.mscoco_shape = (224, 224, 3)

  def test_output_shape_is_mscoco(self):
    inputs = tf.random.uniform([1, *self.mscoco_shape])
    model: tf.keras.Model = Transformer()

    model.build(input_shape=(None, *self.mscoco_shape))
    model.call(tf.keras.layers.Input(shape=self.mscoco_shape))

    y = model.predict(inputs)
    self.assertShapeEqual(y, inputs)

