import numpy as np

import tensorflow as tf

from wavenet import WaveNetModel


class TestGeneration(tf.test.TestCase):
    def setUp(self):
        self.net = WaveNetModel(batch_size=1,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                                filter_width=2,
                                residual_channels=64,
                                skip_channels=256,
                                quantization_channels=128,
                                gc_channels=7,
                                gc_cardinality=7,
                                lc_channels=88)

    def testGenerateFast(self):
        '''Generate a few samples using the fast method and
        perform sanity checks on the output.'''
        waveform = tf.placeholder(tf.int32)
        np.random.seed(0)
        data = np.random.randint(128)
        proba = self.net.predict_proba_incremental(waveform)

        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(self.net.init_ops)
            proba = sess.run(proba, feed_dict={waveform: data})

        self.assertAllEqual(proba.shape, [128])
        self.assertTrue(np.all((proba >= 0) & (proba <= (128 - 1))))


if __name__ == '__main__':
    tf.test.main()
