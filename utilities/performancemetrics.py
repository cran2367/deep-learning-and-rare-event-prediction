import tensorflow as tf
from tensorflow.python.keras import backend as K

########################################
##### Performance metric functions #####
########################################


class FalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, name='false_positive_rate', **kwargs):
        super(FalsePositiveRate, self).__init__(name=name, **kwargs)
        self.negatives = self.add_weight(name='negatives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_negatives',
                                               initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Arguments:
        y_true  The actual y. Passed by default to Metric classes.
        y_pred  The predicted y. Passed by default to Metric classes.

        '''
        # Compute the number of negatives.
        y_true = tf.cast(y_true, tf.bool)

        negatives = tf.reduce_sum(tf.cast(tf.equal(y_true, False), self.dtype))

        self.negatives.assign_add(negatives)

        # Compute the number of false positives.
        y_pred = tf.greater_equal(
            y_pred, 0.5
        )  # Using default threshold of 0.5 to call a prediction as positive labeled.

        false_positive_values = tf.logical_and(tf.equal(y_true, False),
                                               tf.equal(y_pred, True))
        false_positive_values = tf.cast(false_positive_values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(false_positive_values, sample_weight)

        false_positives = tf.reduce_sum(false_positive_values)

        self.false_positives.assign_add(false_positives)

    def result(self):
        return tf.divide(self.false_positives, self.negatives)


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.actual_positives = self.add_weight(name='actual_positives',
                                                initializer='zeros')
        self.predicted_positives = self.add_weight(name='predicted_positives',
                                                   initializer='zeros')
        self.true_positives = self.add_weight(name='true_positives',
                                              initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Arguments:
        y_true  The actual y. Passed by default to Metric classes.
        y_pred  The predicted y. Passed by default to Metric classes.

        '''
        # Compute the number of negatives.
        y_true = tf.cast(y_true, tf.bool)

        actual_positives = tf.reduce_sum(
            tf.cast(tf.equal(y_true, True), self.dtype))
        self.actual_positives.assign_add(actual_positives)

        # Compute the number of false positives.
        y_pred = tf.greater_equal(
            y_pred, 0.5
        )  # Using default threshold of 0.5 to call a prediction as positive labeled.

        predicted_positives = tf.reduce_sum(
            tf.cast(tf.equal(y_pred, True), self.dtype))
        self.predicted_positives.assign_add(predicted_positives)

        true_positive_values = tf.logical_and(tf.equal(y_true, True),
                                              tf.equal(y_pred, True))
        true_positive_values = tf.cast(true_positive_values, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(true_positive_values, sample_weight)

        true_positives = tf.reduce_sum(true_positive_values)

        self.true_positives.assign_add(true_positives)

    def result(self):
        recall = tf.math.divide_no_nan(self.true_positives,
                                       self.actual_positives)
        precision = tf.math.divide_no_nan(self.true_positives,
                                          self.predicted_positives)
        f1_score = 2 * tf.math.divide_no_nan(tf.multiply(recall, precision),
                                             tf.math.add(recall, precision))

        return f1_score