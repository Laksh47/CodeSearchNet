from typing import Dict, Any

import tensorflow as tf

from .seq_encoder import SeqEncoder
from utils.tfutils import write_to_feed_dict, pool_sequence_embedding


class Code2VecEncoder(SeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = { 'nbow_pool_mode': 'weighted_mean',
                         }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    def _make_placeholders(self):
        """
        Creates placeholders "tokens" and "tokens_mask" for masked sequence encoders.
        """
        super()._make_placeholders()
        self.placeholders['tokens_mask'] = \
            tf.compat.v1.placeholder(tf.float32,
                           shape=[None, self.get_hyper('max_num_tokens')],
                           name='tokens_mask')

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['tokens'] = []
        batch_data['tokens_mask'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        write_to_feed_dict(feed_dict, self.placeholders['tokens'], batch_data['tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['tokens_mask'], batch_data['tokens_mask'])

    @property
    def output_representation_size(self):
        return self.get_hyper('token_embedding_size')

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.compat.v1.variable_scope("nbow_encoder"):
            self._make_placeholders()

            seq_tokens_embeddings = self.embedding_layer(self.placeholders['tokens'])
            seq_token_mask = self.placeholders['tokens_mask']
            seq_token_lengths = tf.reduce_sum(input_tensor=seq_token_mask, axis=1)  # B
            return pool_sequence_embedding(self.get_hyper('nbow_pool_mode').lower(),
                                           sequence_token_embeddings=seq_tokens_embeddings,
                                           sequence_lengths=seq_token_lengths,
                                           sequence_token_masks=seq_token_mask)
