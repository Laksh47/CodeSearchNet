from collections import Counter
import numpy as np
from typing import Dict, Any, List, Iterable, Optional, Tuple
import random
import re

from utils.bpevocabulary import BpeVocabulary
from utils.tfutils import convert_and_pad_path_contexts
from utils.tfutils import write_to_feed_dict

import tensorflow as tf
from dpu_utils.codeutils import split_identifier_into_parts
from dpu_utils.mlutils import Vocabulary

from utils.general_utils import java_string_hashcode

from .code2vec_encoder_base import Code2VecEncoderBase, QueryType

IDENTIFIER_TOKEN_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')

class Code2VecEncoder(Code2VecEncoderBase):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {
            'nbow_pool_mode': 'weighted_mean',
            'token_vocab_size': 10000,
            'token_vocab_count_threshold': 10,
            'token_embedding_size': 128,

            'path_vocab_size': 10000,
            'path_vocab_count_threshold': 1,
            'path_embedding_size': 128,

            'use_subtokens': False,
            'mark_subtoken_end': False,

            'max_num_tokens': 200,
            'max_num_paths': 200,
            'max_num_contexts': 200,

            'code_vector_size': 128,
            'context_vector_size': 128,

            'use_bpe': True,
            'pct_bpe': 0.5
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)
        if hyperparameters['%s_use_bpe' % label]:
            assert not hyperparameters['%s_use_subtokens' % label], 'Subtokens cannot be used along with BPE.'
        elif hyperparameters['%s_use_subtokens' % label]:
            assert not hyperparameters['%s_use_bpe' % label], 'Subtokens cannot be used along with BPE.'

    def _make_placeholders(self):
        """
        Creates placeholders "tokens" and "tokens_mask" for masked sequence encoders.
        """
        super()._make_placeholders()

        self.placeholders['source_token'] = \
            tf.compat.v1.placeholder(tf.int32,
                           shape=[None, self.get_hyper('max_num_tokens')],
                           name='source_token')

        self.placeholders['target_token'] = \
            tf.compat.v1.placeholder(tf.int32,
                           shape=[None, self.get_hyper('max_num_tokens')],
                           name='target_token')

        self.placeholders['path'] = \
            tf.compat.v1.placeholder(tf.int32,
                           shape=[None, self.get_hyper('max_num_paths')],
                           name='path')

    def init_minibatch(self, batch_data: Dict[str, Any]) -> None:
        super().init_minibatch(batch_data)
        batch_data['source_tokens'] = []
        batch_data['target_tokens'] = []
        batch_data['paths'] = []

    def minibatch_to_feed_dict(self, batch_data: Dict[str, Any], feed_dict: Dict[tf.Tensor, Any], is_train: bool) -> None:
        super().minibatch_to_feed_dict(batch_data, feed_dict, is_train)
        write_to_feed_dict(feed_dict, self.placeholders['source_token'], batch_data['source_tokens'])
        write_to_feed_dict(feed_dict, self.placeholders['path'], batch_data['paths'])
        write_to_feed_dict(feed_dict, self.placeholders['target_token'], batch_data['target_tokens'])

    @property
    def output_representation_size(self):
        return self.get_hyper('token_embedding_size')

    def make_code2vec_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.compat.v1.variable_scope("code2vec_encoder"):
            self._make_placeholders()

            token_embeddings = tf.compat.v1.get_variable(name='token_embeddings',
                                           initializer=tf.compat.v1.glorot_uniform_initializer(),
                                           shape=[len(self.metadata['token_vocab']),
                                                  self.get_hyper('token_embedding_size')],
                                           )
            path_embeddings = tf.compat.v1.get_variable(name='path_embeddings',
                                           initializer=tf.compat.v1.glorot_uniform_initializer(),
                                           shape=[len(self.metadata['path_vocab']),
                                                  self.get_hyper('path_embedding_size')],
                                           )
            attention_param = tf.compat.v1.get_variable(
                'ATTENTION',
                shape=(self.get_hyper('code_vector_size'), 1), dtype=tf.float32)

            self.__token_embeddings = token_embeddings
            self.__path_embeddings = path_embeddings

            source_word_embed = tf.nn.embedding_lookup(params=token_embeddings, ids=self.placeholders['source_token'])  # (batch, max_contexts, dim)
            path_embed = tf.nn.embedding_lookup(params=path_embeddings, ids=self.placeholders['path'])  # (batch, max_contexts, dim)
            target_word_embed = tf.nn.embedding_lookup(params=token_embeddings, ids=self.placeholders['target_token'])  # (batch, max_contexts, dim)

            context_embed = tf.concat([source_word_embed, path_embed, target_word_embed],
                                  axis=-1)  # (batch, max_contexts, dim * 3)

            self.__context_embeddings = context_embed

            context_embed = tf.nn.dropout(context_embed, rate=1 - (self.placeholders['dropout_keep_rate']))

            flat_embed = tf.reshape(context_embed, [-1, self.get_hyper('context_vector_size')])  # (batch * max_contexts, dim * 3)
            transform_param = tf.compat.v1.get_variable(
                'TRANSFORM', shape=(self.get_hyper('context_vector_size'), self.get_hyper('code_vector_size')), dtype=tf.float32)

            flat_embed = tf.tanh(tf.matmul(flat_embed, transform_param))  # (batch * max_contexts, dim * 3)

            contexts_weights = tf.matmul(flat_embed, attention_param)  # (batch * max_contexts, 1)
            batched_contexts_weights = tf.reshape(
                contexts_weights, [-1, self.get_hyper('max_num_contexts'), 1])  # (batch, max_contexts, 1)

            # mask = tf.math.log(valid_mask)  # (batch, max_contexts)
            # mask = tf.expand_dims(mask, axis=2)  # (batch, max_contexts, 1)
            # batched_contexts_weights += mask  # (batch, max_contexts, 1)

            attention_weights = tf.nn.softmax(batched_contexts_weights, axis=1)  # (batch, max_contexts, 1)

            batched_embed = tf.reshape(flat_embed, shape=[-1, self.get_hyper('max_num_contexts'), self.get_hyper('context_vector_size')])
            code_vectors = tf.reduce_sum(tf.multiply(batched_embed, attention_weights), axis=1)  # (batch, dim * 3)

            return code_vectors

    def make_model(self, is_train: bool=False) -> tf.Tensor:
        with tf.compat.v1.variable_scope("code2vec_encoder"):
            self._make_placeholders()

            return self.make_code2vec_model()

    # def embedding_layer(self, token_inp: tf.Tensor) -> tf.Tensor:
    #     """
    #     Creates embedding layer that is in common between many encoders.

    #     Args:
    #         token_inp:  2D tensor that is of shape (batch size, sequence length)

    #     Returns:
    #         3D tensor of shape (batch size, sequence length, embedding dimension)
    #     """

    #     token_embeddings = tf.compat.v1.get_variable(name='token_embeddings',
    #                                        initializer=tf.compat.v1.glorot_uniform_initializer(),
    #                                        shape=[len(self.metadata['token_vocab']),
    #                                               self.get_hyper('token_embedding_size')],
    #                                        )
    #     self.__embeddings = token_embeddings

    #     token_embeddings = tf.nn.dropout(token_embeddings,
    #                                      rate=1 - (self.placeholders['dropout_keep_rate']))

    #     return tf.nn.embedding_lookup(params=token_embeddings, ids=token_inp)

    @classmethod
    def init_metadata(cls) -> Dict[str, Any]:
        raw_metadata = super().init_metadata()
        raw_metadata['token_counter'] = Counter()
        raw_metadata['path_counter'] = Counter()
        return raw_metadata

    @classmethod
    def _to_subtoken_stream(cls, input_stream: Iterable[str], mark_subtoken_end: bool) -> Iterable[str]:
        for token in input_stream:
            if IDENTIFIER_TOKEN_REGEX.match(token):
                yield from split_identifier_into_parts(token)
                if mark_subtoken_end:
                    yield '</id>'
            else:
                yield token

    @classmethod
    def get_path_tokens(cls, path_contexts, max_paths):
        source_tokens = []
        paths = []
        target_tokens = []

        parts = path_contexts.split(" ")
        # method_name = parts[0]
        contexts = parts[1:]

        for context in contexts[:max_paths]:
            # context = context.replace('METHOD_NAME', method_name)
            context_parts = context.split(",")
            source_token = context_parts[0]
            target_token = context_parts[2]
            path = context_parts[1]

            hashed_path = java_string_hashcode(path)
            source_tokens.append(source_token)
            target_tokens.append(target_token)
            paths.append(hashed_path)

        return (source_tokens, paths, target_tokens)

    @classmethod
    def load_metadata_from_sample(cls, tokens_to_load: Iterable[str], path_to_load: Iterable[str], raw_metadata: Dict[str, Any],
                                  use_subtokens: bool=False, mark_subtoken_end: bool=False) -> None:
        if use_subtokens:
            tokens_to_load = cls._to_subtoken_stream(tokens_to_load, mark_subtoken_end=mark_subtoken_end)
        raw_metadata['token_counter'].update(tokens_to_load)
        raw_metadata['path_counter'].update(path_to_load)

    @classmethod
    def finalise_metadata(cls, encoder_label: str, hyperparameters: Dict[str, Any], raw_metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        final_metadata = super().finalise_metadata(encoder_label, hyperparameters, raw_metadata_list)
        merged_token_counter = Counter()
        merged_path_counter = Counter()

        for raw_metadata in raw_metadata_list:
            merged_token_counter += raw_metadata['token_counter']
            merged_path_counter += raw_metadata['path_counter']

        if hyperparameters['%s_use_bpe' % encoder_label]:
            token_vocabulary = BpeVocabulary(vocab_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                             pct_bpe=hyperparameters['%s_pct_bpe' % encoder_label]
                                             )
            token_vocabulary.fit(merged_token_counter)
        else:
            token_vocabulary = Vocabulary.create_vocabulary(tokens=merged_token_counter,
                                                            max_size=hyperparameters['%s_token_vocab_size' % encoder_label],
                                                            count_threshold=hyperparameters['%s_token_vocab_count_threshold' % encoder_label])

        final_metadata['path_vocab'] = Vocabulary.create_vocabulary(tokens=merged_path_counter,
                                                            max_size=hyperparameters['%s_path_vocab_size' % encoder_label],
                                                            count_threshold=hyperparameters['%s_path_vocab_count_threshold' % encoder_label])
        final_metadata['common_paths'] = merged_path_counter.most_common(50)

        final_metadata['token_vocab'] = token_vocabulary
        # Save the most common tokens for use in data augmentation:
        final_metadata['common_tokens'] = merged_token_counter.most_common(50)
        return final_metadata

    @classmethod
    def load_data_from_sample(cls,
                              encoder_label: str,
                              hyperparameters: Dict[str, Any],
                              metadata: Dict[str, Any],
                              path_contexts: Any,
                              function_name: Optional[str],
                              result_holder: Dict[str, Any],
                              is_test: bool = True) -> bool:

        data_to_load = cls.get_path_tokens(path_contexts, hyperparameters[f'{encoder_label}_max_num_paths'])

        source_tokens, paths, target_tokens = \
                convert_and_pad_path_contexts(metadata['token_vocab'], metadata['path_vocab'], data_to_load,
                                               hyperparameters[f'{encoder_label}_max_num_paths'])

        result_holder['source_tokens'] = source_tokens
        result_holder['paths'] = paths
        result_holder['target_tokens'] = target_tokens

        return True

    def extend_minibatch_by_sample(self, batch_data: Dict[str, Any], sample: Dict[str, Any], is_train: bool=False,
                                   query_type: QueryType = QueryType.DOCSTRING.value) -> bool:
        """
        Implements various forms of data augmentation.
        """
        current_sample = dict()

        current_sample['source_tokens'] = sample['source_tokens']
        current_sample['paths'] = sample['paths']
        current_sample['target_tokens'] = sample['target_tokens']

        # Add the current sample to the minibatch:
        [batch_data[key].append(current_sample[key]) for key in current_sample.keys() if key in batch_data.keys()]

        return False

    def get_token_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return (self.__token_embeddings,
                list(self.metadata['token_vocab'].id_to_token))

    def get_path_embeddings(self) -> Tuple[tf.Tensor, List[str]]:
        return (self.__path_embeddings,
                list(self.metadata['token_vocab'].id_to_token))
