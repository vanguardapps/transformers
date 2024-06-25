# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Generation configuration class and utilities."""

import copy
import json
import os
import sqlite3
import torch
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from itertools import islice
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import faiss
import numpy as np
import regex as re
from tqdm import tqdm

from .. import __version__
from ..configuration_utils import PretrainedConfig
from ..utils import (
    GENERATION_CONFIG_NAME,
    ExplicitEnum,
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    is_remote_url,
    is_torch_available,
    logging,
)


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)
METADATA_FIELDS = (
    "_from_model_config",
    "_commit_hash",
    "_original_object_hash",
    "transformers_version",
)
NEEDS_CACHE_CONFIG = {}

if is_torch_available():
    from ..cache_utils import QuantizedCacheConfig

    NEEDS_CACHE_CONFIG["quantized"] = QuantizedCacheConfig


class GenerationMode(ExplicitEnum):
    """
    Possible generation modes, downstream of the [`~generation.GenerationMixin.generate`] method.
    """

    # Non-beam methods
    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    # Beam methods
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationConfig(PushToHubMixin):
    # no-format
    r"""
    Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
    for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *contrastive search* if `penalty_alpha>0.` and `top_k>1`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
        - *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-search decoding* if `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    To learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).

    <Tip>

    A large number of these flags control the logits or the stopping criteria of the generation. Make sure you check
    the [generate-related classes](https://huggingface.co/docs/transformers/internal/generation_utils) for a full
    description of the possible manipulations, as well as examples of their usage.

    </Tip>

    Arg:
        > Parameters that control the length of the output

        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        min_length (`int`, *optional*, defaults to 0):
            The minimum length of the sequence to be generated. Corresponds to the length of the input prompt +
            `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.
        min_new_tokens (`int`, *optional*):
            The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        early_stopping (`bool` or `str`, *optional*, defaults to `False`):
            Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values:
            `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very unlikely to find better candidates;
            `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical
            beam search algorithm).
        max_time(`float`, *optional*):
            The maximum amount of time you allow the computation to run for in seconds. generation will still finish
            the current pass after allocated time has been passed.
        stop_strings(`str or List[str]`, *optional*):
            A string or a list of strings that should terminate generation if the model outputs them.

        > Parameters that control the generation strategy used

        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        num_beam_groups (`int`, *optional*, defaults to 1):
            Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
            [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
        penalty_alpha (`float`, *optional*):
            The values balance the model confidence and the degeneration penalty in contrastive search decoding.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.

        > Parameters for manipulation of the model output logits

        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        top_k (`int`, *optional*, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between 0 and 1. Typical values are in the 0.01-0.2 range, comparably selective as setting `top_p` in
            the 0.99-0.8 range (use the opposite of normal `top_p` values).
        typical_p (`float`, *optional*, defaults to 1.0):
            Local typicality measures how similar the conditional probability of predicting a target token next is to
            the expected conditional probability of predicting a random token next, given the partial text already
            generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that
            add up to `typical_p` or higher are kept for generation. See [this
            paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
        epsilon_cutoff (`float`, *optional*, defaults to 0.0):
            If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
            `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
            size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
        eta_cutoff (`float`, *optional*, defaults to 0.0):
            Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between
            0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) *
            exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token
            probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3,
            depending on the size of the model. See [Truncation Sampling as Language Model
            Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
        diversity_penalty (`float`, *optional*, defaults to 0.0):
            This value is subtracted from a beam's score if it generates a token same as any beam from other group at a
            particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        encoder_repetition_penalty (`float`, *optional*, defaults to 1.0):
            The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the
            original input. 1.0 means no penalty.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences.
        no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids(`List[List[int]]`, *optional*):
            List of list of token ids that are not allowed to be generated. Check
            [`~generation.NoBadWordsLogitsProcessor`] for further documentation and examples.
        force_words_ids(`List[List[int]]` or `List[List[List[int]]]`, *optional*):
            List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of
            words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this
            triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
            can allow different forms of each word.
        renormalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to renormalize the logits after applying all the logits processors or warpers (including the custom
            ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits
            are normalized but some logit processors or warpers break the normalization.
        constraints (`List[Constraint]`, *optional*):
            Custom constraints that can be added to the generation to ensure that the output will contain the use of
            certain tokens as defined by `Constraint` objects, in the most sensible way possible.
        forced_bos_token_id (`int`, *optional*, defaults to `model.config.forced_bos_token_id`):
            The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful for
            multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be the target
            language token.
        forced_eos_token_id (`Union[int, List[int]]`, *optional*, defaults to `model.config.forced_eos_token_id`):
            The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a
            list to set multiple *end-of-sequence* tokens.
        remove_invalid_values (`bool`, *optional*, defaults to `model.config.remove_invalid_values`):
            Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash.
            Note that using `remove_invalid_values` can slow down generation.
        exponential_decay_length_penalty (`tuple(int, float)`, *optional*):
            This Tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been
            generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where
            penalty starts and `decay_factor` represents the factor of exponential decay
        suppress_tokens  (`List[int]`, *optional*):
            A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their
            log probs to `-inf` so that they are not sampled.
        begin_suppress_tokens  (`List[int]`, *optional*):
            A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit
            processor will set their log probs to `-inf` so that they are not sampled.
        forced_decoder_ids (`List[List[int]]`, *optional*):
            A list of pairs of integers which indicates a mapping from generation indices to token indices that will be
            forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token
            of index 123.
        sequence_bias (`Dict[Tuple[int], float]`, *optional*)):
            Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the
            sequence being selected, while negative biases do the opposite. Check
            [`~generation.SequenceBiasLogitsProcessor`] for further documentation and examples.
        token_healing (`bool`, *optional*, defaults to `False`):
            Heal tail tokens of prompts by replacing them with their appropriate extensions.
            This enhances the quality of completions for prompts affected by greedy tokenization bias.
        guidance_scale (`float`, *optional*):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality.
        low_memory (`bool`, *optional*):
            Switch to sequential beam search and sequential topk for contrastive search to reduce peak memory.
            Used with beam search and contrastive search.
        watermarking_config (Union[`WatermarkingConfig`, `dict`], *optional*):
            Arguments used to watermark the model outputs by adding a small bias to randomly selected set of "green" tokens.
            If passed as `Dict`, it will be converted to a `WatermarkingConfig` internally.
            See [this paper](https://arxiv.org/abs/2306.04634) for more details. Accepts the following keys:
            - greenlist_ratio (`float`):
                Used for watermarking. The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
            - bias (`float`):
                Used with watermarking. The bias added to the selected "green" tokens' logits. Defaults to 2.0.
            - hashing_key (`int`):
                Hahsing key used for watermarking. Defaults to 15485863 (the millionth prime).
            - seeding_scheme (`str`):
                Algorithm to use for watermarking. Accepts values:
                    - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from the paper)
                    - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from the paper)
                        The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
            - context_width(`int`):
                The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
        knn_store (`KNNStore`, *optional*):
            KNN datastore that, if supplied, is used to perform KNN-MT-style interpolation on the logits during decoding.
        knn_k (`int`, *optional): The number of top k target tokens to retrieve from the KNN datastore for interpolation
            with the preexisting base model logits. Note: if k is higher than the number of available tokens, all
            available tokens will be used. Defaults to 20.
        knn_temperature (`float`, *optional*): The temperature to be used when interpolating the top k results from the
            KNN datastore with the results from the base model. See [this paper](https://arxiv.org/abs/2010.00710) for
            details on the exact math used during interpolation. Defaults to 1.
        knn_interpolation_coefficient (`float`, *optional*):
            Determines the proportion of interpolation between the base model logits and the KNN-computed logits. See
            [this paper](https://arxiv.org/abs/2010.00710) for details on the exact math used during interpolation.
            Only has an effect if provided alongside `knn_store`. Defaults to 0.5.

        > Parameters that define the output variables of generate

        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        output_logits (`bool`, *optional*):
            Whether or not to return the unprocessed prediction logit scores. See `logits` under returned tensors for
            more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        > Special tokens that can be used at generation time

        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.

        > Generation parameters exclusive to encoder-decoder models

        encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
            If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
            `decoder_input_ids`.
        decoder_start_token_id (`Union[int, List[int]]`, *optional*):
            If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token or a list of length
            `batch_size`. Indicating a list enables different start ids for each element in the batch
            (e.g. multilingual models with different target languages in one batch)

        > Generation parameters exclusive to assistant generation

        num_assistant_tokens (`int`, *optional*, defaults to 5):
            Defines the number of _speculative tokens_ that shall be generated by the assistant model before being
            checked by the target model at each iteration. Higher values for `num_assistant_tokens` make the generation
            more _speculative_ : If the assistant model is performant larger speed-ups can be reached, if the assistant
            model requires lots of corrections, lower speed-ups are reached.
        num_assistant_tokens_schedule (`str`, *optional*, defaults to `"heuristic"`):
            Defines the schedule at which max assistant tokens shall be changed during inference.
            - `"heuristic"`: When all speculative tokens are correct, increase `num_assistant_tokens` by 2 else
              reduce by 1. `num_assistant_tokens` value is persistent over multiple generation calls with the same assistant model.
            - `"heuristic_transient"`: Same as `"heuristic"` but `num_assistant_tokens` is reset to its initial value after each generation call.
            - `"constant"`: `num_assistant_tokens` stays unchanged during generation
        prompt_lookup_num_tokens (`int`, *optional*, default to `None`):
            The number of tokens to be output as candidate tokens.
        max_matching_ngram_size (`int`, *optional*, default to `None`):
            The maximum ngram size to be considered for matching in the prompt. Default to 2 if not provided.

        > Parameters specific to the caching mechanism:

        cache_implementation (`str`, *optional*, default to `None`):
            Cache class that should be used when generating.
        cache_config (`Union[CacheConfig, dict]`, *optional*, default to `None`):
            Arguments used in the key-value cache class can be passed in `cache_config`. Can be passed as a `Dict` and
            it will be converted to its repsective `CacheConfig` internally.
            Otherwise can be passed as a `CacheConfig` class matching the indicated `cache_implementation`.
        return_legacy_cache (`bool`, *optional*, default to `True`):
            Whether to return the legacy or new format of the cache when `DynamicCache` is used by default.

        > Wild card

        generation_kwargs:
            Additional generation kwargs will be forwarded to the `generate` function of the model. Kwargs that are not
            present in `generate`'s signature will be used in the model forward pass.
    """

    def __init__(self, **kwargs):
        # Parameters that control the length of the output
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        # Parameters that control the generation strategy used
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.use_cache = kwargs.pop("use_cache", True)

        # Parameters for manipulation of the model output logits
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.min_p = kwargs.pop("min_p", None)
        self.typical_p = kwargs.pop("typical_p", 1.0)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", 0.0)
        self.eta_cutoff = kwargs.pop("eta_cutoff", 0.0)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)
        self.constraints = kwargs.pop("constraints", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.exponential_decay_length_penalty = kwargs.pop(
            "exponential_decay_length_penalty", None
        )
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.forced_decoder_ids = kwargs.pop("forced_decoder_ids", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", False)
        self.guidance_scale = kwargs.pop("guidance_scale", None)
        self.low_memory = kwargs.pop("low_memory", None)
        watermarking_config = kwargs.pop("watermarking_config", None)
        if watermarking_config is None:
            self.watermarking_config = None
        elif isinstance(watermarking_config, WatermarkingConfig):
            self.watermarking_config = watermarking_config
        else:
            self.watermarking_config = WatermarkingConfig.from_dict(watermarking_config)
        knn_store = kwargs.pop("knn_store", None)
        if isinstance(knn_store, KNNStore):
            self.knn_store = knn_store
        else:
            self.knn_store = None
        self.knn_k = kwargs.pop("knn_k", 20)
        self.knn_temperature = kwargs.pop("knn_temperature", 1)
        self.knn_interpolation_coefficient = kwargs.pop(
            "knn_interpolation_coefficient", 0.5
        )

        # Parameters that define the output variables of `generate`
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Generation parameters exclusive to encoder-decoder models
        self.encoder_no_repeat_ngram_size = kwargs.pop(
            "encoder_no_repeat_ngram_size", 0
        )
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # Assistant generation
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", 5)
        self.num_assistant_tokens_schedule = kwargs.pop(
            "num_assistant_tokens_schedule", "heuristic"
        )

        # Cache implementation
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        if self.cache_implementation is not None:
            cache_config_class = NEEDS_CACHE_CONFIG[self.cache_implementation]
            if self.cache_config is None:
                self.cache_config = cache_config_class()
            elif isinstance(self.cache_config, dict):
                self.cache_config = cache_config_class.from_dict(self.cache_config)
        self.return_legacy_cache = kwargs.pop("return_legacy_cache", True)

        # Prompt lookup decoding
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_without_metadata = self.to_json_string(
            use_diff=False, ignore_metadata=True
        )
        other_without_metadata = other.to_json_string(
            use_diff=False, ignore_metadata=True
        )
        return self_without_metadata == other_without_metadata

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    def get_generation_mode(
        self, assistant_model: Optional["PreTrainedModel"] = None
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        # TODO joao: find out a way of not depending on external fields (e.g. `assistant_model`), then make this a
        # property and part of the `__repr__`
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams == 1:
            if self.do_sample is False:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if assistant_model is not None or self.prompt_lookup_num_tokens is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode

    def validate(self, is_init=False):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters not validated here are best validated at generate runtime, as they may depend on
        other inputs and/or the model, such as parameters related to the generation length.

        Arg:
            is_init (`bool`, *optional*, defaults to `False`):
                Whether the validation is performed during the initialization of the instance.
        """

        # Validation of individual attributes
        if self.early_stopping not in {True, False, "never"}:
            raise ValueError(
                f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}."
            )
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(
                f"`max_new_tokens` must be greater than 0, but is {self.max_new_tokens}."
            )
        if self.pad_token_id is not None and self.pad_token_id < 0:
            warnings.warn(
                f"`pad_token_id` should be positive but got {self.pad_token_id}. This will cause errors when batch generating, if there is padding. "
                "Please set `pad_token_id` explicitly by `model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation, and ensure your `input_ids` input does not have negative values."
            )

        # Validation of attribute relations:
        fix_location = ""
        if is_init:
            fix_location = (
                " This was detected when initializing the generation config instance, which means the corresponding "
                "file may hold incorrect parameterization and should be fixed."
            )

        # 1. detect sampling-only parameterization when not in sampling mode
        if self.do_sample is False:
            greedy_wrong_parameter_msg = (
                "`do_sample` is set to `False`. However, `{flag_name}` is set to `{flag_value}` -- this flag is only "
                "used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."
                + fix_location
            )
            if self.temperature is not None and self.temperature != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="temperature", flag_value=self.temperature
                    ),
                    UserWarning,
                )
            if self.top_p is not None and self.top_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="top_p", flag_value=self.top_p
                    ),
                    UserWarning,
                )
            if self.min_p is not None:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="min_p", flag_value=self.min_p
                    ),
                    UserWarning,
                )
            if self.typical_p is not None and self.typical_p != 1.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="typical_p", flag_value=self.typical_p
                    ),
                    UserWarning,
                )
            if (
                self.top_k is not None
                and self.top_k != 50
                and self.penalty_alpha is None
            ):  # contrastive search uses top_k
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="top_k", flag_value=self.top_k
                    ),
                    UserWarning,
                )
            if self.epsilon_cutoff is not None and self.epsilon_cutoff != 0.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff
                    ),
                    UserWarning,
                )
            if self.eta_cutoff is not None and self.eta_cutoff != 0.0:
                warnings.warn(
                    greedy_wrong_parameter_msg.format(
                        flag_name="eta_cutoff", flag_value=self.eta_cutoff
                    ),
                    UserWarning,
                )

        # 2. detect beam-only parameterization when not in beam mode
        if self.num_beams is None:
            warnings.warn("`num_beams` is set to None - defaulting to 1.", UserWarning)
            self.num_beams = 1

        if self.num_beams == 1:
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to 1. However, `{flag_name}` is set to `{flag_value}` -- this flag is only used "
                "in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`."
                + fix_location
            )
            if self.early_stopping is not False:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="early_stopping", flag_value=self.early_stopping
                    ),
                    UserWarning,
                )
            if self.num_beam_groups is not None and self.num_beam_groups != 1:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="num_beam_groups", flag_value=self.num_beam_groups
                    ),
                    UserWarning,
                )
            if self.diversity_penalty is not None and self.diversity_penalty != 0.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="diversity_penalty", flag_value=self.diversity_penalty
                    ),
                    UserWarning,
                )
            if self.length_penalty is not None and self.length_penalty != 1.0:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="length_penalty", flag_value=self.length_penalty
                    ),
                    UserWarning,
                )
            if self.constraints is not None:
                warnings.warn(
                    single_beam_wrong_parameter_msg.format(
                        flag_name="constraints", flag_value=self.constraints
                    ),
                    UserWarning,
                )

        # 3. detect incorrect paramaterization specific to advanced beam modes
        else:
            # constrained beam search
            if self.constraints is not None or self.force_words_ids is not None:
                constrained_wrong_parameter_msg = (
                    "one of `constraints`, `force_words_ids` is not `None`, triggering constrained beam search. However, "
                    "`{flag_name}` is set to `{flag_value}`, which is incompatible with this generation mode. Set "
                    "`constraints` and `force_words_ids` to `None` or unset `{flag_name}` to continue."
                    + fix_location
                )
                if self.do_sample is True:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(
                            flag_name="do_sample", flag_value=self.do_sample
                        )
                    )
                if self.num_beam_groups is not None and self.num_beam_groups != 1:
                    raise ValueError(
                        constrained_wrong_parameter_msg.format(
                            flag_name="num_beam_groups", flag_value=self.num_beam_groups
                        )
                    )
            # group beam search
            if self.diversity_penalty != 0.0 or self.num_beam_groups != 1:
                group_error_prefix = (
                    "`diversity_penalty` is not 0.0 or `num_beam_groups` is not 1, triggering group beam search. In "
                    "this generation mode, "
                )
                if self.do_sample is True:
                    raise ValueError(
                        group_error_prefix + "`do_sample` must be set to `False`"
                    )
                if self.num_beams % self.num_beam_groups != 0:
                    raise ValueError(
                        group_error_prefix
                        + "`num_beams` should be divisible by `num_beam_groups`"
                    )
                if self.diversity_penalty == 0.0:
                    raise ValueError(
                        group_error_prefix
                        + "`diversity_penalty` should be greater than `0.0`, otherwise your groups will be identical."
                    )

        # 4. check `num_return_sequences`
        if self.num_return_sequences != 1:
            if self.num_beams == 1:
                if self.do_sample is False:
                    raise ValueError(
                        "Greedy methods without beam search do not support `num_return_sequences` different than 1 "
                        f"(got {self.num_return_sequences})."
                    )
            elif self.num_return_sequences > self.num_beams:
                raise ValueError(
                    f"`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` "
                    f"({self.num_beams})."
                )

        # 5. check `cache_config`
        if self.cache_config is not None:
            cache_class = NEEDS_CACHE_CONFIG.get(self.cache_implementation)
            if cache_class is None:
                raise ValueError(
                    "You provided a `cache_config` but the cache implementation you are using "
                    f"({self.cache_implementation}) does not require any config. Make sure to use the "
                    "correct cache implementation matching your cache config."
                )
            if not isinstance(self.cache_config, cache_class):
                self.cache_config = cache_class.from_dict(self.cache_config)
            self.cache_config.validate()

        # 6.  check watermarking arguments
        if self.watermarking_config is not None:
            if not isinstance(self.watermarking_config, WatermarkingConfig):
                self.watermarking_config = WatermarkingConfig.from_dict(
                    self.watermarking_config
                )
            self.watermarking_config.validate()

        # 7. check KNN datastore arguments
        if self.knn_store is not None:
            self.knn_store.validate()

        # 8. check common issue: passing `generate` arguments inside the generation config
        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
        )
        for arg in generate_arguments:
            if hasattr(self, arg):
                raise ValueError(
                    f"Argument `{arg}` is not a valid argument of `GenerationConfig`. It should be passed to "
                    "`generate()` (or a pipeline) directly."
                )

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        r"""
        Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~GenerationConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be saved in `save_directory`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        # At save time, validate the instance -- if any warning/exception is thrown, we refuse to save the instance.
        # This strictness is enforced to prevent bad configurations from being saved and re-used.
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.validate()
            if len(caught_warnings) > 0:
                raise ValueError(str([w.message for w in caught_warnings]))
        except ValueError as exc:
            raise ValueError(
                "The generation config instance is invalid -- `.validate()` throws warnings and/or exceptions. "
                "Fix these issues to save the configuration.\n\nThrown during validation:\n"
                + str(exc)
            )

        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        config_file_name = (
            config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME
        )

        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: Union[str, os.PathLike],
        config_file_name: Optional[Union[str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "GenerationConfig":
        r"""
        Instantiate a [`GenerationConfig`] from a generation configuration file.

        Args:
            pretrained_model_name (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~GenerationConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be loaded from `pretrained_model_name`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        >>> from transformers import GenerationConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> generation_config = GenerationConfig.from_pretrained("openai-community/gpt2")

        >>> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
        >>> generation_config.save_pretrained("./test/saved_model/")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

        >>> # You can also specify configuration names to your generation configuration file
        >>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

        >>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
        >>> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
        >>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        ...     "openai-community/gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
        ... )
        >>> generation_config.top_k
        1

        >>> unused_kwargs
        {'foo': False}
        ```"""
        config_file_name = (
            config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME
        )

        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        config_path = os.path.join(pretrained_model_name, config_file_name)
        config_path = str(config_path)

        is_local = os.path.exists(config_path)
        if os.path.isfile(os.path.join(subfolder, config_path)):
            # Special case when config_path is a local file
            resolved_config_file = config_path
            is_local = True
        elif is_remote_url(config_path):
            configuration_file = config_path
            resolved_config_file = download_url(config_path)
        else:
            configuration_file = config_file_name
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(
                f"loading configuration file {configuration_file} from cache at {resolved_config_file}"
            )

        if kwargs.get("return_unused_kwargs") is True:
            config, unused_kwargs = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(
                config
            )  # Hash to detect whether the instance was modified
            return config, unused_kwargs
        else:
            config = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(
                config
            )  # Hash to detect whether the instance was modified
            return config

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        # The commit hash might have been updated in the `config_dict`, we don't want the kwargs to erase that update.
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        # The line below allows model-specific config to be loaded as well through kwargs, with safety checks.
        # See https://github.com/huggingface/transformers/pull/21269
        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_torch_dtype_to_str(self, d: Dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("torch_dtype", None) is not None and not isinstance(
            d["torch_dtype"], str
        ):
            d["torch_dtype"] = str(d["torch_dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_torch_dtype_to_str(value)

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
            ):
                serializable_config_dict[key] = value

        self.dict_torch_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Fields to ignore at serialization time
        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]

        # Transformers version when serializing this file
        output["transformers_version"] = __version__

        self.dict_torch_dtype_to_str(output)
        return output

    def to_json_string(
        self, use_diff: bool = True, ignore_metadata: bool = False
    ) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {
                    str(key): convert_keys_to_string(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        def convert_dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {
                    key: convert_dataclass_to_dict(value) for key, value in obj.items()
                }
            elif is_dataclass(obj):
                return obj.to_dict()
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)
        config_dict = convert_dataclass_to_dict(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(
        self, json_file_path: Union[str, os.PathLike], use_diff: bool = True
    ):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`]. This function is useful to convert legacy
        [`PretrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict()
        config_dict.pop("_from_model_config", None)
        config = cls.from_dict(
            config_dict, return_unused_kwargs=False, _from_model_config=True
        )

        # Special case: some models have generation attributes set in the decoder. Use them if still unset in the
        # generation config.
        for decoder_name in ("decoder", "generator", "text_config"):
            if decoder_name in config_dict:
                default_generation_config = GenerationConfig()
                decoder_config = config_dict[decoder_name]
                for attr in config.to_dict().keys():
                    if attr in decoder_config and getattr(config, attr) == getattr(
                        default_generation_config, attr
                    ):
                        setattr(config, attr, decoder_config[attr])

        config._original_object_hash = hash(
            config
        )  # Hash to detect whether the instance was modified
        return config

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        # Confirm that the updated instance is still valid
        self.validate()

        # Remove all the attributes that were updated, without modifying the input dict
        unused_kwargs = {
            key: value for key, value in kwargs.items() if key not in to_remove
        }
        return unused_kwargs


@dataclass
class WatermarkingConfig:
    """
    Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
    See [this paper](https://arxiv.org/abs/2306.04634) for more details on the arguments.

    Accepts the following keys:
        - greenlist_ratio (`float`):
            Used for watermarking. The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
        - bias (`float`):
            Used with watermarking. The bias added to the selected "green" tokens' logits. Defaults to 2.0.
        - hashing_key (`int`):
            Hashing key used for watermarking. Defaults to 15485863 (the millionth prime).
        - seeding_scheme (`str`):
            Algorithm to use for watermarking. Accepts values:
                - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from the paper)
                - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from the paper)
                    The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
        - context_width(`int`):
            The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
    """

    def __init__(
        self,
        greenlist_ratio: Optional[float] = 0.25,
        bias: Optional[float] = 2.0,
        hashing_key: Optional[int] = 15485863,
        seeding_scheme: Optional[str] = "lefthash",
        context_width: Optional[int] = 1,
    ):
        self.greenlist_ratio = greenlist_ratio
        self.bias = bias
        self.hashing_key = hashing_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a WatermarkingConfig instance from a dictionary of parameters.

        Args:
            config_dict (Dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            WatermarkingConfig: Instance of WatermarkingConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (Union[str, os.PathLike]): Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            Dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __iter__(self):
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.

        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        """
        Update the configuration attributes with new values.

        Args:
            **kwargs: Keyword arguments representing configuration attributes and their new values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self):
        watermark_missing_arg_msg = (
            "Some of the keys in `watermarking_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="seeding_scheme",
                    correct_value="[`selfhash`, `lefthash`]",
                    found_value=self.seeding_scheme,
                ),
            )
        if not 0.0 <= self.greenlist_ratio <= 1.0:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="greenlist_ratio",
                    correct_value="in range between 0.0 and 1.0",
                    found_value=self.seeding_scheme,
                ),
            )
        if not self.context_width >= 1:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="context_width",
                    correct_value="a positive integer",
                    found_value=self.context_width,
                ),
            )


@dataclass
class KNNStore(ABC):
    """KNN-MT embeddings store abstract class.

    Note: No database implementation takes place in the abstract class.

    Attributes:
        default_table_prefix (str): (class attribute)
        default_configuration_table_stem (str): (class attribute)
        default_embedding_table_stem (str): (class attribute)
        default_faiss_cache_table_stem (str): (class attribute)
        default_embedding_dtype (str): (class attribute)
        default_embedding_batch_size (int): (class attribute)
        default_target_batch_size (int): (class attribute)
        default_c (int): (class attribute)
        table_prefix (str):
        configuration_table_stem (str):
        embedding_table_stem (str):
        faiss_cache_table_stem (str):
        target_build_table_stem (str):
        configuration_table_name (str):
        embedding_table_name (str):
        faiss_cache_table_name (str):
        embedding_dim (int):
        embedding_dtype (str):
        embedding_batch_size (int):
        target_batch_size (int):
        c (int):
    """

    default_table_prefix = "knn_store"
    default_configuration_table_stem = "config"
    default_embedding_table_stem = "embedding"
    default_faiss_cache_table_stem = "faiss_index"
    default_embedding_batch_size = 50
    default_target_batch_size = 50
    default_embedding_dtype = "float32"
    default_c = 5

    def __init__(
        self,
        embedding_dim=None,
        table_prefix=None,
        configuration_table_stem=None,
        embedding_table_stem=None,
        faiss_cache_table_stem=None,
        embedding_batch_size=None,
        target_batch_size=None,
        embedding_dtype=None,
        c=None,
        **kwargs,
    ):
        """Initializes KNNStore instance.

        Note: Subclasses must call `super().__init_()` with all constructor arguments and any `kwargs`
        needed for subclass implementation of `self._initialize_database(**kwargs)`.

        Args:
            embedding_dim (int):
            table_prefix (str):
            configuration_table_stem (str):
            embedding_table_stem (str):
            faiss_cache_table_stem (str):
            embedding_batch_size (int):
            target_batch_size (int):
            embedding_dtype (str):
            c (int):
            **kwargs (dict):

        """
        self.embedding_dim = embedding_dim

        self.table_prefix = (
            table_prefix if table_prefix is not None else KNNStore.default_table_prefix
        )

        self.configuration_table_stem = (
            configuration_table_stem
            if configuration_table_stem is not None
            else KNNStore.default_configuration_table_stem
        )

        self.embedding_table_stem = (
            embedding_table_stem
            if embedding_table_stem is not None
            else KNNStore.default_embedding_table_stem
        )

        self.faiss_cache_table_stem = (
            faiss_cache_table_stem
            if faiss_cache_table_stem is not None
            else KNNStore.default_faiss_cache_table_stem
        )

        self.embedding_batch_size = (
            embedding_batch_size
            if embedding_batch_size is not None
            else KNNStore.default_embedding_batch_size
        )

        self.target_batch_size = (
            target_batch_size
            if target_batch_size is not None
            else KNNStore.default_target_batch_size
        )

        self.embedding_dtype = (
            embedding_dtype
            if embedding_dtype is not None
            else KNNStore.default_embedding_dtype
        )

        self.c = c if c is not None else KNNStore.default_c

        self.configuration_table_name = (
            self.table_prefix + "_" + self.configuration_table_stem
        )
        self.embedding_table_name = self.table_prefix + "_" + self.embedding_table_stem
        self.faiss_cache_table_name = (
            self.table_prefix + "_" + self.faiss_cache_table_stem
        )

        self._reset_source_token_embeddings_offset()

        self._initialize_database(**kwargs)

    @staticmethod
    def _batched(iterable, n):
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch

    @staticmethod
    def _convert_faiss_index_to_bytestring(faiss_index):
        serialized_index = faiss.serialize_index(faiss_index)
        return serialized_index.tobytes()

    # TODO: Provide KNN batch that does aligning using fast_align. This is going to be kind of a
    # rough spot in the implementation. Makes me want to get back into C++ and learn fast_align
    # from scratch, make a python port of it or something. That would be a real selling point
    # for this module though, as very few people can say they have a sentence aligner in
    # code (I'm actually not sure I should look and see if someone has done this).

    #
    # Methods provided as part of base class
    #

    def ingest(self, knn_batch):
        for (
            source_token_ids,
            target_ids,
            alignments,
            source_embeddings,
            target_embeddings,
        ) in zip(
            knn_batch.input_ids_masked,
            knn_batch.label_ids_masked,
            knn_batch.alignments,
            knn_batch.encoder_last_hidden_state_masked,
            knn_batch.target_hidden_states_masked,
        ):
            for source_index, source_token_id in enumerate(source_token_ids):
                target_index = alignments.get(source_index, None)

                # Ignore any source token that was not aligned to a target token
                if target_index:
                    source_token_id = source_token_id
                    target_token_id = target_ids[target_index]
                    source_embedding = source_embeddings[source_index]
                    target_embedding = target_embeddings[target_index]
                    source_embedding_bytestring = source_embedding.numpy().tobytes()
                    target_embedding_bytestring = target_embedding.numpy().tobytes()
                    self._store_corpus_timestep(
                        source_token_id=source_token_id.item(),
                        target_token_id=target_token_id.item(),
                        source_embedding_bytestring=source_embedding_bytestring,
                        target_embedding_bytestring=target_embedding_bytestring,
                    )

    def _get_new_faiss_index(self):
        return faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))

    def _get_embedding_dtype(self):
        # TODO: add support for dtypes other than np.float32
        if self.embedding_dtype == "float32":
            embedding_dtype = np.float32
        else:
            raise ValueError(f"Unsupported dtype used '{self.embedding_dtype}'")

        return embedding_dtype

    def _increment_source_token_embeddings_offset(self):
        if self.embedding_batch_size < 1:
            raise ValueError("Please ensure `embedding_batch_size` is greater than 0.")
        self._embedding_table_offset += self.embedding_batch_size

    def _reset_source_token_embeddings_offset(self):
        self._embedding_table_offset = 0

    def _get_valid_embedding_offset_and_batch_size(self):
        error_first_sentence = (
            "Please ensure you are not modifying private class members. "
        )

        if not isinstance(self._embedding_table_offset, int):
            raise ValueError(
                f"{error_first_sentence}" "`_embedding_table_offset` must be an `int`."
            )
        if isinstance(self._embedding_table_offset, bool):
            raise ValueError(
                f"{error_first_sentence}"
                "`_embedding_table_offset` must be an `int` and not a `bool`."
            )
        if self._embedding_table_offset < 0:
            raise ValueError(
                f"{error_first_sentence}"
                "`_embedding_table_offset` must be positive or zero."
            )

        if not isinstance(self.embedding_batch_size, int):
            raise ValueError(
                f"{error_first_sentence}" "`embedding_batch_size` must be an `int`."
            )
        if isinstance(self.embedding_batch_size, bool):
            raise ValueError(
                f"{error_first_sentence}"
                "`embedding_batch_size` must be an `int` and not a `bool`."
            )
        if self.embedding_batch_size < 1:
            raise ValueError(
                f"{error_first_sentence}" "`embedding_batch_size` must be positive."
            )

        return self._embedding_table_offset, self.embedding_batch_size

    def _add_bytestrings_to_faiss_index(
        self, faiss_index, batch_ids, batch_bytestrings
    ):
        batch_embeddings_np = np.array(
            [
                np.frombuffer(embedding, dtype=self._get_embedding_dtype())
                for embedding in batch_bytestrings
            ]
        )
        batch_ids_np = np.array(batch_ids, dtype=np.int64)
        faiss.normalize_L2(batch_embeddings_np)
        faiss_index.add_with_ids(batch_embeddings_np, batch_ids_np)

    def build_source_index(self):
        faiss_index = self._get_new_faiss_index()
        source_token_ids = self._retrieve_all_source_token_ids()

        # One source token ID at a time
        for (source_token_id,) in (batches := tqdm(source_token_ids)):
            batches.set_description(
                f"Building index for source token ID {source_token_id}"
            )

            embedding_batches = self._retrieve_source_token_embeddings_batches(
                source_token_id
            )

            while rows := next(embedding_batches):
                batch_ids, batch_bytestrings = zip(*rows)
                self._add_bytestrings_to_faiss_index(
                    faiss_index, batch_ids, batch_bytestrings
                )

            bytestring = KNNStore._convert_faiss_index_to_bytestring(faiss_index)
            self._store_source_faiss_bytestring(source_token_id, bytestring)
            faiss_index.reset()

    def get_source_token_faiss_index(self, source_token_id):
        bytestring = self._retrieve_source_faiss_bytestring(source_token_id)
        if bytestring is not None:
            return faiss.deserialize_index(np.frombuffer(bytestring, dtype=np.uint8))
        return None

    def knn_source_faiss_index(self, source_token_id, source_embedding, k):
        faiss_index = self.get_source_token_faiss_index(source_token_id)

        # TODO: Write the faiss stuff to perform the k nearest neighbor search here
        # and return the list of ids

    def knn_get_logits(self):
        # TODO: Figure out how this all comes together. Need to review math. It's something like
        # calling knn_source_faiss_index() above and then calling build_target_faiss_index(), then
        # searching that index with the target and getting the top k matching target tokens, then
        # going to the math to interpolate with existing model. that will be another class that
        # composes this probably, KNNOperator or something.
        return True

    def build_target_datastore(
        self,
        encoder_input_ids,
        encoder_last_hidden_state,
        c=None,
    ):
        """Builds one target datastore faiss index for each sequence in the batch
        TODO: ROY: Finish this docstring
        """

        c = c if c is not None else self.c

        batch_size = encoder_input_ids.shape[0]
        self.target_datastore = [None] * batch_size

        for index in (batches := tqdm(range(batch_size))):
            batches.set_description(f"Building target datastore batch {index}")

            queries = {}

            # Gather faiss indices for each source_token_id in the sequence along with the queries for each
            for source_token_id, source_embedding in zip(
                encoder_input_ids[index], encoder_last_hidden_state[index]
            ):
                source_token_id = source_token_id.item()

                if queries.get(source_token_id, None) is None:
                    faiss_index = self.get_source_token_faiss_index(source_token_id)
                    if faiss_index is not None:
                        queries[source_token_id] = KNNStore.__FaissQueries__(
                            faiss_index=faiss_index,
                            embedding_dim=self.embedding_dim,
                            embedding_dtype=self._get_embedding_dtype(),
                            k=c,
                        )
                    else:
                        queries[source_token_id] = "no_index"

                if isinstance(queries[source_token_id], KNNStore.__FaissQueries__):
                    queries[source_token_id].add_query(source_embedding)

            unique_source_token_ids = queries.keys()

            if len(unique_source_token_ids) > 0:
                self.target_datastore[index] = KNNStore.__FaissQueries__(
                    faiss_index=self._get_new_faiss_index(),
                    embedding_dim=self.embedding_dim,
                    embedding_dtype=self._get_embedding_dtype(),
                )

            # Run bulk queries against faiss indices for each source token
            for source_token_id in unique_source_token_ids:
                # TODO: ROY: Parameterize based on preferences / environment the `use_gpu` flag
                if isinstance(queries[source_token_id], KNNStore.__FaissQueries__):
                    _, embedding_ids = queries[source_token_id].run(use_gpu=True)
                    unique_embedding_ids = np.unique(embedding_ids.flatten())
                    rows = self._retrieve_target_bytestrings(
                        unique_embedding_ids[unique_embedding_ids > 0].tolist()
                    )
                    batch_ids, batch_bytestrings = zip(*rows)
                    self._add_bytestrings_to_faiss_index(
                        self.target_datastore[index].faiss_index,
                        batch_ids,
                        batch_bytestrings,
                    )

    def search_target_datastore(
        self,
        decoder_last_hidden_state: torch.FloatTensor,
        k: int,
        unfinished_sequences: torch.LongTensor,
        pad_token_id: int = None,
        return_probs: bool = None,
        vocab_dim: int = None,
        temperature: float = None,
    ):
        """Returns the top k target tokens from datastore.
        TODO: ROY: Finish this docstring
        """
        embedding_dtype = self._get_embedding_dtype()

        # TODO: ROY: Try moving all of these tensors to the GPU For the search computations
        # then taking them back off (detaching) at the end

        batch_l2_distances = np.empty((0, k), dtype=embedding_dtype)
        batch_target_token_ids = np.empty((0, k), dtype=np.int64)

        pad_token_id = pad_token_id if pad_token_id is not None else 0

        for query_embedding, faiss_queries, sequence_is_unfinished in zip(
            decoder_last_hidden_state,
            self.target_datastore,
            unfinished_sequences == 1,
        ):
            if sequence_is_unfinished and faiss_queries is not None:
                faiss_queries.add_query(query_embedding)

                # TODO: ROY: Parameterize `use_gpu` based on whether GPU is available
                l2_distances, embedding_ids = faiss_queries.run(k=k, use_gpu=True)

                print(l2_distances, embedding_ids)

                target_token_ids = np.array(
                    self._retrieve_target_token_ids(tuple(embedding_ids[0])),
                    dtype=np.int64,
                ).reshape((1, -1))

                faiss_queries.clear_queries()
            else:
                # Cut down on computational complexity for finished sequences
                l2_distances = np.zeros((1, k), dtype=embedding_dtype)
                target_token_ids = np.full(
                    (1, k), fill_value=pad_token_id, dtype=np.int64
                )

            batch_l2_distances = np.concatenate(
                (batch_l2_distances, l2_distances), axis=0
            )
            batch_target_token_ids = np.concatenate(
                (batch_target_token_ids, target_token_ids), axis=0
            )

        if not return_probs:
            return batch_l2_distances, batch_target_token_ids

        if vocab_dim is None:
            raise ValueError(
                "Missing required parameter `vocab_dim` necessary for calculating logits."
            )

        if temperature is None:
            raise ValueError(
                "Missing required parameter `temperature` necessary for calculating logits."
            )

        batch_size = batch_l2_distances.shape[0]

        # shape (batch_size, k, vocab_dim)
        one_hot_tokens = np.zeros(
            (batch_size, k, vocab_dim), dtype=self._get_embedding_dtype()
        )

        for i in range(batch_size):
            for j in range(k):
                one_hot_tokens[i, j, batch_target_token_ids[i, j]] = (
                    # any token IDs coming back from faiss as -1 should not be considered
                    1
                    if batch_target_token_ids[i, j] != -1
                    else 0
                )

        # TODO: ROY: Investigate whether it would work to strip away the np.exp( part here
        # and just return scores (I don't think so--normalization of values would be weird)

        # shape (batch_size, k)
        exp_term = np.exp(-batch_l2_distances / temperature)

        # Replace any infinitesimal or zero values in `exp_term` with epsilon
        epsilon = 1e-7
        exp_term[exp_term < epsilon] = epsilon

        # shape (batch_size, k, vocab_dim)
        V = one_hot_tokens * exp_term.reshape(batch_size, k, 1)

        # shape (batch_size, 1, 1)
        Z = np.sum(exp_term, axis=1).reshape(batch_size, 1, 1)

        # shape (batch_size, k, vocab_dim)
        knn_probs_per_candidate = V / Z

        # `knn_probs` has shape (batch_size, vocab_dim)
        knn_probs = np.sum(knn_probs_per_candidate, axis=1)

        return knn_probs

    def validate(self):
        """Validate the KNNStore instance.

        Verifies that the instances is of type KNNStore and that the basic required attributes
        are in place. Raises an exception when invalid.
        """

        if not isinstance(self, KNNStore) or not (
            hasattr(self, "embedding_dim")
            and hasattr(self, "embedding_dtype")
            and hasattr(self, "embedding_batch_size")
            and hasattr(self, "target_batch_size")
            and hasattr(self, "c")
        ):
            raise ValueError(
                "Please sure the KNNStore instance is valid and properly constructed."
            )

    #
    # Abstract methods that must be implemented in subclass
    #

    @abstractmethod
    def _initialize_database(self, **kwargs):
        """Initialize DB. This is an abstract method.

        This function initializes the DB with the tables required for the KNN store to run. This
        includes four tables:

        - Table 1: Configuration
        - Table 2: Timesteps. Source and target token IDs and embeddings per timestep
        - Table 3: Faiss indices storing encoder embeddings across each source token ID

        Table 1: Configuration key/value pairs
            >Default table name is `knn_store_config`
            >In code, known as `self.configuration_table_name`
            - name:
                String. Name of the configuration. Primary key.
            - value:
                String. Value of the configuration.

        Table 2: Source and target token IDs and embeddings per timestep
            >Default table name is `knn_store_embedding`
            >In code, known as `self.embedding_table_name`
            - id:
                Source-side ID. System-generated. Represents a unique source token occurrence
                within the corpus. The aligned target token ID and target embedding can be
                referenced by this universal ID as well.

            - source_token_id:
                The token type ID (kind of token) at this location in the source. Depends on
                the tokenizer used upstream. Can be used as a way to group source embeddings,
                target tokens, and target embeddings by the source token type.

            - target_token_id:
                The token type ID (kind of token) of the target token that was aligned to
                the source token during the alignment process.

            - source_embedding:
                The embedding of the source token at this position. This assignment depends on
                the upstream embeddings model and how the output is generated, but usually it is
                the embedding created by the encoder in an encoder/decoder transformer architecture.

            - target_embedding:
                The embedding of the target token at this position. Usually the embedding from
                the last hidden state of the decoder at the timestep t of this particular
                position in the corpus, taking into account the whole source, and all target
                tokens up to the point t in the sequence.


        Table 3: Faiss indices storing encoder embeddings across each source token ID
            >Default table name is `knn_store_faiss_index`
            >In code, known as `self.faiss_cache_table_name`
            - source_token_id:
                The token type ID of the source token for which the list of embeddings are being
                pulled for vector search.
            - faiss_index:
                The byte data for a serialized FAISS index using `faiss.serialize_index(index)`.


        Note: Each of these tables must be implemented according to the type of database chosen
              for the subclass.

        Note: Source tokens without a corresponding entry in `alignments` property of input batch
              to `KNNStore.ingest(input_batch)` (in other words, source tokens that were not able to be
              aligned to target tokens) will be ignored and their related embeddings will not be stored.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            **kwargs:
                Keyword arguments needed to initialize the DB.
        """
        raise NotImplementedError(
            "Make sure to implement `_initialize_database` in a subclass and pass any necessary "
            "`**kwargs` for DB initialization to the base class constructor when constructing "
            "the subclass."
        )

    @abstractmethod
    def _store_corpus_timestep(
        self,
        source_token_id,
        target_token_id,
        source_embedding_bytestring,
        target_embedding_bytestring,
    ):
        """Store source and target token IDs and embeddings for single timestep. This is an abstract method.

        Args:
            source_token_id (int):
            target_token_id (int):
            source_embedding_bytestring (bytes):
            target_embedding_bytestring (bytes):

        Stores the following in Table 2 in the DB:

        source_token_id (int),
        target_token_id (int),
        source_embedding (blob/bytea),
        target_embedding (blob/bytea)

        Table 2: Source and target token IDs and embeddings per timestep

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.
        """
        raise NotImplementedError(
            "Make sure to implement `_store_corpus_timestep` in a subclass."
        )

    @abstractmethod
    def _retrieve_all_source_token_ids(self):
        """Retrieve all source token IDs from Table 2. This is an abstract method.

        Retrieves `source_token_id` across all rows in Table 2 in the DB.

        Table 2: Source and target token IDs and embeddings per timestep

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Returns:
            tuple(int): All source token IDs stored in Table 2.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_all_source_token_ids` in a subclass."
        )

    @abstractmethod
    def _retrieve_source_token_embeddings_batches(self, source_token_id):
        """Retrieves one batch of source token embeddings from the DB. This is an abstract method.

        GENERATOR FUNCTION.

        Yields a single batch of `source_embedding` fields from Table 2. Retrieves one batch
        according to self._embedding_table_offset and self.embedding_batch_size. Usually this
        will be implemented using something akin to `offset` and `limit` in the DB, and utizing
        an `order by` clause to ensure the offset and limit remain meaningful between function calls.

        Note: Must call `self._increment_source_token_embeddings_offset()` before yielding each batch.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            source_token_id (int): Source token ID for which embeddings are retrieved.

        Yields:
            tuple((int, bytes)):
                Tuple of two-element tuples. First element in each two-element tuple is the ID of the
                individual timestep stored in Table 2. Second element in each two-element tuple is the
                bytestring corresponding to the last encoder hidden state for source token at timestep.
                When no more data exists, should yield an empty tuple.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_source_token_embeddings_batches` in a subclass."
        )

    @abstractmethod
    def _store_source_faiss_bytestring(self, source_token_id, bytestring):
        """Stores faiss index for source embeddings across one source token ID. This is an abstract method.

        Stores the faiss index represented in the `bytestring` parameter in Table 3, overwriting any
        previous faiss index bytestring stored for this particular source_token_id. This is done using
        a DB transaction, so an index will only be removed if it is certainly being replaced by another.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            source_token_id (int):
                Source token ID for this source embedding index.
            bytestring (bytes):
                Bytes that make up the serialized version of the faiss index for this source token ID
        """
        raise NotImplementedError(
            "Make sure to implement `_store_source_faiss_bytestring` in a subclass."
        )

    @abstractmethod
    def _retrieve_source_faiss_bytestring(self, source_token_id):
        """Retrieves bytestring of serialized faiss index containing all source embeddings for
        a given source token. This is an abstract method.

        Retrieves serialized faiss index corresponding to the given `source_token_id` as a
        bytestring from the DB. Will return just the bytestring, not the row returned from
        the DB connection utility.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            source_token_id (int):
                Source token ID for this source embedding faiss index.

        Returns:
            bytes: Bytestring of faiss index corresponding to `source_token_id`.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_source_faiss_bytestring` in a subclass."
        )

    @abstractmethod
    def _retrieve_target_bytestrings(self, embedding_ids):
        """Retrieves target token embeddings corresponding to a list of Table 2 IDs.

        Retrieves all target token embeddings according to a list of Table 2 IDs (`embedding_ids`).
        The format of the return should be a tuple of rows where each row is a two-element tuple:

        Ex: ((embedding_id1, target_embedding1), (embedding_id2, target_embedding2))

        If no data is found for the given `embedding_ids`, an empty tuple (`()`) should be returned.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            embedding_ids (list):
                Table 2 row IDs for which to retrieve the `target_embedding` values.

        Returns:
            tuple(tuple(int, bytes)): A tuple of two-element tuples, each containing the timestep /
            embedding ID and the bytestring of the faiss index respectively.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_target_bytestrings` in a subclass."
        )

    @abstractmethod
    def _retrieve_target_token_ids(self, embedding_ids):
        """Retrieves target token IDs to a list of Table 2 IDs.

        Retrieves all target token IDs according to a list of Table 2 IDs (`embedding_ids`).
        The format of the return should be a tuple of integer target token IDs:

        Ex: (target_token_id1, target_token_id2, ...)

        If no data is found for the given `embedding_ids`, an empty tuple (`()`) should be returned.

        Note: No database implementation is given. Use one of the subclasses for a particular
        type of database.

        Args:
            embedding_ids (list):
                Table 2 row IDs for which to retrieve the `target_token_id` values.

        Returns:
            tuple(int): A tuple of integer target token IDs.
        """
        raise NotImplementedError(
            "Make sure to implement `_retrieve_target_token_ids` in a subclass."
        )

    class __FaissQueries__(dict):
        """Helper to contain one faiss index with queries.

        Attributes:
            faiss_index (faiss.swigfaiss_avx2.IndexIDMap):
                The fully initialized FAISS index stored on the CPU with vectors preloaded. Required for
                construction of `__FaissQueries__` object. Note: only CPU-stored faiss indices should be
                passed, as the index will be moved to the GPU at query time and then removed after.
            embedding_dim (int):
                The dimensionality of the query embeddings. Required for construction of `__FaissQueries__` object.
            embedding_dtype (type):
                The data type of the query embeddings. Defaults to `numpy.float32`.
            queries (ndarray(`embedding_dtype`)):
                Array of size (N, `embedding_dim`) where N is the number of query embeddings added. This will be
                directly used as input to `faiss_index.search()`.
            k (int):
                The number of nearest neighbors to return data for when running queries against the stored
                `faiss_index`. Defaults to 3.
        """

        def __init__(
            self, faiss_index=None, embedding_dim=None, embedding_dtype=None, k=None
        ):
            """Initialize a faiss index queries container.

            Args:
                faiss_index (faiss.swigfaiss_avx2.IndexIDMap):
                    The fully initialized FAISS index with vectors preloaded. Note: only CPU-stored faiss
                    indices should be passed, as the index will be moved to the GPU at query time and then
                    removed after.
                embedding_dim (int):
                    The dimensionality of the query embeddings. Required for construction of `__FaissQueries__` object.
                embedding_dtype (type):
                    The data type of the query embeddings. Defaults to `numpy.float32`.
                k (int):
                    The number of nearest neighbors to return data for when running queries against
                    the stored `faiss_index`. Defaults to 3.
            """
            if faiss_index is None:
                raise ValueError("Missing required parameter `faiss_index`.")

            if embedding_dim is None:
                raise ValueError("Missing required parameter `embedding_dim`.")

            self.faiss_index = faiss_index

            self.embedding_dim = embedding_dim
            self.embedding_dtype = (
                embedding_dtype if embedding_dtype is not None else np.float32
            )

            self.k = k if k is not None else 3

            self.queries = np.empty((0, self.embedding_dim), dtype=self.embedding_dtype)

        def add_query(self, query_embedding):
            """Add one embedding to the list of query embeddings for this faiss index.

            Args:
                query_embedding (ndarray):
                    1D array of size (`embedding_dim`) to be concatenated with existing queries. Data type is
                    determined by attribute `embedding_dtype`, set during construction.
            """
            if len(query_embedding.shape) > 1:
                raise ValueError(
                    "Parameter `query_embedding` (ndarray) must have only one dimension."
                )

            if query_embedding.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Parameter `query_embedding` (ndarray) of dimension {query_embedding.shape[0]} "
                    f"does not match configured `embedding_dim` of {self.embedding_dim}. Please "
                    f"only pass query embeddings of dimension {self.embedding_dim}."
                )

            self.queries = np.concatenate(
                (self.queries, query_embedding[np.newaxis, :])
            )

        def clear_queries(self):
            """Clear all queries from the FaissQueries instance."""
            self.queries = np.empty((0, self.embedding_dim), dtype=self.embedding_dtype)

        def run(self, k=None, use_gpu=None):
            """Run all queries currently stored in the container.

            Runs all queries stored in `queries` against `faiss_index`.

            Args:
                k (int):
                    The number of nearest neighbors for which to return data when running queries against
                    the stored `faiss_index`. Defaults to `self.k` on the object, which defaults to 3
                    when not specified at construction time.
                use_gpu (bool):
                    Whether to place the index on the GPU prior to searching. This also implies that the
                    index is automatically deallocated (by calling `faiss_index.reset()`) after the search
                    is complete.

            Returns:
                tuple(ndarray, ndarray):
                    A tuple with first element containing the matrix of L2 distances from the query vector
                    to the neighbor at that column for the query at that row. The second element is the
                    matrix of IDs for the neighbor at that column for the query at that row.
            """
            k = k if k is not None else self.k

            # use_gpu = use_gpu if use_gpu is not None else False

            # TODO: ROY: `faiss-gpu` package is unreliable for newer CUDA versions. Need to use the
            # wheel, but that means figuring out a place to store the wheel and how to integrate it
            # into the Hugging Face setup script. For now, run faiss on CPU and complete testing
            # and debugging, face this problem last.
            use_gpu = False

            if use_gpu:
                # TODO: ROY: Expand this to support multiple GPUs / GPU array
                # Should be something like the following:
                # gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            else:
                faiss_index = self.faiss_index

            distance, ids = faiss_index.search(self.queries, k)

            if use_gpu:
                faiss_index.reset()

            return distance, ids


# Allowed kwargs passed to `sqlite.connect()`
SQLITE_CONNECT_KWARGS = [
    "cached_statements",
    "check_same_thread",
    "database",
    "detect_types",
    "factory",
    "isolation_level",
    "timeout",
    "uri",
]


def extract_key_value_pairs_from_dict(original_dict, subset_keys):
    subset = {}
    keys = [key for key in original_dict.keys()]
    for key in keys:
        if key in subset_keys and original_dict.get(key, None) is not None:
            subset[key] = original_dict.pop(key)
    return subset


@dataclass
class KNNStoreSQLite(KNNStore):
    """KNN-MT embeddings store for SQLite.

    Attributes:
        sqlite_connect_kwargs (dict):

    """

    schema = "public"

    def __init__(
        self,
        embedding_dim=None,
        table_prefix=None,
        configuration_table_stem=None,
        embedding_table_stem=None,
        faiss_cache_table_stem=None,
        embedding_batch_size=None,
        target_batch_size=None,
        embedding_dtype=None,
        c=None,
        **kwargs,
    ):
        """Initializes KNNStore instance.

        Passes relevant `**kwargs` to `sqlite3.connect()` for initialization or restoriation
        of the DB. To initialize in the simplest form, pass `database="your-db-path.db"` and
        a SQLite DB will be either opened or created at "your-db-path.db". See the docs for
        Python's implementation of SQLite here: https://docs.python.org/3/library/sqlite3.html

        Note: The user of `:memory:` as the `database` parameter for `sqlite3.connect()` is
        allowed but not recommended for this use case, as the size of the DB can grow large
        quite quickly when storing high-dimensionality embeddings.

        Args:
            embedding_dim (int):
            table_prefix (str):
            configuration_table_stem (str):
            embedding_table_stem (str):
            faiss_cache_table_stem (str):
            embedding_batch_size (int):
            target_batch_size (int):
            embedding_dtype (str):
            c (int):
            **kwargs (dict):
        """

        self.sqlite_connect_kwargs = extract_key_value_pairs_from_dict(
            kwargs, SQLITE_CONNECT_KWARGS
        )

        if len(self.sqlite_connect_kwargs) < 1:
            raise ValueError(
                "Please specify keyword arguments to intialize database during construction of `KNNStoreSQLite` instance."
            )

        super(KNNStoreSQLite, self).__init__(
            embedding_dim=embedding_dim,
            table_prefix=table_prefix,
            configuration_table_stem=configuration_table_stem,
            embedding_table_stem=embedding_table_stem,
            faiss_cache_table_stem=faiss_cache_table_stem,
            embedding_batch_size=embedding_batch_size,
            target_batch_size=target_batch_size,
            embedding_dtype=embedding_dtype,
            c=c,
        )

    @staticmethod
    def _validate_table_name(table_name):
        safe_table_name_pattern = r"^[\p{L}_][\p{L}\p{N}@$#_]{0,127}$"
        if not re.match(safe_table_name_pattern, table_name):
            raise ValueError(f"Invalid table name supplied: '{table_name}'.")
        return table_name

    def _get_sqlite_connection(self):
        return sqlite3.connect(**self.sqlite_connect_kwargs)

    def _initialize_database(self):
        """Initialize database for SQLite"""

        print(
            f"Creating SQLite database using configuration: {self.sqlite_connect_kwargs}."
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        # Prevent SQL injections to table names
        valid_configuration_table_name = KNNStoreSQLite._validate_table_name(
            self.configuration_table_name
        )
        valid_embedding_table_name = KNNStoreSQLite._validate_table_name(
            self.embedding_table_name
        )
        valid_faiss_cache_table_name = KNNStoreSQLite._validate_table_name(
            self.faiss_cache_table_name
        )

        print(
            f"Creating table '{valid_configuration_table_name}' if it does not exist."
        )
        create_configuration_table_query = (
            f"create table if not exists {valid_configuration_table_name} ( "
            "   name text not null primary key, "
            "   value text "
            ");"
        )
        cur.execute(create_configuration_table_query)
        con.commit()

        print(
            f"Loading any past configurations from table '{valid_configuration_table_name}."
        )
        load_configurations_query = (
            f"select name, value from {valid_configuration_table_name};"
        )
        cur.execute(load_configurations_query)
        rows = cur.fetchall()

        for name, value in rows:
            if value != "None":
                if name == "embedding_dtype":
                    self.embedding_dtype = value
                elif name == "embedding_dim":
                    self.embedding_dim = int(value)

        if self.embedding_dim is None:
            raise ValueError("Missing required parameter `embedding_dim`.")

        print(f"Upserting configurations in '{valid_configuration_table_name}'")
        upsert_embedding_dtype_query = (
            f"insert into {valid_configuration_table_name} (name, value) "
            "values ('embedding_dtype', ?) "
            "on conflict(name) do update set value = ?;"
        )
        upsert_embedding_dim_query = (
            f"insert into {valid_configuration_table_name} (name, value) "
            "values ('embedding_dim', ?) "
            "on conflict(name) do update set value = ?;"
        )

        cur.execute(
            upsert_embedding_dtype_query,
            (
                self.embedding_dtype,
                self.embedding_dtype,
            ),
        )
        cur.execute(
            upsert_embedding_dim_query,
            (
                str(self.embedding_dim),
                str(self.embedding_dim),
            ),
        )
        con.commit()

        print(f"Creating table '{valid_embedding_table_name}' if it does not exist.")
        create_embedding_table_query = (
            f"create table if not exists {valid_embedding_table_name} ( "
            "    id integer primary key autoincrement, "
            "    source_token_id integer, "
            "    target_token_id integer, "
            "    source_embedding blob, "
            "    target_embedding blob "
            ");"
        )
        cur.execute(create_embedding_table_query)
        con.commit()

        print(f"Creating table '{valid_faiss_cache_table_name}' if it does not exist.")
        create_faiss_cache_table_query = (
            f"create table if not exists {valid_faiss_cache_table_name} ( "
            "    source_token_id integer not null unique, "
            "    faiss_index blob "
            ");"
        )
        cur.execute(create_faiss_cache_table_query)
        con.commit()

        cur.execute(f"select name, value from {valid_configuration_table_name};")
        configurations = cur.fetchall()

        cur.close()
        con.close()

        print(f"Current {self.__class__.__name__} instance configurations:")
        print(configurations)

    def _store_corpus_timestep(
        self,
        source_token_id,
        target_token_id,
        source_embedding_bytestring,
        target_embedding_bytestring,
    ):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        insert_embedding_query = (
            f"insert into {valid_embedding_table_name} (source_token_id, target_token_id, source_embedding, target_embedding) "
            "values (?, ?, ?, ?);"
        )

        cur.execute(
            insert_embedding_query,
            (
                source_token_id,
                target_token_id,
                source_embedding_bytestring,
                target_embedding_bytestring,
            ),
        )
        con.commit()

        cur.close()
        con.close()

    def _retrieve_all_source_token_ids(self):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        # Get unique source token IDs to iterate over
        cur.execute(
            f"select distinct source_token_id from {valid_embedding_table_name};"
        )

        source_token_ids = cur.fetchall()

        cur.close()
        con.close()

        return source_token_ids

    def _retrieve_source_token_embeddings_batches(self, source_token_id):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        self._reset_source_token_embeddings_offset()

        while self._embedding_table_offset == 0 or len(rows) > 0:
            valid_embedding_table_offset, valid_embedding_batch_size = (
                self._get_valid_embedding_offset_and_batch_size()
            )

            source_embedding_query = (
                "select id, source_embedding "
                f"from {valid_embedding_table_name} "
                "where source_token_id = ? and target_token_id is not null "
                "order by id "
                f"limit {valid_embedding_batch_size} "
                f"offset {valid_embedding_table_offset};"
            )

            cur.execute(
                source_embedding_query,
                (source_token_id,),
            )
            rows = cur.fetchall()
            self._increment_source_token_embeddings_offset()
            yield rows

        cur.close()
        con.close()

    def _store_source_faiss_bytestring(self, source_token_id, bytestring):
        valid_faiss_cache_table_name = self._validate_table_name(
            self.faiss_cache_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        cur.execute(
            f"delete from {valid_faiss_cache_table_name} where source_token_id = ?;",
            (source_token_id,),
        )
        cur.execute(
            f"insert into {valid_faiss_cache_table_name} (source_token_id, faiss_index) values (?, ?);",
            (
                source_token_id,
                bytestring,
            ),
        )
        con.commit()

        cur.close()
        con.close()

    def _retrieve_source_faiss_bytestring(self, source_token_id):
        valid_faiss_cache_table_name = self._validate_table_name(
            self.faiss_cache_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        cur.execute(
            f"select faiss_index from {valid_faiss_cache_table_name} where source_token_id = ?;",
            (int(source_token_id),),
        )

        result = cur.fetchall()

        cur.close()
        con.close()

        if len(result) < 1 or len(result[0]) < 1:
            return None

        bytestring = result[0][0]

        return bytestring

    def _retrieve_target_bytestrings(self, embedding_ids):
        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        placeholders = len(embedding_ids) * "?"

        cur.execute(
            f"select id, target_embedding from {valid_embedding_table_name} where id in ({','.join(placeholders)});",
            embedding_ids,
        )
        rows = cur.fetchall()

        cur.close()
        con.close()

        return rows

    def _retrieve_target_token_ids(self, embedding_ids):
        # TODO: ROY: Finish docstring
        embedding_ids_len = (
            len(embedding_ids) if isinstance(embedding_ids, tuple) else 0
        )

        if embedding_ids_len < 1:
            return ()

        valid_embedding_table_name = self._validate_table_name(
            self.embedding_table_name
        )

        con = self._get_sqlite_connection()
        cur = con.cursor()

        unique_embedding_ids = tuple(set(embedding_ids))
        placeholders = len(unique_embedding_ids) * "?"

        cur.execute(
            f"select id, target_token_id from {valid_embedding_table_name} where id in ({','.join(placeholders)});",
            tuple(int(embedding_id) for embedding_id in unique_embedding_ids),
        )

        rows = cur.fetchall()

        target_token_dict = {
            embedding_id: target_token_id for (embedding_id, target_token_id) in rows
        }

        target_token_ids = tuple(
            target_token_dict.get(embedding_id, -1) for embedding_id in embedding_ids
        )

        cur.close()
        con.close()

        return target_token_ids

    def validate(self):
        """Validate the KNNStoreSQLite instance.

        Verifies that the instances is of type KNNStoreSQLite and that the basic required attributes
        are in place. Raises an exception when invalid.
        """

        super(KNNStoreSQLite, self).validate()

        if not isinstance(self, KNNStoreSQLite) or not (
            hasattr(self, "sqlite_connect_kwargs")
        ):
            raise ValueError(
                "Please sure the KNNStoreSQLite instance is valid and properly constructed."
            )
