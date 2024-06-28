import numpy as np
import os
import pandas as pd
import regex as re
import subprocess
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.generation import KNNStoreSQLite
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel


class KNNDataset(Dataset):
    def __init__(self, path):
        self._df = pd.read_csv(path, dtype=str, header="infer")

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, index):
        return self._df.iloc[index][0], self._df.iloc[index][1]


def validate_required_params(params, accept_blank_string=False):
    """Validates a dict of required params and throws the first error found if any.

    Args:
        params: Dict containing required params in the form {'param_name': param_name, ...}
    """

    for name, param in params.items():
        if param is None or accept_blank_string or param == "":
            raise ValueError(f"Missing required parameter '{name}'.")


def dict_subset(original, subset_keys):
    subset = {}
    for key, value in original.items():
        if key in subset_keys:
            subset[key] = value
    return subset


"""Batching Module

Batching is approached from an end-to-end standpoint by task. A batch is expected to contain
attributes for both the input and the output of the task, and in this way reduces memory
footprint by never requiring any data duplication. This also increases usability, as data
that belongs together remains together, side-by-side in the batch.
"""


class LMBatch(object):
    """
    Represents a generic Language Model batch.

    Attributes:
        inputs (list): List of input sequences.
        inputs_raw (list): List of raw input sequences.
    """

    supported_params = ["inputs", "inputs_raw"]

    def __init__(self, **kwargs):
        """
        Initializes the LMBatch object with the provided keyword arguments.

        Args:
            **kwargs: Arbitrary keyword arguments. Supported arguments are listed in `supported_params`.
        """
        for param_name in LMBatch.supported_params:
            value = kwargs.get(param_name, None)
            if value is not None:
                setattr(self, param_name, value)

    def collate(self, tokenizer):
        """
        Collates batch of input sequences using the provided tokenizer.

        This method prepares the input sequences for model processing by tokenizing
        and padding them using the given tokenizer.

        Args:
            tokenizer: A tokenizer object used to tokenize the input sequences.

        Note:
            This method assumes that `inputs_raw` attribute contains raw input sequences
            to be tokenized and padded. After collation, the resulting sequences are stored
            in the `inputs` attribute of the LMBatch object.
        """
        self.inputs = tokenizer(self.inputs_raw, padding=True, return_tensors="pt")

    def get_quality_mask(self):
        """Return a list of boolean values as a mask for per-sequence input quality.

        There are two criteria for a quality sequence. One is that the source text must contain
        a reasonable distribution of word characters spread across at least two words. This is
        achieved through several sequential regular expresion patterns, and is currently
        configured to test for at least 5 word characters shared between at minimum 2 words.
        Additionally, it is notable that using this logic, any sentences with only one word will
        automatically be rejected.

        The second criteria is the source text must contain less than 65% non-word characters.

        Note:
            This method assumes that `inputs_raw` attribute contains raw input sequences
            to be graded for quality.
        """
        quality_mask = []

        non_word_threshold = 0.65

        patterns = [
            r"(\pL{1,}).+(\pL{4,})",
            r"(\pL{2,}).+(\pL{3,})",
            r"(\pL{3,}).+(\pL{2,})",
            r"(\pL{4,}).+(\pL{1,})",
        ]
        non_word = r"[^\pL]"

        for sentence in self.inputs_raw:
            regex_results = [
                1 if bool(re.search(pattern, sentence)) else 0 for pattern in patterns
            ]
            if sum(regex_results) == 0:
                quality_mask.append(False)
                continue

            chars = list(sentence)
            punct_mask = [1 if bool(re.search(non_word, char)) else 0 for char in chars]

            if sum(punct_mask) >= non_word_threshold * len(chars):
                quality_mask.append(False)
                continue

            quality_mask.append(True)

        return quality_mask

    def postprocess(self):
        """Postprocess the batch after running through whatever model.

        This will be different for every kind of batch. In a lot of cases
        it will house logic to deal with the values that are expected
        back from the model used for the given task.
        """
        raise NotImplementedError("'postprocess' method not implemented in base class.")


class EmbeddingsBatch(LMBatch):
    """
    Represents a batch for the task of getting last hidden states per input token from
    an encoder / decoder checkpoint from Hugging Face, inheriting from LMBatch.

    Attributes:
        alignments (list[dict]):
            Source token index to target token index alignments.
        inputs (list):
            List of input sequences.
        inputs_raw (list):
            List of raw input sequences.
        labels (list):
            List of target sequences.
        labels_raw (list):
            List of raw target sequences.
        encoder_last_hidden_state (list):
            Last hidden states of the encoder.
        token_hidden_states (list):
            Hidden states of each token.
        input_ids_masked (list[LongTensor]):
            Input token IDs with special tokens removed.
        label_ids_masked (list[LongTensor]):
            Target token IDs with special tokens removed.
        encoder_last_hidden_state_masked (list[FloatTensor]):
            Encoder last hidden states with hidden states for special source tokens removed.
        target_hidden_states_masked (list[FloatTensor]):
            Target last hidden states with hidden states for special target tokens removed.

    Note: labels are required for this task to be completed, as the last hidden states
    are intended to be retrieved from translation bitext where both source text and
    gold targets are known.
    """

    def __init__(self, **kwargs):
        """Initializes the EmbeddingsBatch object with the provided keyword arguments.

        Extends the supported_params attribute to include additional parameters required for translation tasks.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        EmbeddingsBatch.supported_params.extend(
            [
                "labels",
                "labels_raw",
                "encoder_last_hidden_state",
                "target_hidden_states",
            ]
        )
        super(EmbeddingsBatch, self).__init__(**kwargs)

    def collate(self, tokenizer):
        """Collates input and target sequences using a tokenizer.

        Args:
            tokenizer: Tokenizer object for processing input and target sequences.
        """
        self.inputs = tokenizer(
            self.inputs_raw,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        self.labels = tokenizer(
            text_target=self.labels_raw,
            padding=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

    def postprocess(self):
        """Generates postprocessed version of batch data with special character tokens masked out.

        Given:
            l = source sequence length
            m = target sequence length
            e = model embedding dimension

        Creates the following attributes on the class (m = particular sequence length):
            self.input_ids_masked                   (list[LongTensor(l)])
            self.label_ids_masked                   (list[LongTensor(m)])
            self.encoder_last_hidden_state_masked   (list[FloatTensor(m, e)])
            self.target_hidden_states_masked        (list[FloatTensor(m, e)])

        Note: It is expected that the batch has already been processed through a model and that the
        values for self.inputs, self.labels, self.encoder_last_hidden_state, and
        self.target_hidden_states are all available.
        """

        self.input_ids_masked = []
        self.label_ids_masked = []
        self.encoder_last_hidden_state_masked = []
        self.target_hidden_states_masked = []

        for (
            source_ids,
            source_mask,
            target_ids,
            target_mask,
            encoder_hidden_state,
            target_hidden_state,
        ) in zip(
            self.inputs.input_ids,
            self.inputs.special_tokens_mask,
            self.labels.input_ids,
            self.labels.special_tokens_mask,
            self.encoder_last_hidden_state,
            self.target_hidden_states,
        ):
            source_mask = np.invert(np.array(source_mask, dtype=bool))
            target_mask = np.invert(np.array(target_mask, dtype=bool))
            self.input_ids_masked.append(torch.LongTensor(source_ids[source_mask]))
            self.label_ids_masked.append(torch.LongTensor(target_ids[target_mask]))
            self.encoder_last_hidden_state_masked.append(
                torch.FloatTensor(encoder_hidden_state[source_mask])
            )
            self.target_hidden_states_masked.append(
                torch.FloatTensor(target_hidden_state[target_mask])
            )

    def generate_alignments(self, tokenizer, temp_filepath1=None, temp_filepath2=None):
        input_corpus_filepath = (
            temp_filepath1
            if temp_filepath1 is not None
            else "fast_align_temp_file1.tmp"
        )
        output_alignments_filepath = (
            temp_filepath2
            if temp_filepath2 is not None
            else "fast_align_temp_file2.tmp"
        )

        with open(input_corpus_filepath, "w") as corpus_file:
            for source_ids, target_ids in zip(
                self.input_ids_masked, self.label_ids_masked
            ):
                line = (
                    " ".join(tokenizer.convert_ids_to_tokens(source_ids))
                    + " ||| "
                    + " ".join(tokenizer.convert_ids_to_tokens(target_ids))
                    + "\n"
                )

                corpus_file.write(line)

        subprocess.run(
            f"./bin/fast_align -i {input_corpus_filepath} -d -o -v > {output_alignments_filepath}",
            shell=True,
        )

        with open(output_alignments_filepath) as alignments_file:
            alignments = []
            for line in alignments_file:
                pairs = re.split(r"\s+", line)
                line_alignments = {}

                for pair in pairs:
                    if not pair:
                        continue

                    split_pair = pair.split("-")
                    source_index = split_pair[0]
                    target_index = split_pair[1]

                    # Accept first alignment only for given source token
                    if not line_alignments.get(int(source_index), None):
                        line_alignments[int(source_index)] = int(target_index)

                alignments.append(line_alignments)

            self.alignments = alignments

        os.remove(input_corpus_filepath)
        os.remove(output_alignments_filepath)


"""Hugging Face parameter lists.

Add parameters herein if using specialized models / tokenizers / model configs
with parameters not already listed (but allowed by the underlying entity).
"""
HF_GENERATE_FUNCTION_PARAMS = ["logits_processor"]
HF_MODEL_FROM_PRETRAINED_PARAMS = [
    "pretrained_model_name_or_path",
    "model_args",
    "config",
    "state_dict",
    "cache_dir",
    "from_tf",
    "force_download",
    "resume_download",
    "proxies",
    "output_loading_info(bool,",
    "local_files_only(bool,",
    "revision",
    "trust_remote_code",
    "code_revision",
    "token",
]
HF_MODEL_CONFIG_PARAMS = [
    "pretrained_model_name_or_path",
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "revision",
    "return_unused_kwargs",
    "trust_remote_code",
    "name_or_path",
    "output_hidden_states",
    "output_attentions",
    "return_dict",
    "is_encoder_decoder",
    "is_decoder",
    "cross_attention_hidden_size",
    "add_cross_attention",
    "tie_encoder_decoder",
    "prune_heads",
    "chunk_size_feed_forward",
    "max_length",
    "min_length",
    "do_sample",
    "early_stopping",
    "num_beams",
    "num_beam_groups",
    "diversity_penalty",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "encoder_no_repeat_ngram_size",
    "bad_words_ids",
    "num_return_sequences",
    "output_scores",
    "return_dict_in_generate",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "remove_invalid_values",
    "architectures",
    "finetuning_task",
    "id2label",
    "label2id",
    "num_labels",
    "task_specific_params",
    "problem_type",
    "bos_token_id",
    "pad_token_id",
    "eos_token_id",
    "decoder_start_token_id",
    "sep_token_id",
    "torchscript",
    "tie_word_embeddings",
    "torch_dtype",
]
HF_TOKENIZER_PARAMS = [
    "pretrained_model_name_or_path",
    "model_max_length",
    "padding_side",
    "truncation_side",
    "chat_template",
    "model_input_names",
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
    "clean_up_tokenization_spaces",
    "split_special_tokens",
    "inputs",
    "config",
    "cache_dir",
    "force_download",
    "resume_download",
    "proxies",
    "revision",
    "subfolder",
    "use_fast",
    "tokenizer_type",
    "trust_remote_code",
    "src_lang",
    "tgt_lang",
]


class ModelFromCheckpoint(PreTrainedModel):
    """Model from Hugging Face checkpoint.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        config (AutoConfig): Configuration for the model.
        model (AutoModelForSeq2SeqLM): Instantiated model.
        custom_device (torch.device or str): Device for model computations.
        generate_kwargs (dict): Keyword arguments to be passed to the model.generate().
    """

    def __init__(self, checkpoint=None, use_cpu=False, **kwargs):
        """Initialize the adapter model and tokenizer with a Hugging Face checkpoint.

        Args:
            checkpoint:
                Specifies the string checkpoint of the model to load. Either a HF
                hub checkpoint or the path to the local model checkpoint directory.
            use_cpu:
                When true, tells the model to make the device 'cpu' even when a GPU is
                available. When false, the model well always attempt to use any available
                GPU(s) available in the runtime.
            **kwargs:
                Keyword arguments to be passed as necessary to the model, model config,
                and tokenizer initialization functions.

        """
        validate_required_params(dict(checkpoint=checkpoint))

        model_kwargs = dict_subset(kwargs, HF_MODEL_FROM_PRETRAINED_PARAMS)
        model_config_kwargs = dict_subset(kwargs, HF_MODEL_CONFIG_PARAMS)
        tokenizer_kwargs = dict_subset(kwargs, HF_TOKENIZER_PARAMS)
        self.generate_kwargs = dict_subset(kwargs, HF_GENERATE_FUNCTION_PARAMS)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)

        forced_bos_token_tgt_lang = kwargs.get("forced_bos_token_tgt_lang", None)
        if forced_bos_token_tgt_lang:
            model_config_kwargs["forced_bos_token_id"] = (
                self.tokenizer.convert_tokens_to_ids([forced_bos_token_tgt_lang])[0]
            )

        if not checkpoint:
            raise ValueError("Missing required parameter 'checkpoint'")
        self.config = AutoConfig.from_pretrained(
            checkpoint,
            **model_config_kwargs,
        )
        super(ModelFromCheckpoint, self).__init__(self.config)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, config=self.config, **model_kwargs
        )

        self.custom_device = (
            torch.device("cuda")
            if torch.cuda.is_available() and (not use_cpu)
            else "cpu"
        )
        self.model.to(self.custom_device)

        print(
            f"ModelFromCheckpoint: Loaded checkpoint '{checkpoint}'. Using device '{self.custom_device}'."
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Attention mask tensor.
            labels (torch.Tensor, optional): Labels tensor.

        Raises:
            NotImplementedError: This method is not implemented in the base class.
        """
        raise NotImplementedError(
            "ModelFromCheckpoint.forward not implemented in base class."
        )


class NLLBCheckpointOutput(object):
    """
    Represents the output of a Next-Language Learning Benchmark model checkpoint.

    Attributes:
        example_quality_mask (torch.Tensor): Mask indicating the quality of examples in the batch.
        batch_output_text (list): List of output texts generated by the model for each example in the batch.
    """

    def __init__(self, batch_output_text):
        """
        Initializes the NLLBCheckpointOutput object.

        Args:
            batch_output_text (list): List of output texts generated by the model for each example in the batch.
        """
        super(NLLBCheckpointOutput, self).__init__()
        self.batch_output_text = batch_output_text


class NLLBCheckpoint(ModelFromCheckpoint):
    """Wrapper to load NLLB checkpoint model"""

    def __init__(
        self,
        checkpoint,
        src_lang,
        tgt_lang,
        **kwargs,
    ):
        """Initialize the NLLB model and tokenizer with a Hugging Face checkpoint.

        Args:
            checkpoint:
                Specifies the string checkpoint of the model to load. Either a HF
                hub checkpoint or the path to the local model checkpoint directory.
            src_lang:
                The BCP-47 code for the source language.
            tgt_lang:
                The BCP-47 code for the source language.
            **kwargs:
                Keyword arguments to be passed as necessary to the model, model config,
                and tokenizer initialization functions.
        """
        super(NLLBCheckpoint, self).__init__(
            checkpoint,
            **(
                kwargs
                | dict(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    forced_bos_token_tgt_lang=tgt_lang,
                )
            ),
        )

    def forward(self, batch):
        batch.collate(tokenizer=self.tokenizer)
        batch_input_ids = batch.inputs.input_ids.to(self.custom_device)
        batch_input_attention_mask = batch.inputs.attention_mask.to(self.custom_device)

        print('self.generate_kwargs', self.generate_kwargs)
        output_ids = self.model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_input_attention_mask,
            **self.generate_kwargs,
        )

        print('got here')

        batch_output_text = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return NLLBCheckpointOutput(
            batch_output_text=batch_output_text,
        )


class NLLBEmbeddingsModel(NLLBCheckpoint):
    """Helper class to return autoregressive decoder embeddings from NLLB checkpoint."""

    def __init__(
        self,
        checkpoint,
        src_lang,
        tgt_lang,
        **kwargs,
    ):
        """Initialize the NLLB model and configure model to output hidden states.

        Args:
            checkpoint:
                Specifies the string checkpoint of the model to load. Either a HF
                hub checkpoint or the path to the local model checkpoint directory.
            src_lang:
                The BCP-47 code for the source language.
            tgt_lang:
                The BCP-47 code for the source language.
            **kwargs:
                Keyword arguments to be passed as necessary to the model, model config,
                and tokenizer initialization functions.
        """
        super(NLLBEmbeddingsModel, self).__init__(
            checkpoint,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            **(kwargs | dict(output_hidden_states=True)),
        )

        # Freeze base model (not necessary as the primary forward() operation is run using torch.no_grad())
        for param in self.model.base_model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        batch.collate(tokenizer=self.tokenizer)

        batch_input_ids = batch.inputs.input_ids.to(self.custom_device)
        batch_input_attention_mask = batch.inputs.attention_mask.to(self.custom_device)
        batch_label_ids = batch.labels.input_ids.to(self.custom_device)

        batch_size = batch.inputs.input_ids.size(0)

        target_hidden_states = torch.empty(
            (batch_size, 0, self.config.hidden_size), dtype=torch.float32
        ).to(self.custom_device)

        label_id_count = batch.labels.input_ids.size(1)

        self.model.eval()

        with torch.no_grad():
            for index in range(1, label_id_count + 1):
                # Forward pass model
                model_outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_input_attention_mask,
                    decoder_input_ids=batch_label_ids[:, :index].to(self.custom_device),
                )

                # Get all of the encoder representations on the first pass
                if index == 1:
                    encoder_last_hidden_state = (
                        # .detach not necessary but illustrates intent (would still work without torch.no_grad())
                        model_outputs.encoder_last_hidden_state.detach()
                    )

                # Get index of token currently being decoded
                token_rep_idx = index - 1

                # Get decoder representation
                # .detach() not necessary but illustrates intent (would still work without torch.no_grad())
                last_hidden_state = model_outputs.decoder_hidden_states[-1].detach()
                token_representation = last_hidden_state[:, np.newaxis, token_rep_idx]

                # Concatenate decoder last hidden state of current token
                target_hidden_states = torch.cat(
                    (target_hidden_states, token_representation), dim=1
                )

        batch.encoder_last_hidden_state = encoder_last_hidden_state.cpu()
        batch.target_hidden_states = target_hidden_states.cpu()


# GOAL USAGE


def main():
    checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    batch_size = 10

    # TODO: ROY: Rewrite this! We want to support HF datsets so we will use load_dataset(csv) or whatnot!
    dataset_path = "data/de-en-emea-medical-clean.csv"
    dataset = KNNDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=batch_size)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # NOTE: showing this construction for demonstration of desired under-the-hood functionality,
    # but these parameters will be passed directly to the KNNStore as below
    model = EmbeddingsModel(
        checkpoint=checkpoint,
        src_lang=src_lang,  # should be able to use these if necessary, but not require them
        tgt_lang=tgt_lang,  # (such as for models that just go from one source to one target)
    )

    knn_store = KNNStoreSQLite(
        checkpoint=checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        # embedding_dim=model.config.hidden_size,    <--- THIS SHOULD BE INFERRED FROM checkpoint NOW
        database="db/test_db.db",
    )

    # having initialized the store, we want to let them choose when to perform ingest like so
    knn_store.ingest(dataset, run_id="some_identifier_for_the_run")

    # ALSO note, we want the ingest to be resumable, thus the run_id to track and restore from using the DB

    # TODO: ROY: REPLACE all of this with the one call to knn_store.ingest() above!!!
    # The encapsulation should look professional. We may want to temper the OOP idea (though Python is steeped in OOP)
    # and go with a more functional programming approach, perhaps still in OOP form, that applies actions to whole arrays / sets of data at once
    count = 0
    for english, german in (batches := tqdm(loader)):
        batches.set_description(f"Processing {dataset_path} in batches of {batch_size}")

        batch = EmbeddingsBatch(inputs_raw=english, labels_raw=german)
        model(batch)
        batch.postprocess()
        batch.generate_alignments(tokenizer=model.tokenizer)
        knn_store.ingest(batch)

        if count > 10:
            break

    count += 1

    # TODO: ROY: Make this resumable. We could also use a run_id here, but the DB is actually capable of describing
    # whether or not all the source indices have been created for what is in the embedding table, so I think a more
    # automated approach her is in order.
    knn_store.build_source_index()

    # AFTER doing all of the above work, the below should "just work"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, src_lang=src_lang, tgt_lang=tgt_lang
    )

    output_ids = model.generate(
        **tokenizer(
            [
                "I am doing fine",
                "Hello, how are you",
                "What is the best antibiotic?",
                "Where are my pants?",
                "I like boobs.",
            ],
            padding='max_length',
            truncation=True,
            max_length=150,
            return_tensors="pt",
        ),
        knn_store=knn_store,
        knn_interpolation_coefficient=0.5,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids([tgt_lang])[0],
    )

    print(output_ids)
    print(tokenizer.batch_decode(output_ids))


if __name__ == "__main__":
    main()
