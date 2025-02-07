import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Union, Any
from abc import ABC, abstractmethod

import onnxruntime as ort
import torch
import torch.nn as nn
import openvino as ov
from openvino import InferRequest
from safetensors import safe_open
from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from transformers import AutoTokenizer
from gliner import GLiNER,GLiNERConfig
from gliner.modeling.base import GLiNERModelOutput, BaseModel, SpanModel
from gliner.data_processing import SpanProcessor, TokenProcessor, SpanBiEncoderProcessor
from gliner.data_processing.tokenizer import WordsSplitter
from gliner.decoding import SpanDecoder

class BaseVINOModel(ABC, nn.Module):
    def __init__(self, model_path: Path):
        super().__init__()
        self.core = ov.Core()
        self.compiled_model = self.core.compile_model(model_path, "AUTO")
        self.input_names = [name for input_layer in self.compiled_model.inputs 
                            for name in input_layer.get_names()]
        self.shared = False

    def prepare_inputs(self, inputs: Dict[str, torch.Tensor]) -> InferRequest:
        """
        Prepare inputs for VINO model inference.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of input names and tensors.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of input names and numpy arrays.
        """
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary of input names and tensors.")
        
        infer_request = self.compiled_model.create_infer_request()
        for key, tensor in inputs.items():
            if key not in self.input_names:
                warnings.warn(f"Input key '{key}' not found in VINO model's input names. Ignored.")
                continue
            input_tensor = ov.Tensor(array=tensor.numpy(), shared_memory=self.shared)
            infer_request.set_tensor(key, input_tensor)
        return infer_request

    @abstractmethod
    def forward(self, input_ids, attention_mask, **kwargs) -> Dict[str, Any]:
        """
        Abstract method to perform forward pass. Must be implemented by subclasses.
        """
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class SpanVINOModel(BaseVINOModel):
    
    async def async_request(self, request:InferRequest)-> ov.Tensor:
        request.start_async()
        request.wait()
        result = request.get_tensor("logits")
        return result
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                words_mask: torch.Tensor, text_lengths: torch.Tensor, 
                span_idx: torch.Tensor, span_mask: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for span model using ViNO inference.
        Synchronous

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            span_idx (torch.Tensor): Span indices tensor.
            span_mask (torch.Tensor): Span mask tensor.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, Any]: Model outputs.
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_mask': words_mask,
            'text_lengths': text_lengths,
            'span_idx': span_idx,
            'span_mask': span_mask
        }
        inference_output = self.compiled_model(inputs)
        outputs = GLiNERModelOutput(
            logits=inference_output['logits']
        )
        return outputs
    
    async def async_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                words_mask: torch.Tensor, text_lengths: torch.Tensor, 
                span_idx: torch.Tensor, span_mask: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """
        Forward pass for span model using VINO inference.
        Async version

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            attention_mask (torch.Tensor): Attention mask tensor.
            span_idx (torch.Tensor): Span indices tensor.
            span_mask (torch.Tensor): Span mask tensor.
            **kwargs: Additional arguments.
        
        Returns:
            Dict[str, Any]: Model outputs.
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'words_mask': words_mask,
            'text_lengths': text_lengths,
            'span_idx': span_idx,
            'span_mask': span_mask
        }
        inference_request = self.prepare_inputs(inputs)
        inference_output = await self.async_request(inference_request)
        outputs = GLiNERModelOutput(
            logits=inference_output.data
        )
        return outputs


class VinoGLiNER(GLiNER):
    """
    Custom Class to accept VINO models with the possibility to load
    them from huggingface hub or custom preprocessing.
    """
    def __init__(
        self,
        config: GLiNERConfig,
        model: Optional[Union[BaseModel, BaseVINOModel]] = None,
        tokenizer: Optional[Union[str, AutoTokenizer]] = None,
        words_splitter: Optional[Union[str, WordsSplitter]] = None,
        data_processor: Optional[Union[SpanProcessor, TokenProcessor]] = None,
        encoder_from_pretrained: bool = True
    ):
        """
        Initialize the GLiNER model.

        Args:
            config (GLiNERConfig): Configuration object for the GLiNER model.
            model (Optional[Union[BaseModel, BaseVINOModel]]): GLiNER model to use for predictions. Defaults to None.
            tokenizer (Optional[Union[str, AutoTokenizer]]): Tokenizer to use. Can be a string (path or name) or an AutoTokenizer instance. Defaults to None.
            words_splitter (Optional[Union[str, WordsSplitter]]): Words splitter to use. Can be a string or a WordsSplitter instance. Defaults to None.
            data_processor (Optional[Union[SpanProcessor, TokenProcessor]]): Data processor - object that prepare input to a model. Defaults to None.
            encoder_from_pretrained (bool): Whether to load the encoder from a pre-trained model or init from scratch. Defaults to True.
        """
        # Start the parent classes, but not the GLiNER class
        nn.Module.__init__(self)
        PyTorchModelHubMixin.__init__(self)
        self.config = config

        if tokenizer is None and data_processor is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        if words_splitter is None and data_processor is None:
            words_splitter = WordsSplitter(config.words_splitter_type)

        if config.span_mode == "token_level":
            raise NotImplementedError(f"No implementation for {config.span_mode} span_mode")
        else:
            if model is None:
                self.model = SpanModel(config, encoder_from_pretrained)
            else:
                self.model = model
            if data_processor is None:
                if config.labels_encoder is not None:
                    labels_tokenizer = AutoTokenizer.from_pretrained(config.labels_encoder)
                    self.data_processor = SpanBiEncoderProcessor(config, tokenizer, words_splitter, labels_tokenizer)
                else:
                    self.data_processor = SpanProcessor(config, tokenizer, words_splitter)
            else:
                self.data_processor = data_processor
            self.decoder = SpanDecoder(config)

        if config.vocab_size != -1 and config.vocab_size != len(
            self.data_processor.transformer_tokenizer
        ):
            warnings.warn(f"""Vocab size of the model ({config.vocab_size}) does't match length of tokenizer ({len(self.data_processor.transformer_tokenizer)}). 
                            You should to consider manually add new tokens to tokenizer or to load tokenizer with added tokens.""")

        if isinstance(self.model, BaseVINOModel):
            self.onnx_model = False
            self.vino_model = True
        else:
            self.onnx_model = False
            self.vino_model = False

        # to suppress an AttributeError when training
        self._keys_to_ignore_on_save = None
    
    @property
    def device(self):
        if self.onnx_model:
            providers = self.model.session.get_providers()
            if 'CUDAExecutionProvider' in providers:
                return torch.device('cuda')
            return torch.device('cpu')
        elif self.vino_model:
            return 'cpu'
        else:
            device = next(self.model.parameters()).device
            return device
    
    @torch.no_grad()
    def batch_predict_entities(
        self, texts, labels, flat_ner=True, threshold=0.5, multi_label=False
    ):
        """
        Predict entities for a batch of texts.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.

        Returns:
            The list of lists with predicted entities.
        """

        model_input, raw_batch = self.prepare_model_inputs(texts, labels)

        model_output = self.model(**model_input)[0]

        if not isinstance(model_output, torch.Tensor):
            model_output = torch.from_numpy(model_output)

        outputs = self.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][
                i
            ]
            end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append(
                    {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                )
            all_entities.append(entities)

        return all_entities
    
    async def async_batch_predict_entities(
        self, texts, labels, flat_ner=True, threshold=0.5, multi_label=False
    ):
        """
        Predict entities for a batch of texts. Async call.

        Args:
            texts (List[str]): A list of input texts to predict entities for.
            labels (List[str]): A list of labels to predict.
            flat_ner (bool, optional): Whether to use flat NER. Defaults to True.
            threshold (float, optional): Confidence threshold for predictions. Defaults to 0.5.
            multi_label (bool, optional): Whether to allow multiple labels per token. Defaults to False.

        Returns:
            The list of lists with predicted entities.
        """

        model_input, raw_batch = self.prepare_model_inputs(texts, labels)

        model_output = await self.model.async_forward(**model_input)

        if not isinstance(model_output[0], torch.Tensor):
            model_output = torch.from_numpy(model_output[0])

        outputs = self.decoder.decode(
            raw_batch["tokens"],
            raw_batch["id_to_classes"],
            model_output,
            flat_ner=flat_ner,
            threshold=threshold,
            multi_label=multi_label,
        )

        all_entities = []
        for i, output in enumerate(outputs):
            start_token_idx_to_text_idx = raw_batch["all_start_token_idx_to_text_idx"][
                i
            ]
            end_token_idx_to_text_idx = raw_batch["all_end_token_idx_to_text_idx"][i]
            entities = []
            for start_token_idx, end_token_idx, ent_type, ent_score in output:
                start_text_idx = start_token_idx_to_text_idx[start_token_idx]
                end_text_idx = end_token_idx_to_text_idx[end_token_idx]
                entities.append(
                    {
                        "start": start_token_idx_to_text_idx[start_token_idx],
                        "end": end_token_idx_to_text_idx[end_token_idx],
                        "text": texts[i][start_text_idx:end_text_idx],
                        "label": ent_type,
                        "score": ent_score,
                    }
                )
            all_entities.append(entities)

        return all_entities
    
    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        load_tokenizer: Optional[bool] = False,
        resize_token_embeddings: Optional[bool] = True,
        load_vino_model: Optional[bool] = False,
        vino_model_file: Optional[str] = "model.xml",
        compile_torch_model: Optional[bool] = False,
        session_options: Optional[ort.SessionOptions] = None,
        _attn_implementation: Optional[str] = None,
        max_length: Optional[int] = None,
        max_width: Optional[int] = None,
        post_fusion_schema: Optional[str] = None,
        **model_kwargs,
    ):
        """
        Load a pretrained model from a given model ID.

        Args:
            model_id (str): Identifier of the model to load.
            revision (Optional[str]): Specific model revision to use.
            cache_dir (Optional[Union[str, Path]]): Directory to store downloaded models.
            force_download (bool): Force re-download even if the model exists.
            proxies (Optional[Dict]): Proxy configuration for downloads.
            resume_download (bool): Resume interrupted downloads.
            local_files_only (bool): Use only local files, don't download.
            token (Union[str, bool, None]): Token for API authentication.
            map_location (str): Device to map model to. Defaults to "cpu".
            strict (bool): Enforce strict state_dict loading.
            load_tokenizer (Optional[bool]): Whether to load the tokenizer. Defaults to False.
            resize_token_embeddings (Optional[bool]): Resize token embeddings. Defaults to True.
            load_vino_model (Optional[bool]): Load VINO version of the model. Defaults to False.
            vino_model_file (Optional[str]): Filename for VINO model. Defaults to 'model.xml'.
            compile_torch_model (Optional[bool]): Compile the PyTorch model. Defaults to False.
            session_options (Optional[onnxruntime.SessionOptions]): ONNX Runtime session options. Defaults to None.
            **model_kwargs: Additional keyword arguments for model initialization.

        Returns:
            An instance of the model loaded from the pretrained weights.
        """
        # Newer format: Use "pytorch_model.bin" and "gliner_config.json"
        model_dir = Path(model_id)  # / "pytorch_model.bin"
        if not model_dir.exists():
            model_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        model_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "pytorch_model.bin")
        config_file = Path(model_dir) / "gliner_config.json"

        if load_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        else:
            if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            else:
                tokenizer = None
        with open(config_file, "r") as f:
            config_ = json.load(f)
        config = GLiNERConfig(**config_)

        if _attn_implementation is not None:
            config.encoder_config._attn_implementation = _attn_implementation
        if max_length is not None:
            config.max_len = max_length
        if max_width is not None:
            config.max_width = max_width
        if post_fusion_schema is not None:
            config.post_fusion_schema = post_fusion_schema
            print('Post fusion is set.')

        add_tokens = ["[FLERT]", config.ent_token, config.sep_token]

        if not load_vino_model:
            gliner = cls(config, tokenizer=tokenizer, encoder_from_pretrained=False)
            # to be able to load GLiNER models from previous version
            if (
                config.class_token_index == -1 or config.vocab_size == -1
            ) and resize_token_embeddings and not config.labels_encoder:
                gliner.resize_token_embeddings(add_tokens=add_tokens)
            if model_file.endswith("safetensors"):
                state_dict = {}
                with safe_open(model_file, framework="pt", device=map_location) as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
            gliner.model.load_state_dict(state_dict, strict=strict)
            gliner.model.to(map_location)
            if compile_torch_model and "cuda" in map_location:
                print("Compiling torch model...")
                gliner.compile()
            elif compile_torch_model:
                warnings.warn(
                    "It's not possible to compile this model putting it to CPU, you should set `map_location` to `cuda`."
                )
            gliner.eval()
        else:
            model_file = Path(model_dir) / vino_model_file
            if not os.path.exists(model_file):
                raise FileNotFoundError(
                    f"The VINO model can't be loaded from {model_file}."
                )
            if config.span_mode == "token_level":
                raise NotImplementedError(f"No implementation for {config.span_mode} mode")
            else:
                model = SpanVINOModel(model_file)

            gliner = cls(config, tokenizer=tokenizer, model=model)
            if (
                config.class_token_index == -1 or config.vocab_size == -1
            ) and resize_token_embeddings:
                gliner.data_processor.transformer_tokenizer.add_tokens(add_tokens)

            gliner.model.to(map_location)
        if (len(gliner.data_processor.transformer_tokenizer)!=gliner.config.vocab_size
                                                        and gliner.config.vocab_size!=-1):
            new_num_tokens = len(gliner.data_processor.transformer_tokenizer)
            _ = gliner.model.token_rep_layer.resize_token_embeddings(
                new_num_tokens, None
            )
        return gliner