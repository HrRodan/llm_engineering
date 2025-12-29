import os
import io
import base64
import torch
import getpass
from typing import Optional, Union, List, Dict, Any, Generator, Literal
from PIL import Image

# Hugging Face & PyTorch imports
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
    pipeline,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
)
from diffusers import AutoPipelineForText2Image
from IPython.display import display, Markdown, Audio

# API Key Retrieval
# Priority 1: Google Colab Userdata (for native Colab environment)
try:
    from google.colab import userdata  # type: ignore

    HF_TOKEN = userdata.get("HF_TOKEN")
except (ImportError, AttributeError, Exception):
    HF_TOKEN = None

# Priority 2: Environment Variables (local development, .env files)
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")

# Priority 3: Interactive Prompt (VS Code Colab extension, fallback)
if not HF_TOKEN:
    # Only prompt if we look like we are in an interactive session that might support it
    # and we haven't already tried to get it.
    try:
        HF_TOKEN = getpass.getpass("HF_TOKEN: ")
    except Exception:
        pass

if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=True)
    except Exception as e:
        print(f"Warning: Failed to login to Hugging Face: {e}")
else:
    print(
        "Warning: HF_TOKEN not found. Some models may fail to load if they require authentication."
    )


class HuggingFaceQuery:
    """
    A unified interface for interacting with Hugging Face models, supporting both
    low-level model access (for fine-grained control and quantization) and high-level
    pipelines (for ease of use and multi-modal tasks).
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quantization: Literal["4bit", "8bit", "none"] = "4bit",
        system_prompt: str = "You are a helpful AI assistant.",
        image_model: str = "stabilityai/sdxl-turbo",
        stt_model: str = "openai/whisper-medium.en",
        tts_model: str = "microsoft/speecht5_tts",
        image_to_text_model: str = "Salesforce/blip-image-captioning-base",
    ):
        """
        Initialize the HuggingFaceQuery instance.

        Args:
            model_name (str): The name of the LLM model to use for text generation.
            device (str): The device to run models on ('cuda' or 'cpu').
            quantization (str): Quantization mode ('4bit', '8bit', or 'none').
            system_prompt (str): Default system prompt for chat interactions.
            image_model (str): Default model for image generation.
            stt_model (str): Default model for speech-to-text (transcription).
            tts_model (str): Default model for text-to-speech.
            image_to_text_model (str): Default model for image captioning/analysis.
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.system_prompt = system_prompt

        # Task-specific model names
        self.image_model_name = image_model
        self.stt_model_name = stt_model
        self.tts_model_name = tts_model
        self.image_to_text_model_name = image_to_text_model

        # Lazy loading variables (debugging access)
        self._tokenizer = None
        self._model = None
        self._pipelines: Dict[str, Any] = {}

    @property
    def tokenizer(self):
        """Access the tokenizer, loading it if necessary."""
        if self._tokenizer is None:
            print(f"Loading tokenizer for {self.model_name}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self):
        """Access the low-level LLM model, loading it if necessary."""
        if self._model is None:
            self._load_text_model()
        return self._model

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Create the quantization configuration based on initialization settings."""
        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _load_text_model(self):
        """Load the text generation model with the specified quantization."""
        print(
            f"Loading model {self.model_name} on {self.device} with {self.quantization} quantization..."
        )

        quant_config = self._get_quantization_config()

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto" if self.device == "cuda" else None,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            if self.device == "cpu":
                self._model.to("cpu")  # Explicitly move to cpu if not using device_map

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def query(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 1000,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_full_text: bool = False,
    ) -> str:
        """
        Generate text response for a given prompt using low-level API.

        Args:
            user_prompt (str): The user's input/query.
            system_prompt (str, optional): Override default system prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            do_sample (bool): Whether to use sampling or greedy decoding.
            return_full_text (bool): If True, returns input + output. If False, returns only output.

        Returns:
            str: The generated text response.
        """
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

        outputs = self.model.generate(  # pyrefly: ignore
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not return_full_text:
            # Attempt to strip the prompt if we can leverage the chat template structure logic
            # Simplistic approach: find the last occurrence of "assistant" header if visible?
            # Since decode strips special tokens, we might just rely on splitting if we knew the format.
            # However, apply_chat_template usually creates a string.
            # For robustness, we'll strip the input length from the decoded output input_ids.
            output_tokens = outputs[0][input_ids.shape[-1] :]
            generated_text = self.tokenizer.decode(
                output_tokens, skip_special_tokens=True
            )

        return generated_text

    def query_stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 10000,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Stream text response for a given prompt to stdout/display.

        Args:
            user_prompt (str): The user's input/query.
            system_prompt (str, optional): Override default system prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            do_sample (bool): Whether to use sampling.
        """
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        streamer = TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")

        self.model.generate(  # pyrefly: ignore
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def generic_pipeline(self, task: str, model: Optional[str] = None, **kwargs) -> Any:
        """
        Run any Hugging Face pipeline.

        Args:
            task (str): The task name (e.g., "sentiment-analysis", "ner").
            model (str, optional): Model to use. Defaults to Hugging Face default for task.
            **kwargs: Additional arguments passed to the pipeline call.

        Returns:
            Any: The pipeline result.
        """
        pipeline_key = f"{task}_{model}"
        if pipeline_key not in self._pipelines:
            print(f"Loading pipeline for task '{task}'...")
            self._pipelines[pipeline_key] = pipeline(  # pyrefly: ignore
                task,
                model=model,
                device=self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

        return self._pipelines[pipeline_key](**kwargs)

    def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
    ) -> Image.Image:
        """
        Generate an image using a diffusion model.

        Args:
            prompt (str): The text prompt.
            model (str, optional): Override default image model.
            num_inference_steps (int): Number of denoising steps.
            guidance_scale (float): Classifier-free guidance scale.

        Returns:
            PIL.Image.Image: The generated image.
        """
        target_model = model if model else self.image_model_name
        pipeline_key = f"image_{target_model}"

        if pipeline_key not in self._pipelines:
            print(f"Loading image model {target_model}...")
            # Use AutoPipelineForText2Image for flexibility
            pipe = AutoPipelineForText2Image.from_pretrained(
                target_model,
                dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
            if self.device == "cuda":
                pipe.to("cuda")  # pyrefly: ignore
            self._pipelines[pipeline_key] = pipe

        image = self._pipelines[pipeline_key](
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        return image

    def transcribe_audio(
        self,
        audio_source: Union[str, bytes],
        model: Optional[str] = None,
        return_timestamps: bool = True,
    ) -> str:
        """
        Transcribe audio using a speech-to-text model (e.g., Whisper).

        Args:
            audio_source (Union[str, bytes]): Path to audio file or audio bytes.
            model (str, optional): Override default STT model.
            return_timestamps (bool): Whether to return timestamps (advanced usage).

        Returns:
            str: Transcribed text.
        """
        target_model = model if model else self.stt_model_name
        pipeline_key = f"stt_{target_model}"

        if pipeline_key not in self._pipelines:
            print(f"Loading ASR model {target_model}...")
            self._pipelines[pipeline_key] = pipeline(
                "automatic-speech-recognition",
                model=target_model,
                device=self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )

        # Handle bytes: pipelines typically accept filenames or numpy arrays.
        # If bytes, pipeline might handle it or we might need to convert to something else.
        # HF pipeline usually handles filenames nicely.

        result = self._pipelines[pipeline_key](
            audio_source, return_timestamps=return_timestamps
        )
        return result["text"]

    def generate_tts(
        self,
        text: str,
        model: Optional[str] = None,
        forward_params: Optional[Dict] = None,
    ) -> Any:
        """
        Generate speech from text.

        Args:
            text (str): Input text.
            model (str, optional): Override default TTS model.
            forward_params (Dict, optional): Params like speaker embeddings.

        Returns:
            Any: Audio data (often dictionary with 'audio' and 'sampling_rate').
        """
        target_model = model if model else self.tts_model_name
        pipeline_key = f"tts_{target_model}"

        if pipeline_key not in self._pipelines:
            print(f"Loading TTS model {target_model}...")
            # Note: Some models like speecht5 require embeddings, pipeline handles basic usage usually
            self._pipelines[pipeline_key] = pipeline(  # pyrefly: ignore
                "text-to-speech", model=target_model, device=self.device
            )

        kwargs = forward_params if forward_params else {}
        speech = self._pipelines[pipeline_key](text, **kwargs)
        return speech

    def image_analysis(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Analyze an image (captioning or VQA).

        Args:
            image (Union[str, Image.Image]): Image path or PIL Image object.
            prompt (str, optional): Optional prompt for VQA models.
            model (str, optional): Override default model.

        Returns:
            str: Description or answer.
        """
        target_model = model if model else self.image_to_text_model_name
        pipeline_key = f"img2txt_{target_model}"

        if pipeline_key not in self._pipelines:
            print(f"Loading Image analysis model {target_model}...")
            self._pipelines[pipeline_key] = pipeline(
                "image-to-text", model=target_model, device=self.device
            )

        result = self._pipelines[pipeline_key](image, prompt=prompt)
        # Results are usually list of dicts: [{'generated_text': '...'}]
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", str(result))
        return str(result)
