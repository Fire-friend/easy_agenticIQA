"""
VLM (Vision-Language Model) client abstraction for AgenticIQA.
Provides unified interface for multiple VLM providers (OpenAI, Anthropic, Google).
"""

import base64
import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any

from PIL import Image


# ==================== Exceptions ====================

class VLMClientError(Exception):
    """Base exception for VLM client errors."""
    pass


class AuthenticationError(VLMClientError):
    """Raised when API authentication fails."""
    pass


class RateLimitError(VLMClientError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutError(VLMClientError):
    """Raised when API request times out."""
    pass


class ResponseParseError(VLMClientError):
    """Raised when API response cannot be parsed."""
    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(message)
        self.raw_response = raw_response


# ==================== Abstract Base Class ====================

class VLMClient(ABC):
    """Abstract base class for vision-language model clients."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize VLM client.

        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            base_url: Optional custom API endpoint
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend provider name."""
        pass

    @property
    def supports_vision(self) -> bool:
        """Return whether this client supports vision inputs."""
        return True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        **kwargs
    ) -> str:
        """
        Generate text response from VLM.

        Args:
            prompt: Text prompt
            images: List of PIL Image objects
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            Generated text response

        Raises:
            AuthenticationError: If API authentication fails
            RateLimitError: If rate limit is exceeded
            TimeoutError: If request times out
            ResponseParseError: If response cannot be parsed
        """
        pass


# ==================== OpenAI Client ====================

class OpenAIVLMClient(VLMClient):
    """OpenAI VLM client (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI VLM client.

        Args:
            model_name: OpenAI model name (e.g., "gpt-4o")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional custom API endpoint (defaults to OPENAI_BASE_URL env var)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise AuthenticationError("OpenAI API key not provided and OPENAI_API_KEY environment variable not set")

        super().__init__(model_name, api_key, base_url)

        # Lazy import to avoid dependency if not using OpenAI
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for OpenAI client. Install with: pip install openai")

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    @property
    def backend_name(self) -> str:
        return "openai"

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = io.BytesIO()
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        top_p: float = 0.1,
        **kwargs
    ) -> str:
        """Generate text response using OpenAI API."""
        try:
            # Construct multimodal content
            content = []

            # Add images first
            for image in images:
                image_base64 = self._encode_image_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt
            })

            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                **kwargs
            )

            # Extract response text
            return response.choices[0].message.content

        except Exception as e:
            # Handle specific OpenAI errors
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str or "401" in error_str:
                raise AuthenticationError(f"OpenAI authentication failed: {e}")
            elif "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(f"OpenAI rate limit exceeded: {e}")
            elif "timeout" in error_str:
                raise TimeoutError(f"OpenAI request timed out: {e}")
            else:
                raise ResponseParseError(f"OpenAI request failed: {e}")


# ==================== Anthropic Client ====================

class AnthropicVLMClient(VLMClient):
    """Anthropic VLM client (Claude 3.5 Sonnet, etc.)."""

    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Anthropic VLM client.

        Args:
            model_name: Anthropic model name (e.g., "claude-3-5-sonnet-20241022")
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: Optional custom API endpoint (defaults to ANTHROPIC_BASE_URL env var)
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

        if not api_key:
            raise AuthenticationError("Anthropic API key not provided and ANTHROPIC_API_KEY environment variable not set")

        super().__init__(model_name, api_key, base_url)

        # Lazy import
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package is required for Anthropic client. Install with: pip install anthropic")

        # Initialize Anthropic client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = Anthropic(**client_kwargs)

    @property
    def backend_name(self) -> str:
        return "anthropic"

    def _encode_image_base64(self, image: Image.Image) -> tuple[str, str]:
        """
        Encode PIL Image to base64 string with media type.

        Returns:
            Tuple of (base64_data, media_type)
        """
        buffered = io.BytesIO()

        # Determine format and media type
        if image.mode == 'RGBA':
            image.save(buffered, format="PNG")
            media_type = "image/png"
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            media_type = "image/jpeg"

        base64_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_data, media_type

    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text response using Anthropic API."""
        try:
            # Construct content blocks
            content = []

            # Add images
            for image in images:
                base64_data, media_type = self._encode_image_base64(image)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                })

            # Add text prompt
            content.append({
                "type": "text",
                "text": prompt
            })

            # Make API call
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                **kwargs
            )

            # Extract response text
            return response.content[0].text

        except Exception as e:
            # Handle specific Anthropic errors
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str or "401" in error_str:
                raise AuthenticationError(f"Anthropic authentication failed: {e}")
            elif "rate limit" in error_str or "429" in error_str:
                raise RateLimitError(f"Anthropic rate limit exceeded: {e}")
            elif "timeout" in error_str:
                raise TimeoutError(f"Anthropic request timed out: {e}")
            else:
                raise ResponseParseError(f"Anthropic request failed: {e}")


# ==================== Google Client (Stub) ====================

class GoogleVLMClient(VLMClient):
    """Google VLM client (Gemini Pro Vision, etc.). Note: Stub implementation for Phase 2."""

    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Google VLM client.

        Args:
            model_name: Google model name (e.g., "gemini-1.5-pro")
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            base_url: Optional custom API endpoint (defaults to GOOGLE_API_BASE_URL env var)
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        base_url = base_url or os.getenv("GOOGLE_API_BASE_URL")

        if not api_key:
            raise AuthenticationError("Google API key not provided and GOOGLE_API_KEY environment variable not set")

        super().__init__(model_name, api_key, base_url)

        # Lazy import
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai package is required for Google client. Install with: pip install google-genai")

        # Create client
        self.client = genai.Client(api_key=self.api_key)
        if base_url:
            # Custom endpoint support (if needed)
            self.client._client_options = {"api_endpoint": base_url}
        self.types = types

    @property
    def backend_name(self) -> str:
        return "google"

    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """Generate text response using Google Gemini API."""
        try:
            # Prepare content parts (text + images)
            import io
            content_parts = [prompt]

            # Convert PIL images to bytes for the new API
            for img in images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                content_parts.append(self.types.Part.from_bytes(
                    data=img_byte_arr.getvalue(),
                    mime_type='image/png'
                ))

            # Generate response using new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=self.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            return response.text

        except Exception as e:
            error_str = str(e).lower()
            if "authentication" in error_str or "api key" in error_str or "401" in error_str:
                raise AuthenticationError(f"Google authentication failed: {e}")
            elif "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                raise RateLimitError(f"Google rate limit exceeded: {e}")
            elif "timeout" in error_str:
                raise TimeoutError(f"Google request timed out: {e}")
            else:
                raise ResponseParseError(f"Google request failed: {e}")


# ==================== Image Loading Utilities ====================

def load_image(image_path: str, max_size: Optional[int] = 2048) -> Image.Image:
    """
    Load image from file path.

    Args:
        image_path: Path to image file
        max_size: Maximum size (longest edge) for resizing. None to disable resizing.

    Returns:
        PIL Image object

    Raises:
        ValueError: If file format is unsupported or file doesn't exist
    """
    path = Path(image_path)

    if not path.exists():
        raise ValueError(f"Image file not found: {image_path}")

    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Unsupported image format: {path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}"
        )

    # Load image
    try:
        image = Image.open(path)

        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')

        # Resize if necessary
        if max_size and max(image.size) > max_size:
            # Calculate new size preserving aspect ratio
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Warning: Resized image from {image.size} to {new_size} (max_size={max_size})")

        return image

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {e}")


# ==================== Factory Function ====================

def create_vlm_client(backend: str, config: Optional[Dict[str, Any]] = None) -> VLMClient:
    """
    Factory function to create VLM client based on backend identifier.

    Args:
        backend: Backend identifier (e.g., "openai.gpt-4o", "anthropic.claude-3.5-sonnet")
        config: Optional configuration dict with keys: api_key, base_url, temperature, max_tokens, etc.

    Returns:
        Initialized VLM client instance

    Raises:
        ValueError: If backend is not supported

    Examples:
        >>> client = create_vlm_client("openai.gpt-4o")
        >>> client = create_vlm_client("anthropic.claude-3-5-sonnet-20241022", {"api_key": "..."})
    """
    config = config or {}

    # Parse backend identifier
    if backend.startswith("openai."):
        model_name = backend.replace("openai.", "")
        return OpenAIVLMClient(
            model_name=model_name,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )

    elif backend.startswith("anthropic."):
        model_name = backend.replace("anthropic.", "")
        return AnthropicVLMClient(
            model_name=model_name,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )

    elif backend.startswith("google."):
        model_name = backend.replace("google.", "")
        return GoogleVLMClient(
            model_name=model_name,
            api_key=config.get("api_key"),
            base_url=config.get("base_url")
        )

    else:
        supported_backends = ["openai.*", "anthropic.*", "google.*"]
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: {', '.join(supported_backends)}"
        )
