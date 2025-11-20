"""
Unit tests for VLM client abstraction in src/agentic/vlm_client.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from PIL import Image

from src.agentic.vlm_client import (
    VLMClient,
    OpenAIVLMClient,
    AnthropicVLMClient,
    GoogleVLMClient,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ResponseParseError,
    load_image,
    create_vlm_client
)


class TestImageLoading:
    """Tests for image loading utility."""

    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image."""
        image_path = tmp_path / "test_image.jpg"
        # Create a simple 100x100 RGB image
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
        return str(image_path)

    @pytest.fixture
    def large_image(self, tmp_path):
        """Create a large test image."""
        image_path = tmp_path / "large_image.jpg"
        img = Image.new('RGB', (3000, 2000), color='blue')
        img.save(image_path)
        return str(image_path)

    def test_load_valid_image(self, temp_image):
        """Test loading a valid image file."""
        image = load_image(temp_image)
        assert isinstance(image, Image.Image)
        assert image.size == (100, 100)

    def test_load_nonexistent_image(self):
        """Test that loading nonexistent image raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_image("/nonexistent/image.jpg")
        assert "Image file not found" in str(exc_info.value)

    def test_load_invalid_format(self, tmp_path):
        """Test that invalid image format raises ValueError."""
        text_file = tmp_path / "not_image.txt"
        text_file.touch()

        with pytest.raises(ValueError) as exc_info:
            load_image(str(text_file))
        assert "Unsupported image format" in str(exc_info.value)

    def test_image_resizing(self, large_image, capsys):
        """Test that large images are resized."""
        image = load_image(large_image, max_size=2048)
        assert max(image.size) <= 2048
        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: Resized image" in captured.out

    def test_no_resizing(self, temp_image):
        """Test that small images are not resized."""
        image = load_image(temp_image, max_size=2048)
        assert image.size == (100, 100)


class TestOpenAIVLMClient:
    """Tests for OpenAI VLM client."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch('openai.OpenAI') as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance

            # Mock successful response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            client_instance.chat.completions.create.return_value = mock_response

            yield mock, client_instance

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new('RGB', (50, 50), color='green')

    def test_init_with_api_key(self, mock_openai):
        """Test initializing with API key."""
        client = OpenAIVLMClient(api_key="test-key")
        assert client.backend_name == "openai"
        assert client.api_key == "test-key"

    def test_init_from_env(self, mock_openai):
        """Test initializing from environment variable."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):
            client = OpenAIVLMClient()
            assert client.api_key == "env-key"

    def test_init_without_api_key(self):
        """Test that missing API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(AuthenticationError):
                OpenAIVLMClient()

    def test_generate_success(self, mock_openai, test_image):
        """Test successful text generation."""
        mock_class, mock_instance = mock_openai

        client = OpenAIVLMClient(api_key="test-key")
        response = client.generate("Test prompt", [test_image])

        assert response == "Test response"
        # Verify API was called
        mock_instance.chat.completions.create.assert_called_once()

    def test_generate_with_params(self, mock_openai, test_image):
        """Test generation with custom parameters."""
        mock_class, mock_instance = mock_openai

        client = OpenAIVLMClient(api_key="test-key")
        response = client.generate(
            "Test prompt",
            [test_image],
            temperature=0.5,
            max_tokens=1000,
            top_p=0.9
        )

        # Verify parameters were passed
        call_kwargs = mock_instance.chat.completions.create.call_args[1]
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 1000
        assert call_kwargs['top_p'] == 0.9

    def test_authentication_error(self, mock_openai, test_image):
        """Test handling authentication error."""
        mock_class, mock_instance = mock_openai
        mock_instance.chat.completions.create.side_effect = Exception("401 authentication failed")

        client = OpenAIVLMClient(api_key="invalid-key")
        with pytest.raises(AuthenticationError):
            client.generate("Test prompt", [test_image])

    def test_rate_limit_error(self, mock_openai, test_image):
        """Test handling rate limit error."""
        mock_class, mock_instance = mock_openai
        mock_instance.chat.completions.create.side_effect = Exception("429 rate limit exceeded")

        client = OpenAIVLMClient(api_key="test-key")
        with pytest.raises(RateLimitError):
            client.generate("Test prompt", [test_image])


class TestAnthropicVLMClient:
    """Tests for Anthropic VLM client."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock Anthropic client."""
        with patch('anthropic.Anthropic') as mock:
            client_instance = MagicMock()
            mock.return_value = client_instance

            # Mock successful response
            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = "Test response"
            client_instance.messages.create.return_value = mock_response

            yield mock, client_instance

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new('RGB', (50, 50), color='blue')

    def test_init_with_api_key(self, mock_anthropic):
        """Test initializing with API key."""
        client = AnthropicVLMClient(api_key="test-key")
        assert client.backend_name == "anthropic"
        assert client.api_key == "test-key"

    def test_init_from_env(self, mock_anthropic):
        """Test initializing from environment variable."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key'}):
            client = AnthropicVLMClient()
            assert client.api_key == "env-key"

    def test_generate_success(self, mock_anthropic, test_image):
        """Test successful text generation."""
        mock_class, mock_instance = mock_anthropic

        client = AnthropicVLMClient(api_key="test-key")
        response = client.generate("Test prompt", [test_image])

        assert response == "Test response"
        mock_instance.messages.create.assert_called_once()

    def test_generate_with_params(self, mock_anthropic, test_image):
        """Test generation with custom parameters."""
        mock_class, mock_instance = mock_anthropic

        client = AnthropicVLMClient(api_key="test-key")
        response = client.generate(
            "Test prompt",
            [test_image],
            temperature=0.5,
            max_tokens=1000
        )

        call_kwargs = mock_instance.messages.create.call_args[1]
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['max_tokens'] == 1000


@pytest.mark.skip(reason="Google client is stub implementation for Phase 2; requires google-genai package")
class TestGoogleVLMClient:
    """Tests for Google VLM client (stub implementation)."""

    @pytest.fixture
    def mock_google(self):
        """Mock Google Gemini client."""
        with patch('google.genai') as mock:
            # Mock client
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Test response"
            mock_client.models.generate_content.return_value = mock_response

            mock.Client.return_value = mock_client

            yield mock, mock_client

    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new('RGB', (50, 50), color='yellow')

    def test_init_with_api_key(self, mock_google):
        """Test initializing with API key."""
        client = GoogleVLMClient(api_key="test-key")
        assert client.backend_name == "google"
        assert client.api_key == "test-key"

    def test_generate_success(self, mock_google, test_image):
        """Test successful text generation."""
        mock_genai, mock_model = mock_google

        client = GoogleVLMClient(api_key="test-key")
        response = client.generate("Test prompt", [test_image])

        assert response == "Test response"
        mock_model.generate_content.assert_called_once()


class TestVLMClientFactory:
    """Tests for VLM client factory function."""

    def test_create_openai_client(self):
        """Test creating OpenAI client via factory."""
        with patch('openai.OpenAI'):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                client = create_vlm_client("openai.gpt-4o")
                assert isinstance(client, OpenAIVLMClient)
                assert client.model_name == "gpt-4o"

    def test_create_anthropic_client(self):
        """Test creating Anthropic client via factory."""
        with patch('anthropic.Anthropic'):
            with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
                client = create_vlm_client("anthropic.claude-3-5-sonnet-20241022")
                assert isinstance(client, AnthropicVLMClient)
                assert client.model_name == "claude-3-5-sonnet-20241022"

    @pytest.mark.skip(reason="Google client is stub implementation for Phase 2")
    def test_create_google_client(self):
        """Test creating Google client via factory."""
        with patch('google.genai'):
            with patch.dict('os.environ', {'GOOGLE_API_KEY': 'test-key'}):
                client = create_vlm_client("google.gemini-1.5-pro")
                assert isinstance(client, GoogleVLMClient)
                assert client.model_name == "gemini-1.5-pro"

    def test_create_with_config(self):
        """Test creating client with custom config."""
        with patch('openai.OpenAI'):
            config = {"api_key": "custom-key", "base_url": "https://custom.endpoint"}
            client = create_vlm_client("openai.gpt-4o", config)
            assert client.api_key == "custom-key"
            assert client.base_url == "https://custom.endpoint"

    def test_unsupported_backend(self):
        """Test that unsupported backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_vlm_client("invalid.backend")
        assert "Unsupported backend" in str(exc_info.value)
