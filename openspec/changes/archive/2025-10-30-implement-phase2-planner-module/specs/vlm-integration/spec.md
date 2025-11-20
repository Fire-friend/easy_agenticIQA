# Capability: vlm-integration

Unified VLM client abstraction supporting multiple vision-language model providers for multimodal (text + image) inputs.

## ADDED Requirements

### Requirement: VLM Client Abstract Interface
The system SHALL provide a common interface for all VLM providers to enable consistent usage across the pipeline.

#### Scenario: Abstract VLMClient base class
- **Given** the need to support multiple VLM providers
- **When** designing the VLM client architecture
- **Then** define an abstract `VLMClient` base class with:
  - `generate(prompt: str, images: List[Image], **kwargs) -> str` method
  - `backend_name: str` property
  - `supports_vision: bool` property
- **And** all provider implementations inherit from this base class

#### Scenario: Configuration-based client instantiation
- **Given** a backend identifier (e.g., "openai.gpt-4o")
- **When** `create_vlm_client(backend: str, config: dict)` is called
- **Then** it returns the appropriate provider client instance
- **And** the client is initialized with API keys and configuration from environment

### Requirement: OpenAI VLM Client Implementation
The system SHALL support OpenAI's GPT-4o and other vision-capable models via the official Python SDK.

#### Scenario: OpenAI client initialization
- **Given** backend is "openai.gpt-4o"
- **When** creating the OpenAI VLM client
- **Then** it initializes with `OPENAI_API_KEY` from environment
- **And** optionally uses `OPENAI_BASE_URL` if set (for custom endpoints)
- **And** sets model name to "gpt-4o"

#### Scenario: OpenAI image encoding
- **Given** a PIL Image object and text prompt
- **When** calling `generate()` on OpenAI client
- **Then** it encodes the image as base64
- **And** constructs messages with role="user" and multimodal content
- **And** includes both text and image_url in the message

#### Scenario: OpenAI API call
- **Given** a properly formatted multimodal request
- **When** the OpenAI client makes the API call
- **Then** it uses `client.chat.completions.create()` with:
  - `model="gpt-4o"`
  - `messages=[{role: user, content: [text, image_url]}]`
  - `temperature`, `max_tokens`, `top_p` from configuration
- **And** extracts the response text from `choices[0].message.content`
- **And** returns the text response

### Requirement: Anthropic VLM Client Implementation
The system SHALL support Anthropic's Claude 3.5 Sonnet and other vision models via the official Python SDK.

#### Scenario: Anthropic client initialization
- **Given** backend is "anthropic.claude-3.5-sonnet"
- **When** creating the Anthropic VLM client
- **Then** it initializes with `ANTHROPIC_API_KEY` from environment
- **And** optionally uses `ANTHROPIC_BASE_URL` if set
- **And** sets model name to "claude-3-5-sonnet-20241022"

#### Scenario: Anthropic image encoding
- **Given** a PIL Image and text prompt
- **When** calling `generate()` on Anthropic client
- **Then** it encodes the image as base64
- **And** determines the media_type (e.g., "image/png")
- **And** constructs content blocks: `[{type: image, source: {type: base64, media_type, data}}, {type: text, text: prompt}]`

#### Scenario: Anthropic API call
- **Given** a properly formatted multimodal request
- **When** the Anthropic client makes the API call
- **Then** it uses `client.messages.create()` with:
  - `model="claude-3-5-sonnet-20241022"`
  - `messages=[{role: user, content: [image_block, text_block]}]`
  - `max_tokens`, `temperature` from configuration
- **And** extracts the response from `content[0].text`
- **And** returns the text response

### Requirement: Google Gemini VLM Client Implementation
The system SHALL support Google's Gemini Pro Vision via the official Python SDK.

#### Scenario: Google client initialization
- **Given** backend is "google.gemini-pro-vision"
- **When** creating the Google VLM client
- **Then** it initializes with `GOOGLE_API_KEY` from environment
- **And** optionally uses `GOOGLE_API_BASE_URL` if set
- **And** configures the model as "gemini-pro-vision" or "gemini-1.5-pro"

#### Scenario: Google multimodal generation
- **Given** a PIL Image and text prompt
- **When** calling `generate()` on Google client
- **Then** it converts the PIL Image to bytes
- **And** calls `model.generate_content([prompt, image_bytes])`
- **And** extracts the text response
- **And** returns the response

### Requirement: VLM Client Error Handling
All VLM clients SHALL handle common error scenarios consistently.

#### Scenario: API authentication failure
- **Given** an invalid or missing API key
- **When** the VLM client attempts an API call
- **Then** it raises an `AuthenticationError` with clear message
- **And** logs the error (without exposing the API key)

#### Scenario: Rate limit handling
- **Given** the API returns a rate limit error (429 status)
- **When** the VLM client receives this error
- **Then** it raises a `RateLimitError`
- **And** includes retry-after information if available

#### Scenario: Network timeout handling
- **Given** the API call times out due to network issues
- **When** the timeout occurs
- **Then** the client raises a `TimeoutError`
- **And** includes request details for debugging

#### Scenario: Invalid response handling
- **Given** the API returns an unexpected response format
- **When** parsing the response
- **Then** the client raises a `ResponseParseError`
- **And** includes the raw response for debugging

### Requirement: Image Loading and Preprocessing
VLM clients SHALL handle image loading and format conversion consistently.

#### Scenario: Load image from file path
- **Given** an image file path (e.g., "/path/to/image.jpg")
- **When** the VLM client loads the image
- **Then** it uses PIL.Image.open() to load the image
- **And** converts to RGB if necessary (handling RGBA, grayscale)
- **And** returns a PIL Image object

#### Scenario: Image format validation
- **Given** an image file with unsupported format
- **When** attempting to load the image
- **Then** it raises a `ValueError` with supported formats listed
- **Supported formats**: jpg, jpeg, png, bmp, tiff

#### Scenario: Image size handling
- **Given** a very large image (>20MB)
- **When** loading the image for VLM input
- **Then** it optionally resizes to a reasonable size (e.g., max 2048px on longest side)
- **And** logs a warning about resizing
- **And** preserves aspect ratio

### Requirement: VLM Client Factory and Registry
The system SHALL provide a factory function to instantiate VLM clients based on backend identifier.

#### Scenario: Client factory function
- **Given** a backend string like "openai.gpt-4o"
- **When** calling `create_vlm_client(backend, config)`
- **Then** it parses the provider prefix ("openai")
- **And** instantiates the appropriate client class (OpenAIVLMClient)
- **And** passes configuration parameters to the client
- **And** returns the initialized client instance

#### Scenario: Unsupported backend handling
- **Given** an unrecognized backend string (e.g., "invalid.model")
- **When** calling `create_vlm_client()`
- **Then** it raises a `ValueError` with list of supported backends
- **Supported backends**: "openai.*", "anthropic.*", "google.*"

#### Scenario: Client caching
- **Given** multiple requests for the same backend configuration
- **When** creating VLM clients
- **Then** optionally cache and reuse client instances
- **And** avoid redundant SDK initialization
