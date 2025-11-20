# dependency-management Spec Delta

## MODIFIED Requirements

### Requirement: Python Package Requirements
The system SHALL specify all Python dependencies with version constraints in requirements.txt.

#### Scenario: API client dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - openai>=2.0.0,<3.0.0 (for GPT-4o API, upgraded from 1.35.7 for httpx 0.28+ compatibility)
  - anthropic>=0.72.0,<1.0.0 (for Claude API, upgraded from 0.30.0)
  - google-genai (Google's library for Gemini API, requires httpx>=0.28.1)
  - qwen-vl-utils==0.0.8 (for local Qwen2.5-VL)
  - httpx>=0.28.1,<1.0.0 (required by google-genai and compatible with openai 2.x)

#### Scenario: LangGraph framework dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - langgraph (latest stable)
  - langchain-core>=1.0.0,<2.0.0 (upgraded for compatibility)
  - langchain-openai>=1.0.0,<2.0.0 (for OpenAI integration, upgraded for openai 2.x support)
  - langchain-anthropic>=1.0.0,<2.0.0 (for Anthropic integration, upgraded for anthropic 0.72+ support)

### Requirement: Dependency Version Compatibility
The system SHALL ensure all dependencies are mutually compatible.

#### Scenario: API client version compatibility
- **WHEN** API clients are installed
- **THEN** versions SHALL be compatible with:
  - openai 2.x API (breaking changes from 1.x handled by VLM abstraction layer)
  - anthropic 0.72.x API (updated from 0.30.0)
  - google-genai latest (requires httpx>=0.28.1)
  - httpx 0.28.x (proxies parameter removed, handled by openai 2.x)
- **AND** pin versions to avoid breaking changes

#### Scenario: HTTP client dependency resolution
- **WHEN** httpx is installed
- **THEN** version SHALL satisfy all dependent packages:
  - google-genai requires httpx>=0.28.1
  - openai 2.x is compatible with httpx 0.28.x (removed proxies parameter usage)
  - anthropic 0.72.x is compatible with httpx 0.28.x
  - langchain packages use httpx through API client libraries
- **AND** no conflicting proxy configuration attempts

#### Scenario: LangChain integration compatibility
- **WHEN** langchain packages are installed
- **THEN** versions SHALL be compatible with:
  - langchain-core 1.0.x (upgraded from 0.2.x)
  - langchain-openai 1.0.x with openai 2.x support
  - langchain-anthropic 1.0.x with anthropic 0.72.x support
- **AND** no version conflicts between langchain-* packages

## Rationale

This change addresses a critical dependency conflict where:
1. `openai 1.35.7` passes a `proxies` parameter to httpx Client that was removed in httpx 0.28.x
2. `google-genai` requires `httpx>=0.28.1` for proper functionality
3. Upgrading to `openai 2.x` resolves the httpx compatibility issue
4. LangChain packages need corresponding upgrades to support the new API client versions

The VLM client abstraction layer (`src/agentic/vlm_client.py`) already handles API differences, so no code changes are required—only dependency version updates.
