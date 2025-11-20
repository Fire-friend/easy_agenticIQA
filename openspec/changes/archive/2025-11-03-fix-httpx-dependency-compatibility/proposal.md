# Proposal: Fix HTTP Library Dependency Compatibility

## Metadata
- **Change ID**: `fix-httpx-dependency-compatibility`
- **Type**: Bug Fix
- **Priority**: High
- **Status**: Proposed
- **Created**: 2025-11-03

## Why

The AgenticIQA pipeline is completely blocked due to a critical dependency version conflict. The `openai==1.35.7` package passes a `proxies` parameter to httpx that was removed in httpx 0.28.x, causing `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` on every VLM client initialization. This prevents demo.sh execution and blocks all pipeline functionality (Planner, Executor, Summarizer).

## What Changes

Upgrade the dependency versions to compatible versions:

- **openai**: `1.35.7` → `2.6.1` (latest stable)
- **httpx**: `0.27.2` → `0.28.1` (required by google-genai >=0.28.1)
- **langchain-openai**: `0.1.20` → `1.0.1` (compatible with openai 2.x)
- **langchain-anthropic**: `0.1.23` → `1.0.1` (compatible with langchain-core 1.0.1)
- **anthropic**: `0.30.0` → `0.72.0` (required by langchain-anthropic 1.0.1)

## Impact

### Affected Components
- **VLM Client Layer** (`src/agentic/vlm_client.py`): Uses openai and anthropic packages
- **LangGraph Integration**: Uses langchain-openai and langchain-anthropic packages
- **Environment Setup**: Documentation updated with new dependency versions
- **Requirements File**: Updated with version constraints

### Benefits
1. **Immediate**: Resolves the blocking httpx compatibility issue
2. **Stability**: Uses latest stable versions with better bug fixes and features
3. **Future-proof**: Aligns with current ecosystem standards
4. **Consistency**: Resolves all dependency conflicts (no more pip warnings)

### Risks and Mitigation
- **API Changes**: openai 2.x may have breaking changes compared to 1.x (Mitigated: VLM client abstraction layer handles API differences)
- **Testing Required**: Need to verify all VLM clients work correctly (Mitigated: Basic verification with demo.sh completed)

### Migration Notes
This is a dependency-only change with no code modifications required. Users upgrading should run:
```bash
pip install --upgrade "openai>=2.0.0,<3.0.0" "anthropic>=0.72.0,<1.0.0" "httpx>=0.28.1,<1.0.0" \
  "langchain-openai>=1.0.0,<2.0.0" "langchain-anthropic>=1.0.0,<2.0.0"
```

### Alternatives Considered
1. **Downgrade httpx to 0.27.x**: Not viable because google-genai requires httpx >=0.28.1
2. **Pin openai to older version**: Creates conflicts with langchain packages
3. **Lock all dependencies**: Makes future updates difficult and doesn't solve the root issue
