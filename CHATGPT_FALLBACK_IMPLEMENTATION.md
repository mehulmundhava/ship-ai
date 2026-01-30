# ChatGPT Fallback Implementation Guide

## Overview

This document outlines the changes needed to add ChatGPT as a fallback when Groq API fails. The current code uses Groq as the primary LLM provider. When Groq fails at any step, the system should automatically fall back to ChatGPT without changing the logic or flow.

---

## Current Architecture

### LLM Service Layer
- **File**: `app/services/llm_service.py`
- **Class**: `LLMService`
- **Function**: Creates LLM instances based on `LLM_PROVIDER` environment variable
- **Current Behavior**: Returns either Groq or OpenAI LLM based on configuration

### Agent Graph Layer
- **File**: `app/core/agent/agent_graph.py`
- **Class**: `SQLAgentGraph`
- **LLM Usage**: The LLM instance is passed to the constructor and stored as `self.llm`

### Controller Layer
- **File**: `app/controllers/chat_controller.py`
- **Function**: `process_chat()` creates LLM instance and passes it to SQLAgentGraph

---

## Where Groq is Currently Used

The LLM is invoked in **5 locations** in `agent_graph.py`:

### 1. Security Guard Validation (Line ~393)
```python
security_response = self.llm.invoke(security_messages)
```
- **Purpose**: Validates user queries for security
- **Context**: First step in agent flow for non-admin users

### 2. Main Agent Node with Tools (Line ~654)
```python
response = self.llm_with_tools.invoke(messages)
```
- **Purpose**: Primary agent decision-making with tool binding
- **Context**: Main agent node that decides which tools to call

### 3. Fallback Without Tools (Line ~680)
```python
response = self.llm.invoke(messages)
```
- **Purpose**: Fallback when tool calls fail
- **Context**: Attempts to get text response when tool binding fails

### 4. Format Answer Node (Line ~1222)
```python
response = self.llm.invoke(final_messages)
```
- **Purpose**: Generates final answer from query results
- **Context**: Formats SQL query results into natural language

### 5. Format Answer Node (Alternative Path) (Line ~1370)
```python
response = self.llm.invoke(final_messages)
```
- **Purpose**: Alternative path for formatting answers
- **Context**: Handles cases where SQL is extracted from final message

---

## Required Changes

### Approach: Add Fallback Method in SQLAgentGraph

Add a helper method in `SQLAgentGraph` that:
1. Attempts the LLM call with the current provider (Groq)
2. Catches Groq-specific errors
3. Automatically switches to ChatGPT and retries
4. Preserves the same logic/flow

---

## Implementation Strategy

### 1. Modify `LLMService` to Support Fallback Instance Creation

**File**: `app/services/llm_service.py`

Add a method to get a fallback LLM instance:

```python
def get_fallback_llm_model(
    self, 
    temperature: float = 0, 
    model: str = None
) -> BaseChatModel:
    """
    Get fallback LLM instance (always OpenAI/ChatGPT).
    Used when primary provider (Groq) fails.
    
    Args:
        temperature (float): Controls randomness in responses. Default: 0
        model (str): Model name. If None, uses OpenAI default.
    
    Returns:
        BaseChatModel: Configured OpenAI LLM instance
    """
    return self.openai_llm_model(
        temperature=temperature,
        model=model or "gpt-4o"
    )
```

**Location**: Add this method after the `groq_llm_model()` method (around line 132)

---

### 2. Modify `SQLAgentGraph` to Support Fallback

**File**: `app/core/agent/agent_graph.py`

#### 2.1. Store Provider Information in Constructor

**Location**: In `__init__` method (around line 99, after `self.llm = llm`)

```python
# Store provider info for fallback logic
from app.services.llm_service import LLMService
llm_service = LLMService()
self.provider = llm_service.get_provider()
```

#### 2.2. Add Method to Get/Create Fallback LLM

**Location**: Add after `_bind_tools_conditionally()` method (around line 179)

```python
def _get_fallback_llm(self, use_tools=False):
    """
    Get or create fallback LLM instance (ChatGPT).
    
    Args:
        use_tools: Whether to bind tools to fallback LLM
    
    Returns:
        BaseChatModel: Configured OpenAI LLM instance
    """
    if not hasattr(self, '_fallback_llm') or self._fallback_llm is None:
        # Import here to avoid circular dependency
        from app.services.llm_service import LLMService
        llm_service = LLMService()
        self._fallback_llm = llm_service.get_fallback_llm_model()
        
        logger.info("✅ Created ChatGPT fallback LLM instance")
        print(f"✅ Created ChatGPT fallback LLM instance")
    
    # If tools are needed, bind them
    if use_tools:
        if not hasattr(self, '_fallback_llm_with_tools') or self._fallback_llm_with_tools is None:
            # Determine which tools to bind (same as regular LLM)
            question = getattr(self, '_last_question', '')
            is_journey = self._is_journey_question(question)
            tools_to_bind = self.journey_tools if is_journey else self.regular_tools
            self._fallback_llm_with_tools = self._fallback_llm.bind_tools(tools_to_bind)
        return self._fallback_llm_with_tools
    else:
        return self._fallback_llm
```

#### 2.3. Create Helper Method for LLM Invocation with Fallback

**Location**: Add after `_get_fallback_llm()` method

```python
def _invoke_llm_with_fallback(self, messages, use_tools=False):
    """
    Invoke LLM with automatic fallback to ChatGPT if Groq fails.
    
    Args:
        messages: List of messages to send to LLM
        use_tools: Whether to use tool binding (llm_with_tools)
    
    Returns:
        LLM response
    """
    try:
        if use_tools and self.llm_with_tools:
            return self.llm_with_tools.invoke(messages)
        else:
            return self.llm.invoke(messages)
    except Exception as e:
        # Check if it's a Groq-specific error
        error_str = str(e)
        error_type = type(e).__name__
        
        # Groq-specific error patterns
        is_groq_error = (
            "groq" in error_str.lower() or
            "BadRequestError" in error_type or
            "tool_use_failed" in error_str or
            "Failed to call a function" in error_str or
            "groq.BadRequestError" in error_type
        )
        
        if is_groq_error and self.provider == "GROQ":
            logger.warning(f"Groq API error detected: {error_type}. Falling back to ChatGPT...")
            print(f"⚠️  Groq API error detected: {error_type}")
            print(f"🔄 Falling back to ChatGPT...")
            
            # Get fallback LLM (ChatGPT)
            fallback_llm = self._get_fallback_llm(use_tools)
            
            # Retry with ChatGPT
            try:
                response = fallback_llm.invoke(messages)
                
                logger.info("✅ Successfully got response from ChatGPT fallback")
                print(f"✅ Successfully got response from ChatGPT fallback")
                return response
            except Exception as e2:
                logger.error(f"ChatGPT fallback also failed: {e2}")
                logger.exception("ChatGPT fallback error")
                print(f"❌ ChatGPT fallback also failed: {e2}")
                raise e  # Re-raise original Groq error
        else:
            # Not a Groq error, re-raise
            raise e
```

#### 2.4. Replace All LLM Invocations with Fallback Helper

**Location 1 - Security Guard (Line ~393)**:

```python
# OLD:
security_response = self.llm.invoke(security_messages)

# NEW:
security_response = self._invoke_llm_with_fallback(security_messages, use_tools=False)
```

**Location 2 - Main Agent Node (Line ~654)**:

```python
# OLD:
response = self.llm_with_tools.invoke(messages)

# NEW:
response = self._invoke_llm_with_fallback(messages, use_tools=True)
```

**Location 3 - Fallback Without Tools (Line ~680)**:

```python
# OLD:
response = self.llm.invoke(messages)

# NEW:
response = self._invoke_llm_with_fallback(messages, use_tools=False)
```

**Location 4 - Format Answer (Line ~1222)**:

```python
# OLD:
response = self.llm.invoke(final_messages)

# NEW:
response = self._invoke_llm_with_fallback(final_messages, use_tools=False)
```

**Location 5 - Format Answer Alternative (Line ~1370)**:

```python
# OLD:
response = self.llm.invoke(final_messages)

# NEW:
response = self._invoke_llm_with_fallback(final_messages, use_tools=False)
```

#### 2.5. Update Error Handling

**Location**: In `_agent_node` method (Lines ~655-690)

The existing Groq error handling can be simplified since the fallback logic will handle it:

```python
# Simplified error handling - fallback is handled in _invoke_llm_with_fallback
try:
    response = self._invoke_llm_with_fallback(messages, use_tools=True)
except Exception as e:
    # This will only catch errors if both Groq AND ChatGPT fail
    error_str = str(e)
    error_type = type(e).__name__
    
    logger.error(f"Both Groq and ChatGPT failed")
    logger.error(f"Error Type: {error_type}")
    logger.error(f"Error Message: {error_str}")
    logger.exception("Full exception details")
    
    print(f"\n{'='*80}")
    print(f"❌ BOTH GROQ AND CHATGPT FAILED")
    print(f"{'='*80}")
    print(f"Error Type: {error_type}")
    print(f"Error Message: {error_str}")
    raise e
```

**Note**: The existing error handling code (lines 655-690) that tries to continue without tools can be removed or simplified, as the fallback method will handle Groq errors automatically.

---

## Environment Variables

No changes required. The existing environment variables are sufficient:

- `LLM_PROVIDER=GROQ` (primary provider)
- `API_KEY` or `OPENAI_API_KEY` (required for fallback)
- `GROQ_API_KEY` (required for primary provider)

**Important**: Both API keys must be set in the `.env` file for the fallback to work.

---

## Testing Considerations

### 1. Test Groq Success
- Verify normal flow when Groq works
- Ensure no fallback is triggered
- Check that logs show Groq usage

### 2. Test Groq Failure
- Mock/simulate Groq errors to trigger fallback
- Test different error types:
  - `BadRequestError`
  - `tool_use_failed`
  - Network errors
  - Rate limit errors

### 3. Test ChatGPT Fallback
- Verify ChatGPT is used when Groq fails
- Check that logs show fallback activation
- Ensure response quality is maintained

### 4. Test Both Failures
- Verify error handling when both providers fail
- Ensure proper error messages are logged
- Check that user receives appropriate error response

### 5. Test All 5 Locations
- Ensure fallback works at all LLM invocation points:
  - Security guard validation
  - Main agent node with tools
  - Fallback without tools
  - Format answer node
  - Format answer alternative path

### 6. Test Tool Binding
- Verify that fallback LLM has tools bound correctly
- Test both journey tools and regular tools
- Ensure tool execution works with ChatGPT fallback

---

## Benefits of This Approach

1. **No Logic Changes**: The flow remains identical, only the LLM provider changes
2. **Automatic Fallback**: Transparent to the user - no manual intervention needed
3. **Error Resilience**: System continues to work even when Groq is down
4. **Minimal Code Changes**: Only adds helper methods and replaces invoke calls
5. **Backward Compatible**: Existing functionality remains unchanged
6. **Lazy Initialization**: Fallback LLM is only created when needed
7. **Consistent Tool Binding**: Fallback LLM uses same tool binding logic as primary LLM

---

## Summary of Files to Modify

### Files Requiring Changes:

1. **`app/services/llm_service.py`**
   - Add `get_fallback_llm_model()` method (after line 132)

2. **`app/core/agent/agent_graph.py`**
   - Add `_get_fallback_llm()` method (after line 179)
   - Add `_invoke_llm_with_fallback()` method (after `_get_fallback_llm()`)
   - Store provider info in `__init__` (around line 99)
   - Replace 5 LLM invocation calls with fallback helper:
     - Line ~393: Security guard
     - Line ~654: Main agent node
     - Line ~680: Fallback without tools
     - Line ~1222: Format answer
     - Line ~1370: Format answer alternative
   - Simplify existing error handling (lines ~655-690)

### Files NOT Requiring Changes:

- `app/controllers/chat_controller.py`
- `app/config/settings.py`
- `app/main.py`
- Any other files

---

## Implementation Notes

1. **Lazy Initialization**: The fallback LLM is created lazily (on first error) to avoid unnecessary initialization when Groq is working properly.

2. **Tool Binding**: Tool binding for fallback LLM follows the same conditional logic as the primary LLM (journey tools vs regular tools).

3. **Error Logging**: Error logging distinguishes between Groq errors and ChatGPT errors, making debugging easier.

4. **Token Usage Tracking**: The system maintains the same token usage tracking regardless of which provider is used.

5. **Provider Detection**: The system checks if the current provider is Groq before attempting fallback, ensuring fallback only happens when appropriate.

6. **Error Pattern Matching**: Multiple error patterns are checked to reliably detect Groq-specific errors:
   - Error message contains "groq"
   - Error type is "BadRequestError"
   - Error message contains "tool_use_failed"
   - Error message contains "Failed to call a function"

7. **State Preservation**: The fallback mechanism preserves all message state and context, ensuring seamless transition between providers.

---

## Error Scenarios Handled

### Scenario 1: Groq API Failure
- **Trigger**: Groq API returns an error
- **Action**: Automatically switch to ChatGPT
- **Result**: Request continues with ChatGPT

### Scenario 2: Groq Tool Call Error
- **Trigger**: Groq fails to handle tool calls properly
- **Action**: Automatically switch to ChatGPT with same tools
- **Result**: Request continues with ChatGPT using same tool binding

### Scenario 3: Groq Network Error
- **Trigger**: Network timeout or connection error with Groq
- **Action**: Automatically switch to ChatGPT
- **Result**: Request continues with ChatGPT

### Scenario 4: Both Providers Fail
- **Trigger**: Both Groq and ChatGPT fail
- **Action**: Raise original error with detailed logging
- **Result**: User receives error message, system logs full details

---

## Code Change Summary

### Total Changes Required:
- **2 files** to modify
- **3 new methods** to add
- **5 method calls** to replace
- **1 constructor modification** (add provider storage)
- **1 error handling block** to simplify

### Lines of Code:
- **~50 lines** of new code (helper methods)
- **~5 lines** modified (constructor)
- **~5 lines** replaced (invocation calls)
- **~30 lines** simplified (error handling)

**Total**: Approximately **90 lines** of changes across 2 files.

---

## Conclusion

This approach ensures that when Groq fails at any step, ChatGPT automatically takes over without changing any business logic or workflow. The implementation is minimal, transparent, and maintains full backward compatibility with existing functionality.

The fallback mechanism is designed to be:
- **Automatic**: No manual intervention required
- **Transparent**: Users don't notice the switch
- **Reliable**: Handles multiple error scenarios
- **Efficient**: Only creates fallback LLM when needed
- **Maintainable**: Clean separation of concerns
