"""
ViraAI EdTech Deep-Thinking ReAct Loop with Self-Validation Demo
=================================================================

1. ReAct Pattern (Reason - Act -Observe- Synthesize -Validate- Iterate)
2. Self-validation of results before returning to user
3. Iterative improvement until a quality threshold is met
4. Deep reasoning with explicit thought process (via LLM reasoning JSON)
5. Pydantic-based validation for:
   - Pipeline outputs
   - LLM reasoning/validation JSON
6. Course recommendations backed by MXBAI-embedded course catalog
   (courses_payload.pkl / courses_faiss.index), with clear source
   tagging per course.

Run: python viraai_edtech_deep_thinking.py

Requirements:
    pip install google-cloud-aiplatform pydantic
    # Optional (for direct FAISS index experimentation):
    # pip install faiss-cpu
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field, ValidationError

# Anthropic (Claude) imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[WARN] anthropic package not installed. Claude support disabled.")

try:
    from viraai_memory import ViraAIMemoryService
    VIRAAI_MEMORY_SERVICE = ViraAIMemoryService()
    print("[INFO] ViraAI Memory Service initialized correctly.")
except ImportError:
    VIRAAI_MEMORY_SERVICE = None
    print("[WARN] viraai_memory.py not found or failed to load. Memory layer disabled.")

# ============================================================================
# CONFIGURATION & INITIALIZATION
# ============================================================================


class ViraAIConfig:
    """Configuration for ViraAI EdTech deep-thinking system"""

    # Provider selection ("claude" only now)
    PROVIDER = os.getenv("VIRAAI_PROVIDER", "claude")
    
    # Claude/Anthropic configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
    
    # Try dynamic load (Streamlit Cloud priority)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
            ANTHROPIC_API_KEY = str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except:
        pass
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
    MODEL_NAME = CLAUDE_MODEL

    # ============================================================================

    # Quality thresholds
    QUALITY_THRESHOLD = 0.75  # Minimum quality score to accept result
    MAX_ITERATIONS = 3  # Maximum refinement loops

    # Agent configuration
    AGENT_TIMEOUT = 30  # seconds (conceptual timeout for calls / planning)
    
    # ============================================================================
    # PRODUCTION-READY CONFIGURATION (NEW)
    # ============================================================================
    
    # Timeout Configuration
    LLM_CALL_TIMEOUT = int(os.getenv("VIRAAI_TIMEOUT", "30"))  # seconds per LLM API call
    PIPELINE_TIMEOUT = int(os.getenv("VIRAAI_PIPELINE_TIMEOUT", "60"))  # seconds per pipeline execution
    
    # Retry Configuration
    MAX_RETRIES = int(os.getenv("VIRAAI_MAX_RETRIES", "3"))  # max retry attempts for failed calls
    BACKOFF_FACTOR = float(os.getenv("VIRAAI_BACKOFF_FACTOR", "2.0"))  # exponential backoff multiplier
    INITIAL_RETRY_DELAY = float(os.getenv("VIRAAI_INITIAL_DELAY", "2.0"))  # initial delay in seconds
    
    # Concurrency Configuration
    MAX_PARALLEL_OPERATIONS = int(os.getenv("VIRAAI_MAX_PARALLEL", "3"))  # max concurrent calls
    ENABLE_PARALLEL_EXECUTION = os.getenv("VIRAAI_ENABLE_PARALLEL", "true").lower() == "true"
    
    # Rate Limiting Configuration
    REQUESTS_PER_MINUTE = int(os.getenv("VIRAAI_RPM", "60"))  # max requests per minute
    ENABLE_RATE_LIMITING = os.getenv("VIRAAI_ENABLE_RATE_LIMIT", "true").lower() == "true"
    
    # Observability Configuration
    ENABLE_METRICS = os.getenv("VIRAAI_ENABLE_METRICS", "true").lower() == "true"
    VERBOSE_ERRORS = os.getenv("VIRAAI_VERBOSE_ERRORS", "false").lower() == "true"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return current configuration as a dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "quality_threshold": cls.QUALITY_THRESHOLD,
            "max_iterations": cls.MAX_ITERATIONS,
            "llm_timeout": cls.LLM_CALL_TIMEOUT,
            "pipeline_timeout": cls.PIPELINE_TIMEOUT,
            "max_retries": cls.MAX_RETRIES,
            "backoff_factor": cls.BACKOFF_FACTOR,
            "max_parallel": cls.MAX_PARALLEL_OPERATIONS,
            "parallel_enabled": cls.ENABLE_PARALLEL_EXECUTION,
            "rpm_limit": cls.REQUESTS_PER_MINUTE,
            "rate_limit_enabled": cls.ENABLE_RATE_LIMITING,
            "metrics_enabled": cls.ENABLE_METRICS,
        }



# Enhanced MXBAI Configuration Constants (for documentation / reference)
MXBAI_CONFIG = {
    "model_id": "mixedbread-ai/mxbai-embed-large-v1",
    "dimensions": 1024,
    "max_length": 512,
    "similarity_threshold": 0.05,
    "top_k_results": 50,
    "normalize_embeddings": True,
}

# Vector Database Connection Settings
# ----------------------------------------------------------------------------



# ============================================================================
# UNIFIED MODEL WRAPPER (Multi-Provider Support)
# ============================================================================

class UnifiedModel:
    """
    Unified wrapper for Claude LLM models.
    Provides generate_content() with built-in retry for transient API errors.
    """
    def __init__(self, provider: str, claude_client=None, claude_model_name=None):
        self.provider = provider
        self.claude_client = claude_client
        self.claude_model_name = claude_model_name
        
    def generate_content(self, prompt: str, **kwargs):
        """
        Content generation with automatic retry for transient API errors.
        Returns a response object with .text attribute.
        """
        if self.provider != "claude":
            raise ValueError(f"Unsupported provider: {self.provider}. Only 'claude' is supported.")

        max_retries = kwargs.pop("_retries", 3)
        for attempt in range(max_retries):
            try:
                response = self.claude_client.messages.create(
                    model=self.claude_model_name,
                    max_tokens=kwargs.get("max_tokens", 4096),
                    messages=[{"role": "user", "content": prompt}]
                )
                class ClaudeResponse:
                    def __init__(self, content):
                        self.content = [type('obj', (object,), {'text': content})()]
                        self.text = content
                return ClaudeResponse(response.content[0].text)
            except Exception as e:
                error_str = str(e)
                is_transient = any(code in error_str for code in ["500", "529", "overloaded", "Internal server"])
                if is_transient and attempt < max_retries - 1:
                    wait = (attempt + 1) * 2
                    print(f"   [RETRY] API transient error (attempt {attempt+1}/{max_retries}). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                raise


def initialize_model(provider: str = None, model_name: str = None) -> UnifiedModel:
    """
    Initialize the model based on provider selection.
    
    Args:
        provider: "claude" (defaults to ViraAIConfig.PROVIDER)
        model_name: specific model name (defaults to config values)
    
    Returns:
        UnifiedModel instance
    """
    provider = provider or ViraAIConfig.PROVIDER
    
    if provider == "claude":
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        model_name = model_name or ViraAIConfig.CLAUDE_MODEL
        claude_client = anthropic.Anthropic(api_key=ViraAIConfig.ANTHROPIC_API_KEY)
        print(f"[OK] Claude model initialized: {model_name}")
        return UnifiedModel(provider="claude", claude_client=claude_client, claude_model_name=model_name)
    
    else:
        raise ValueError(f"Provider {provider} is not supported or deprecated. Use 'claude'.")


# ============================================================================
# PRODUCTION-READY INFRASTRUCTURE (NEW)
# ============================================================================

# ---- Custom Exceptions -------------------------------------------------------

class ViraAIError(Exception):
    """Base exception for ViraAI-related errors"""
    pass

class RateLimitError(ViraAIError):
    """Raised when rate limit is exceeded"""
    pass

class TimeoutError(ViraAIError):
    """Raised when operation times out"""
    pass

class ValidationError(ViraAIError):
    """Raised when validation fails"""
    pass

class PipelineExecutionError(ViraAIError):
    """Raised when pipeline execution fails"""
    pass


# ---- Async Executor ----------------------------------------------------------

class AsyncExecutor:
    """
    Manages parallel execution of LLM API calls with concurrency limiting.
    Provides fallback to sequential execution if async fails.
    """
    
    def __init__(self, max_parallel: int = None):
        self.max_parallel = max_parallel or ViraAIConfig.MAX_PARALLEL_OPERATIONS
        self.enabled = ViraAIConfig.ENABLE_PARALLEL_EXECUTION
    
    def execute_parallel(self, tasks: List[tuple], timeout: Optional[float] = None) -> List[Any]:
        """
        Execute multiple tasks in parallel using threading.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            timeout: Optional timeout for each task
            
        Returns:
            List of results in the same order as input tasks
        """
        if not self.enabled or len(tasks) <= 1:
            # Fallback to sequential execution
            return self._execute_sequential(tasks)
        
        try:
            import concurrent.futures
            import threading
            
            results = [None] * len(tasks)
            semaphore = threading.Semaphore(self.max_parallel)
            
            def execute_with_semaphore(index, func, args, kwargs):
                with semaphore:
                    try:
                        return index, func(*args, **kwargs)
                    except Exception as e:
                        return index, e
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                futures = []
                for i, (func, args, kwargs) in enumerate(tasks):
                    future = executor.submit(execute_with_semaphore, i, func, args, kwargs)
                    futures.append(future)
                
                # Wait for all futures with timeout
                task_timeout = timeout or ViraAIConfig.LLM_CALL_TIMEOUT
                for future in concurrent.futures.as_completed(futures, timeout=task_timeout * len(tasks)):
                    index, result = future.result()
                    if isinstance(result, Exception):
                        raise result
                    results[index] = result
            
            return results
            
        except Exception as e:
            print(f"   [WARN] Parallel execution failed: {e}. Falling back to sequential.")
            return self._execute_sequential(tasks)
    
    def _execute_sequential(self, tasks: List[tuple]) -> List[Any]:
        """Fallback sequential execution"""
        results = []
        for func, args, kwargs in tasks:
            result = func(*args, **kwargs)
            results.append(result)
        return results


# ---- Rate Limiter ------------------------------------------------------------

class RateLimiter:
    """
    Token bucket rate limiter to prevent API quota exhaustion.
    Tracks requests per minute with sliding window.
    """
    
    def __init__(self, requests_per_minute: int = None):
        self.rpm = requests_per_minute or ViraAIConfig.REQUESTS_PER_MINUTE
        self.enabled = ViraAIConfig.ENABLE_RATE_LIMITING
        self.request_times: List[float] = []
        self.lock = None
        
        if self.enabled:
            import threading
            self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        if not self.enabled:
            return
        
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            cutoff = now - 60.0
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Check if we're at the limit
            if len(self.request_times) >= self.rpm:
                # Calculate wait time
                oldest = self.request_times[0]
                wait_time = 60.0 - (now - oldest) + 0.1  # Add small buffer
                if wait_time > 0:
                    print(f"   [RATE_LIMIT] Waiting {wait_time:.1f}s to respect rate limit...")
                    time.sleep(wait_time)
                    now = time.time()
            
            # Record this request
            self.request_times.append(now)
    
    def get_current_rate(self) -> float:
        """Get current requests per minute"""
        if not self.enabled:
            return 0.0
        
        with self.lock:
            now = time.time()
            cutoff = now - 60.0
            recent = [t for t in self.request_times if t > cutoff]
            return len(recent)


# ---- Error Handler -----------------------------------------------------------

class ErrorHandler:
    """
    Centralized error handling with retry logic and detailed logging.
    """
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.total_retries = 0
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle an error and return structured error information.
        
        Args:
            error: The exception that occurred
            context: Context string describing where the error occurred
            
        Returns:
            Dictionary with error details
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Track error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Classify error
        is_rate_limit = any(x in error_msg for x in ["429", "ResourceExhausted", "Quota exceeded"])
        is_timeout = "timeout" in error_msg.lower()
        is_network = any(x in error_msg for x in ["connection", "network", "unreachable"])
        
        error_info = {
            "type": error_type,
            "message": error_msg,
            "context": context,
            "is_rate_limit": is_rate_limit,
            "is_timeout": is_timeout,
            "is_network": is_network,
            "is_retryable": is_rate_limit or is_timeout or is_network,
            "timestamp": datetime.now().isoformat(),
        }
        
        if ViraAIConfig.VERBOSE_ERRORS:
            import traceback
            error_info["traceback"] = traceback.format_exc()
        
        return error_info
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors encountered"""
        return {
            "error_counts": self.error_counts.copy(),
            "total_retries": self.total_retries,
            "total_errors": sum(self.error_counts.values()),
        }


# ---- Metrics Collector -------------------------------------------------------

class MetricsCollector:
    """
    Collects execution metrics for observability and debugging.
    """
    
    def __init__(self):
        self.enabled = ViraAIConfig.ENABLE_METRICS
        self.llm_calls = 0
        self.llm_latencies: List[float] = []
        self.pipeline_executions: Dict[str, int] = {}
        self.pipeline_latencies: Dict[str, List[float]] = {}
        self.parallel_executions = 0
        self.sequential_executions = 0
        self.start_time = time.time()
    
    def record_llm_call(self, latency: float):
        """Record an LLM API call"""
        if not self.enabled:
            return
        self.llm_calls += 1
        self.llm_latencies.append(latency)
    
    def record_pipeline(self, pipeline_name: str, latency: float):
        """Record a pipeline execution"""
        if not self.enabled:
            return
        self.pipeline_executions[pipeline_name] = self.pipeline_executions.get(pipeline_name, 0) + 1
        if pipeline_name not in self.pipeline_latencies:
            self.pipeline_latencies[pipeline_name] = []
        self.pipeline_latencies[pipeline_name].append(latency)
    
    def record_execution_mode(self, is_parallel: bool):
        """Record whether execution was parallel or sequential"""
        if not self.enabled:
            return
        if is_parallel:
            self.parallel_executions += 1
        else:
            self.sequential_executions += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.enabled:
            return {"enabled": False}
        
        total_time = time.time() - self.start_time
        
        avg_llm_latency = (
            sum(self.llm_latencies) / len(self.llm_latencies)
            if self.llm_latencies else 0.0
        )
        
        return {
            "enabled": True,
            "total_execution_time": round(total_time, 2),
            "llm_calls": self.llm_calls,
            "avg_llm_latency": round(avg_llm_latency, 2),
            "pipeline_executions": self.pipeline_executions.copy(),
            "parallel_executions": self.parallel_executions,
            "sequential_executions": self.sequential_executions,
            "execution_mode": "parallel" if self.parallel_executions > 0 else "sequential",
        }


# Global instances
ASYNC_EXECUTOR = AsyncExecutor()
RATE_LIMITER = RateLimiter()
ERROR_HANDLER = ErrorHandler()
METRICS_COLLECTOR = MetricsCollector()





# ============================================================================
# Pydantic MODELS (VALIDATION LAYER)
# ============================================================================

# ---- Pipeline output models -------------------------------------------------


class SkillGapAnalysis(BaseModel):
    current_skills: List[str]
    required_skills: List[str]
    skill_gaps: List[str]
    proficiency_levels: Dict[str, float]
    estimated_learning_time: str
    confidence: float


class CourseInfo(BaseModel):
    skill: str
    course_name: str
    provider: str
    duration: str
    rating: float
    price: str
    # Source indicator for recommendation origin
    # "vector_catalog" (from MXBAI course payload) or "llm_fallback"
    source: str = "vector_catalog"
    # Mandatory origin label: "internal" or "external"
    course_origin: str = "internal"
    # Course URL for hyperlinking in final output
    url: str = ""
    # Explicit course ID
    course_id: str = ""


class CourseRecommendations(BaseModel):
    recommended_courses: List[CourseInfo]
    total_estimated_time: str
    total_cost: str
    learning_path_order: List[str]
    confidence: float


class CareerPathAnalysis(BaseModel):
    transition_difficulty: str
    typical_timeline: str
    intermediate_roles: List[str]
    success_factors: List[str]
    salary_progression: Dict[str, str]
    confidence: float


class JobMarketAnalysis(BaseModel):
    total_openings: int
    demand_trend: str
    top_companies: List[str]
    average_salary: str
    required_experience: str
    top_skills: List[str]
    confidence: float


class SkillsFetcherOutput(BaseModel):
    source: str
    role: Optional[str] = None
    skills: List[str]
    inferred_level: Optional[str] = None
    confidence: float


class IntentAnalysis(BaseModel):
    primary_intent: str
    secondary_intents: List[str]
    signals_used: List[str]
    time_window: str
    confidence: float


class ContextState(BaseModel):
    user_id: str
    session_id: str
    recent_queries: List[str]
    recent_skills: List[str]
    last_recommended_courses: List[str]
    timestamp: str
    notes: Optional[str] = None
    confidence: float


class OrchestrationStep(BaseModel):
    pipeline: str
    reason: str
    parameters: Dict[str, Any]


class OrchestrationPlan(BaseModel):
    query: str
    suggested_pipelines: List[OrchestrationStep]
    expected_coverage: str
    confidence: float


# ---- LLM reasoning & validation models -------------------------------------


class ReasoningPlannedAction(BaseModel):
    pipeline: str
    why: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_output: Optional[str] = None

    class Config:
        extra = "allow"


class ReasoningOutput(BaseModel):
    reasoning: str
    planned_actions: List[ReasoningPlannedAction] = Field(default_factory=list)
    strategy: Optional[str] = None

    class Config:
        extra = "allow"


class ValidationOutput(BaseModel):
    completeness: float = 0.0
    accuracy: float = 0.0
    relevance: float = 0.0
    actionability: float = 0.0
    clarity: float = 0.0
    overall_quality: Optional[float] = None
    passes_threshold: Optional[bool] = None
    issues: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    suggestions_for_improvement: List[str] = Field(default_factory=list)
    quality_score: Optional[float] = None

    class Config:
        extra = "allow"


class ReflectionOutput(BaseModel):
    success: bool
    mistake: Optional[str] = None
    correction: Optional[str] = None
    why_success_or_fail: str

    class Config:
        extra = "allow"


class IntermediateValidationOutput(BaseModel):
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    confidence: float

    class Config:
        extra = "allow"


    class Config:
        extra = "allow"


class AlternativePath(BaseModel):
    name: str
    description: str
    pros: str
    cons: str
    score: float

    class Config:
        extra = "allow"


class PathExplorationOutput(BaseModel):
    paths: List[AlternativePath]
    selected_path: str
    reasoning: str

    class Config:
        extra = "allow"


class RecoveryStrategy(BaseModel):
    strategy_name: str
    description: str
    action_plan: str
    confidence: float

    class Config:
        extra = "allow"


# ============================================================================
# COURSE VECTOR STORE (MXBAI payload integration)
# ============================================================================


class CourseVectorStore:
    """
    This implementation:
      - Interfaces with the CareerVira Vector Search API
      - Provides course recommendations via remote vector storage
      - Maintains strict internal-only course enforcement
    """

    # Vector Database API configuration
    VECTOR_API_URL = "http://43.204.221.112/vector/v0/search"
    VECTOR_API_KEY = "api_vecToR#3"
    VECTOR_API_COLLECTION = "courses_all"

    def __init__(self):
        self.available = True  # Always available via Vector DB API
        
        # [CACHE] Per-query model cache
        self._model_cache = {}
        
    def get_model(self, key: str, loader_func):
        """
        Retrieves a model from the cache or loads it if missing.
        Ensures models are reused within a single query lifecycle.
        """
        if key not in self._model_cache:
            print(f"[CACHE] Loading model/resource for key: {key}")
            self._model_cache[key] = loader_func()
        return self._model_cache[key]

    def clear_cache(self):
        """
        Clears the per-query cache to ensure clean state for new queries.
        """
        if self._model_cache:
            print(f"[CACHE] Clearing {len(self._model_cache)} items from query-local cache.")
            self._model_cache.clear()



    @staticmethod
    def _normalize_tokens(text: str) -> List[str]:
        if not text:
            return []
        # simple tokenization by splitting on spaces and commas
        tokens = []
        for part in text.replace(",", " ").split():
            t = part.strip().lower()
            if t:
                tokens.append(t)
        return tokens

    @staticmethod
    def _normalize_skill_tokens(skill_text: str) -> List[str]:
        if not skill_text:
            return []
        tokens = [s.strip().lower() for s in skill_text.split(",")]
        return [t for t in tokens if t]

    def expand_query(self, query: str) -> List[str]:
        """
        Expands the user query into related synonyms, variations, and domain-specific terms.
        Uses LLM to generate query expansions for better recall.
        """
        if not query:
            return []

        # Try LLM-based expansion
        try:
            expansion_prompt = f"""
You are a query expansion expert for an EdTech platform. Given a user query, generate 3-5 related search terms, synonyms, or variations that would help find relevant courses.

USER QUERY: {query}

Generate expansions as a JSON array of strings. Include:
- Synonyms and variations
- Related technical terms
- Common abbreviations or full forms
- Domain-specific terminology

Example: For "ML courses", return: ["machine learning", "deep learning", "neural networks", "AI courses", "artificial intelligence"]

Respond ONLY with a JSON array of strings:
["term1", "term2", "term3", ...]
"""
            # Use the global model if available
            if hasattr(self, 'model') and self.model:
                response = self.model.generate_content(expansion_prompt)
                raw_text = response.text.strip()
                # Extract JSON array
                if raw_text.startswith('[') and raw_text.endswith(']'):
                    import json
                    expanded_terms = json.loads(raw_text)
                    if isinstance(expanded_terms, list):
                        return [query] + [str(t).lower() for t in expanded_terms if t]
        except Exception as e:
            print(f"[WARN] Query expansion failed: {e}. Using original query only.")

        # Fallback: return original query
        return [query]

    def rerank_semantic(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks candidate courses using semantic similarity (embeddings).
        Falls back to original order if sentence-transformers is not available.
        """
        if not candidates:
            return candidates

        try:
            from sentence_transformers import SentenceTransformer, util
            
            # [CACHE] Use cached model loader
            def load_st_model():
                return SentenceTransformer('all-MiniLM-L6-v2')
                
            model = self.get_model('st_minilm', load_st_model)
            
            # Encode query
            query_embedding = model.encode(query, convert_to_tensor=True)
            
            # Encode candidate course descriptions
            course_texts = []
            for course in candidates:
                # Combine title, description, and skills for better semantic matching
                title = str(course.get('title', ''))
                desc = str(course.get('description', ''))
                skills = str(course.get('skills', ''))
                combined_text = f"{title}. {desc}. Skills: {skills}"
                course_texts.append(combined_text)
            
            course_embeddings = model.encode(course_texts, convert_to_tensor=True)
            
            # Compute cosine similarities
            similarities = util.cos_sim(query_embedding, course_embeddings)[0]
            
            # Sort candidates by similarity score (descending)
            scored_candidates = list(zip(candidates, similarities.cpu().numpy()))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            reranked = [candidate for candidate, score in scored_candidates]
            print(f"[OK] Semantic reranking applied to {len(reranked)} candidates")
            return reranked
            
        except ImportError:
            print("[INFO] sentence-transformers not installed. Skipping semantic reranking.")
            return candidates
        except Exception as e:
            # Handle TQDM/Stream issues by checking for isatty manually if needed
            print(f"[WARN] Semantic reranking failed: {e}. Returning original order.")
            return candidates
            return candidates

    def score_with_cross_encoder(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Assigns confidence and relevance scores to each candidate using cross-encoder.
        Scores range from 0.0 (low) to 1.0 (high confidence).
        """
        if not candidates:
            return candidates

        try:
            from sentence_transformers import CrossEncoder
            
            # [CACHE] Use cached model loader
            def load_ce_model():
                return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                
            model = self.get_model('ce_msmarco', load_ce_model)
            
            # Prepare query-course pairs for scoring
            pairs = []
            for course in candidates:
                title = str(course.get('title', ''))
                desc = str(course.get('description', ''))
                skills = str(course.get('skills', ''))
                combined_text = f"{title}. {desc}. Skills: {skills}"
                pairs.append([query, combined_text])
            
            # Get relevance scores
            scores = model.predict(pairs)
            
            # Normalize scores to 0-1 range and categorize
            import numpy as np
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # Add scores to candidates
            scored_candidates = []
            for i, course in enumerate(candidates):
                course_copy = course.copy()
                relevance_score = float(scores_normalized[i])
                course_copy['relevance_score'] = relevance_score
                course_copy['confidence_score'] = relevance_score  # Same for now
                
                # Categorize confidence
                if relevance_score >= 0.7:
                    course_copy['confidence_category'] = 'high'
                elif relevance_score >= 0.4:
                    course_copy['confidence_category'] = 'medium'
                else:
                    course_copy['confidence_category'] = 'speculative'
                
                scored_candidates.append(course_copy)
            
            # Sort by relevance score
            scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            print(f"[OK] Cross-encoder scoring applied to {len(scored_candidates)} candidates")
            return scored_candidates
            
        except ImportError:
            print("[INFO] sentence-transformers cross-encoder not available. Using LLM-based scoring fallback.")
            return self._llm_based_scoring(query, candidates)
        except Exception as e:
            print(f"[WARN] Cross-encoder scoring failed: {e}. Returning original order.")
            return candidates

    def _llm_based_scoring(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback LLM-based scoring when cross-encoder is not available.
        """
        if not hasattr(self, 'model') or not self.model:
            # No LLM available, return candidates with default scores
            for course in candidates:
                course['relevance_score'] = 0.5
                course['confidence_score'] = 0.5
                course['confidence_category'] = 'medium'
            return candidates
        
        try:
            # Score in batches to avoid token limits
            scored_candidates = []
            for course in candidates[:10]:  # Limit to top 10 for efficiency
                title = str(course.get('title', ''))
                desc = str(course.get('description', ''))[:200]  # Truncate description
                
                scoring_prompt = f"""
Rate the relevance of this course to the query on a scale of 0.0 to 1.0.

QUERY: {query}
COURSE: {title}
DESCRIPTION: {desc}

Respond with ONLY a number between 0.0 and 1.0:
"""
                response = self.model.generate_content(scoring_prompt)
                try:
                    score = float(response.text.strip())
                    score = max(0.0, min(1.0, score))  # Clamp to 0-1
                except:
                    score = 0.5  # Default
                
                course_copy = course.copy()
                course_copy['relevance_score'] = score
                course_copy['confidence_score'] = score
                course_copy['confidence_category'] = 'high' if score >= 0.7 else ('medium' if score >= 0.4 else 'speculative')
                scored_candidates.append(course_copy)
            
            # Add remaining candidates with default scores
            for course in candidates[10:]:
                course_copy = course.copy()
                course_copy['relevance_score'] = 0.3
                course_copy['confidence_score'] = 0.3
                course_copy['confidence_category'] = 'speculative'
                scored_candidates.append(course_copy)
            
            scored_candidates.sort(key=lambda x: x['relevance_score'], reverse=True)
            print(f"[OK] LLM-based scoring applied to {len(scored_candidates)} candidates")
            return scored_candidates
            
        except Exception as e:
            print(f"[WARN] LLM-based scoring failed: {e}. Returning original order.")
            for course in candidates:
                course['relevance_score'] = 0.5
                course['confidence_score'] = 0.5
                course['confidence_category'] = 'medium'
            return candidates

    def validate_catalog_existence(self, course: Dict[str, Any]) -> Dict[str, Any]:
        """
        In API-first mode, we trust results returned directly by the internal Vector API.
        """
        course_copy = course.copy()
        course_copy['hallucinated'] = False
        course_copy['validation_method'] = 'api_trusted'
        return course_copy

    def filter_hallucinated_courses(
        self, 
        candidates: List[Dict[str, Any]], 
        replace_with_real: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Bypasses local hallucination filtering as the Vector DB API is the source of truth.
        """
        return candidates

    def recommend_courses_for_skill_gaps(
        self,
        skill_gaps: List[str],
        user_query: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Recommend top-k courses via the external Vector Database API.

        Calls POST http://service.careervira.com/vector/v0/search with the
        combined query built from skill_gaps and user_query, using the
        existing top_k limit passed by the caller.

        Returns:
            List[Dict] in the same structure expected by course_recommender:
            each dict has keys like title, partner_name, duration, price,
            average_rating, skills, course_origin, url, description, etc.
        """
        # Build the query string from skill_gaps and optional user_query
        query_parts = list(skill_gaps or [])
        if user_query:
            query_parts.insert(0, user_query)
        query_string = " ".join(query_parts).strip()

        if not query_string:
            return []

        print(f"[VECTOR-API] Querying Vector DB API with: '{query_string}' (limit={top_k})")

        try:
            response = requests.post(
                self.VECTOR_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.VECTOR_API_KEY.strip(),
                },
                json={
                    "collection": self.VECTOR_API_COLLECTION,
                    "query": query_string,
                    "limit": top_k,
                },
                timeout=15,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[VECTOR-API] API request failed: {e}")
            return []

        try:
            api_data = response.json()
        except ValueError:
            print("[VECTOR-API] Failed to parse API response as JSON.")
            return []

        # ---- Normalise API response into the structure downstream expects ----
        # The API may return results under a key like "results", "data",
        # "matches", or directly as a list.  Handle all common shapes.
        if isinstance(api_data, list):
            raw_results = api_data
        elif isinstance(api_data, dict):
            raw_results = (
                api_data.get("results")
                or api_data.get("data")
                or api_data.get("matches")
                or api_data.get("courses")
                or []
            )
            # If the extracted value is still a dict (single result), wrap it
            if isinstance(raw_results, dict):
                raw_results = [raw_results]
        else:
            raw_results = []

        if not raw_results:
            print("[VECTOR-API] API returned 0 results.")
            return []

        # ---- Map each API result to the dict shape consumed by course_recommender ----
        api_candidates: List[Dict[str, Any]] = []
        for item in raw_results:
            # Handle nested payload (some vector DBs wrap the document)
            payload = item.get("payload", item) if isinstance(item, dict) else item
            if not isinstance(payload, dict):
                continue

            # CRITICAL: Use point_id as mandatory ID. Fallback to id if point_id is missing.
            course_id = str(item.get("point_id") or item.get("id") or "")
            # CRITICAL: Use original_course_url as mandatory URL.
            course_url = str(payload.get("original_course_url") or "")

            # If mandatory fields are missing, EXCLUDE the course
            if not course_id or not course_url:
                continue

            course_dict: Dict[str, Any] = {
                "course_id": course_id,
                "title": payload.get("title", payload.get("course_name", "Untitled Course")),
                "partner_name": payload.get("partner_name", payload.get("provider_name", payload.get("provider", ""))),
                "provider_name": payload.get("provider_name", payload.get("partner_name", payload.get("provider", ""))),
                "duration": payload.get("duration", "Self-paced"),
                "price": payload.get("price", ""),
                "average_rating": payload.get("average_rating", payload.get("rating", 0)),
                "skills": payload.get("skills", ""),
                "url": course_url,
                "description": payload.get("description", ""),
                "categories": payload.get("categories", ""),
                "sub_categories": payload.get("sub_categories", ""),
                "topics": payload.get("topics", ""),
                "reviews_count": payload.get("reviews_count", 0),
                "course_origin": "internal",
            }
            api_candidates.append(course_dict)

        if not api_candidates:
            return []

        # ---- LOCAL INTELLIGENCE LAYER (Accuracy Boost) ----
        # 1. Semantic Reranking: Re-order API results using local embeddings
        print(f"[LOCAL] Applying semantic reranking to {len(api_candidates)} API candidates...")
        reranked = self.rerank_semantic(query_string, api_candidates)

        # 2. Cross-Encoder Scoring: Assign deep relevance confidence
        print(f"[LOCAL] Assigning confidence scores via cross-encoder...")
        final_scored = self.score_with_cross_encoder(query_string, reranked)

        print(f"[VECTOR-API] Final delivery: {len(final_scored[:top_k])} high-confidence courses.")
        return final_scored[:top_k]


# single global instance used by pipeline
COURSE_STORE = CourseVectorStore()

# Set model reference for query expansion (will be set after model initialization)
def set_course_store_model(model):
    """Set the LLM model reference for COURSE_STORE to enable query expansion."""
    COURSE_STORE.model = model


# ============================================================================
# UNIVERSAL SEARCH SYSTEM
# ============================================================================


class UniversalSearch:
    """
    Universal search system that handles general knowledge queries and partial matches.
    Works alongside existing pipelines to provide comprehensive answers.
    """
    
    def __init__(self, model=None):
        self.model = model
        
    def search(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Perform universal search using the LLM's knowledge base.
        
        Args:
            query: User's search query
            context: Optional context for more relevant results
            
        Returns:
            Dictionary with search results
        """
        if not self.model:
            return {
                "found": False,
                "summary": "Search unavailable (Model not connected)",
                "snippets": [],
                "confidence": 0.0
            }
            
        try:
            search_prompt = f"""You are a knowledgeable assistant for an EdTech platform called CareerVira.
Answer the following query with accurate, helpful information.

If the query is about CareerVira, explain that it's an educational technology platform 
that helps users with career guidance, skill development, and learning path recommendations.

QUERY: {query}

Provide a clear, concise answer. If you're unsure, say so.

Respond in JSON format:
{{
    "found": true/false,
    "summary": "Main answer to the query",
    "snippets": ["Key point 1", "Key point 2", ...],
    "confidence": 0.0-1.0
}}
"""
            response = self.model.generate_content(search_prompt)
            text = response.text.strip()
            
            # Parse JSON
            import json
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
                
            result = json.loads(text.strip())
            return result
            
        except Exception as e:
            print(f"[WARN] Universal search failed: {e}")
            return {
                "found": False,
                "summary": f"Search error: {str(e)}",
                "snippets": [],
                "confidence": 0.0
            }


# Global universal search instance
UNIVERSAL_SEARCH = None

def set_universal_search_model(model):
    """Initialize universal search with the model."""
    global UNIVERSAL_SEARCH
    UNIVERSAL_SEARCH = UniversalSearch(model)


# ============================================================================
# DUMMY PIPELINES (actual RAG pipelines would replace these)
# ============================================================================


class DummyPipelines:
    """
    These simulate actual EdTech RAG pipelines.
    In this version, course_recommender is backed by your MXBAI course catalog.
    """

    # ------------------------------------------------------------------ #
    # 1. SKILL GAP ANALYZER
    # ------------------------------------------------------------------ #

    @staticmethod
    def skill_gap_analyzer(user_profile: Dict, target_role: str) -> Dict:
        """
        Simulates a skill gap analysis pipeline:
        Desired role skills - current role/user profile skills.
        """
        print("    [PIPE] Running skill_gap_analyzer...")
        time.sleep(0.5)

        current_skills = user_profile.get(
            "current_skills", ["Python", "Data Analysis", "SQL"]
        )
        target_role_skills_map = {
            "Machine Learning Engineer": [
                "Python",
                "Machine Learning",
                "Deep Learning",
                "MLOps",
                "SQL",
            ],
            "Data Scientist": [
                "Python",
                "Statistics",
                "Machine Learning",
                "SQL",
                "Experiment Design",
            ],
            "Senior Data Analyst": [
                "SQL",
                "Advanced Analytics",
                "Stakeholder Communication",
                "Business Acumen",
                "Dashboard Design",
            ],
            "AI Engineer": [
                "Python",
                "Linear Algebra",
                "Machine Learning",
                "Deep Learning",
                "Data Structures",
            ],
            "Product Manager": [
                "Product Strategy",
                "User Research",
                "Agile Methodologies",
                "Data Analysis",
                "Stakeholder Management",
            ],
            "UX Designer": [
                "Figma",
                "User Research",
                "Prototyping",
                "Interaction Design",
                "Wireframing",
            ],
        }
        required_skills = target_role_skills_map.get(
            target_role, ["Python", "Problem Solving"]
        )
        gaps = [s for s in required_skills if s not in current_skills]

        raw = {
            "current_skills": current_skills,
            "required_skills": required_skills,
            "skill_gaps": gaps,
            "proficiency_levels": {
                skill: user_profile.get("proficiency", {}).get(skill, 0.5)
                for skill in current_skills
            },
            "estimated_learning_time": "6-8 months",
            "confidence": 0.85,
        }

        validated = SkillGapAnalysis(**raw)
        return validated.dict()

    # ------------------------------------------------------------------ #
    # 2. COURSE RECOMMENDER (BACKED BY YOUR COURSE CATALOG)
    # ------------------------------------------------------------------ #

    @staticmethod
    def course_recommender(
        skill_gaps: List[str],
        learning_style: str = "online",
        user_query: Optional[str] = None,
    ) -> Dict:
        """
        Course recommendation pipeline backed by the MXBAI course catalog.
        Falls back to synthetic LLM courses if the catalog is unavailable or returns too few results.
        """
        print("    [PIPE] Running course_recommender...")
        time.sleep(0.3)

        courses_info: List[CourseInfo] = []
        vector_used = False
        fallback_used = False

        if COURSE_STORE.available:
            print(
                f"    [INFO] Using Vector DB API for course retrieval"
            )
            raw_courses = COURSE_STORE.recommend_courses_for_skill_gaps(
                skill_gaps, user_query=user_query, top_k=20
            )
            
            # If we found internal courses, use them!
            if raw_courses:
                 vector_used = True

            for course in raw_courses:
                title = str(course.get("title", "Untitled Course"))
                partner_name = course.get("partner_name") or course.get(
                    "provider_name"
                )
                provider = (
                    str(partner_name)
                    if partner_name not in (None, "", "nan")
                    else "Unknown Provider"
                )
                duration = str(course.get("duration", "Self-paced"))
                price_val = course.get("price", "")
                price = str(price_val) if price_val not in (None, "") else "N/A"

                try:
                    rating = float(course.get("average_rating", 0) or 0)
                except Exception:
                    rating = 0.0

                # Choose a representative skill label for this course
                skill_for_course = (
                    skill_gaps[0] if skill_gaps else "General Skill"
                )
                skills_text = str(course.get("skills", ""))
                course_tokens = CourseVectorStore._normalize_skill_tokens(
                    skills_text
                )
                for gap in skill_gaps or []:
                    if gap.strip().lower() in course_tokens:
                        skill_for_course = gap
                        break
                
                # Determine origin (should be 'internal' from recommend_courses_for_skill_gaps)
                origin = course.get('course_origin', 'internal')

                course_url = str(course.get("url", course.get("original_course_url", ""))) or ""
                course_id = str(course.get("course_id", ""))

                if not course_url.strip() or not course_id.strip():
                    continue

                ci = CourseInfo(
                    skill=skill_for_course,
                    course_name=title,
                    provider=provider,
                    duration=duration,
                    rating=rating,
                    price=price,
                    source="vector_catalog",
                    course_origin=origin,
                    url=course_url,
                    course_id=course_id
                )
                courses_info.append(ci)

            if courses_info:
                vector_used = True

        # No external fallbacks allowed! Strictly only internal courses.
        if not courses_info:
            print("    [WARN] No internal matches found and external fallback is disabled by policy.")
            return {
                "recommended_courses": [],
                "total_estimated_time": "0 hours",
                "total_cost": "$0",
                "learning_path_order": [],
                "confidence": 0.0,
                "vector_courses_used": False,
                "llm_fallback_used": False,
                "num_vector_courses": 0,
                "num_fallback_courses": 0
            }

        raw = {
            "recommended_courses": [c.dict() for c in courses_info],
            "total_estimated_time": f"{len(courses_info) * 30} hours",
            "total_cost": f"${len(courses_info) * 79}",
            "learning_path_order": skill_gaps or ["General Skill"],
            "confidence": 0.85 if vector_used else 0.80,
        }

        validated = CourseRecommendations(**raw)
        result_dict = validated.dict()
        # attach metadata for ReAct / validation
        result_dict["vector_courses_used"] = vector_used
        result_dict["llm_fallback_used"] = fallback_used
        result_dict["num_vector_courses"] = sum(
            1 for c in result_dict["recommended_courses"] if c["source"] == "vector_catalog"
        )
        result_dict["num_fallback_courses"] = sum(
            1 for c in result_dict["recommended_courses"] if c["source"] == "llm_fallback"
        )
        return result_dict

    # ------------------------------------------------------------------ #
    # 3. SKILLS FETCHER
    # ------------------------------------------------------------------ #

    @staticmethod
    def skills_fetcher(
        user_profile: Optional[Dict] = None,
        role: Optional[str] = None,
        course_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Simulates fetching skills from:
        - Job role taxonomy
        - User profile
        - Selected courses
        """
        print("    [PIPE] Running skills_fetcher...")
        time.sleep(0.5)

        base_skills = set()
        source_parts = []

        if role:
            source_parts.append("role")
            role_skill_map = {
                "Data Analyst": ["SQL", "Excel", "Data Visualization"],
                "Machine Learning Engineer": [
                    "Python",
                    "Machine Learning",
                    "MLOps",
                ],
                "Frontend Developer": ["JavaScript", "React", "CSS"],
                "Senior Data Analyst": [
                    "SQL",
                    "Advanced Analytics",
                    "Business Storytelling",
                ],
                "AI Engineer": ["Python", "Machine Learning", "Deep Learning"],
            }
            base_skills.update(role_skill_map.get(role, []))

        if user_profile:
            source_parts.append("user_profile")
            base_skills.update(user_profile.get("current_skills", []))

        if course_ids:
            source_parts.append("courses")
            # Dummy: we pretend courses add some skills
            for cid in course_ids:
                if "ml" in cid.lower():
                    base_skills.add("Machine Learning")
                if "ds" in cid.lower():
                    base_skills.add("Data Science")

        inferred_level = "intermediate" if len(base_skills) > 4 else "beginner"

        raw = {
            "source": "+".join(source_parts) if source_parts else "unknown",
            "role": role,
            "skills": sorted(base_skills),
            "inferred_level": inferred_level,
            "confidence": 0.78,
        }

        validated = SkillsFetcherOutput(**raw)
        return validated.dict()

    # ------------------------------------------------------------------ #
    # 4. CLAUDE FALLBACK SEARCH PIPELINE
    # ------------------------------------------------------------------ #

    @staticmethod
    def claude_fallback_pipeline(query: str, missing_info_type: str = "general_info") -> Dict:
        """
        Fallback pipeline that uses the LLM to provide information NOT found internally.
        Triggered ONLY when internal pipelines fail to provide specific data.
        
        Args:
            query: The user's original query or specific question.
            missing_info_type: What is missing? (e.g., "salary", "niche_course", "trend")
        """
        print(f"    [PIPE] Running ClaudeFallbackPipeline for '{missing_info_type}'...")
        time.sleep(1.0)
        
        # In a real implementation, this would call the Google Search grounding tool.
        # Here we simulate it using the LLM's internal knowledge as "External Web Knowledge".
        
        search_prompt = f"""
You are acting as a Web Search fallback for an EdTech agent.
The internal database did not have information about: {missing_info_type}.
Based on your general knowledge (simulating a web search), answer the following query.

QUERY: {query}

Provide a concise, factual answer.
DO NOT suggest any courses, books, or online learning resources here. Focus strictly on definitions, salary data, or market trends.

Respond in JSON:
{{
    "found": true/false,
    "summary": "Summary of findings...",
    "external_resources": [
        {{
            "name": "Resource Name",
            "url": "URL or description",
            "type": "course/article/report",
            "origin": "external"
        }}
    ],
    "confidence": 0.8
}}
"""
        try:
            # We need access to the model. Since this is a static method in DummyPipelines,
            # we'll use the global COURSE_STORE.model reference if available, or fail gracefully.
            if hasattr(COURSE_STORE, 'model') and COURSE_STORE.model:
                response = COURSE_STORE.model.generate_content(search_prompt)
                
                # Parse JSON
                import json
                text = response.text.strip()
                if text.startswith("```json"): text = text[7:-3]
                elif text.startswith("```"): text = text[3:-3]
                
                try:
                    result = json.loads(text.strip())
                except:
                    # Fallback parsing
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start != -1 and end != -1:
                        result = json.loads(text[start:end])
                    else:
                        raise ValueError("Could not parse JSON")
                        
                return result
            else:
                return {
                    "found": False,
                    "summary": "Search unavailable (Model not connected).",
                    "external_resources": [],
                    "confidence": 0.0
                }
                
        except Exception as e:
            print(f"    [WARN] Claude Fallback Search failed: {e}")
            return {
                "found": False,
                "summary": "Search failed due to error.",
                "external_resources": [],
                "confidence": 0.0
            }

    # ------------------------------------------------------------------ #
    # 5. INTENT PIPELINE
    # ------------------------------------------------------------------ #

    @staticmethod
    def intent_detector(
        user_activities: List[Dict], recent_query: str
    ) -> Dict:
        """
        Simulates an intent pipeline based on user activities and patterns.
        """
        print("    [PIPE] Running intent_detector...")
        time.sleep(0.5)

        texts = " ".join(
            [a.get("page", "") + " " + a.get("action", "") for a in user_activities]
        ).lower() + " " + recent_query.lower()

        if "job" in texts or "switch" in texts or "career" in texts:
            primary = "career_transition"
            secondary = ["upskilling", "job_search"]
        elif "beginner" in texts or "start" in texts:
            primary = "onboarding"
            secondary = ["exploration"]
        elif "devops" in texts or "kubernetes" in texts or "cloud" in texts:
            primary = "cloud_devops_upskilling"
            secondary = ["certification", "career_transition"]
        else:
            primary = "upskilling"
            secondary = ["exploration"]

        raw = {
            "primary_intent": primary,
            "secondary_intents": secondary,
            "signals_used": [
                "recent_query",
                "page_views",
                "course_clicks",
                "search_terms",
            ],
            "time_window": "last_30_days",
            "confidence": 0.82,
        }

        validated = IntentAnalysis(**raw)
        return validated.dict()

    # ------------------------------------------------------------------ #
    # 5. CONTEXT PRESERVATION PIPELINE
    # ------------------------------------------------------------------ #

    @staticmethod
    def context_preserver(
        user_id: str,
        session_id: str,
        new_event: Dict,
        previous_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Context preservation pipeline.
        Accumulates queries, skills, and last recommended courses across turns.
        previous_context is sourced from the last run's context_preserver output,
        ensuring information is properly carried over across subsequent queries.
        """
        print("    [PIPE] Running context_preserver...")

        prev = previous_context or {}
        recent_queries = list(prev.get("recent_queries", []))
        recent_skills = list(prev.get("recent_skills", []))
        last_courses = list(prev.get("last_recommended_courses", []))
        existing_notes = prev.get("notes", "")

        if "query" in new_event:
            # Keep last 10 queries, dedup while preserving order
            q = new_event["query"]
            if q not in recent_queries:
                recent_queries.append(q)
            recent_queries = recent_queries[-10:]

        if "skills" in new_event:
            # Merge skills deduped, preserve existing ones
            incoming = new_event.get("skills", [])
            merged = list(dict.fromkeys(recent_skills + incoming))  # dedup, order-preserving
            recent_skills = merged

        if "courses" in new_event:
            incoming_courses = new_event.get("courses", [])
            # Merge course list keeping last 20, newest at end
            merged_courses = list(dict.fromkeys(last_courses + incoming_courses))
            last_courses = merged_courses[-20:]

        notes = existing_notes
        if new_event.get("query"):
            notes = f"Last query: {new_event['query'][:120]}"

        raw = {
            "user_id": user_id,
            "session_id": session_id,
            "recent_queries": recent_queries,
            "recent_skills": recent_skills,
            "last_recommended_courses": last_courses,
            "timestamp": datetime.now().isoformat(),
            "notes": notes,
            "confidence": 0.9,
        }

        validated = ContextState(**raw)
        return validated.dict()

    # ------------------------------------------------------------------ #
    # 6. MULTI-PIPELINE ORCHESTRATOR
    # ------------------------------------------------------------------ #

    @staticmethod
    def multi_pipeline_orchestrator(query: str) -> Dict:
        """
        Simulates a multi-pipeline orchestrator that recommends which pipelines
        to run for a given query. In real life this might be its own LLM call.
        """
        print("    [PIPE] Running multi_pipeline_orchestrator...")
        time.sleep(0.4)

        q = query.lower()
        steps: List[OrchestrationStep] = []

        if "become" in q or "switch" in q or "transition" in q:
            steps.append(
                OrchestrationStep(
                    pipeline="skill_gap_analyzer",
                    reason="Understand skill gaps for career transition.",
                    parameters={"user_profile": {"current_skills": []}, "target_role": "UNKNOWN"},
                )
            )
            steps.append(
                OrchestrationStep(
                    pipeline="course_recommender",
                    reason="Recommend courses to close gaps.",
                    parameters={"skill_gaps": []},
                )
            )
        elif "course" in q or "learn" in q:
            steps.append(
                OrchestrationStep(
                    pipeline="skills_fetcher",
                    reason="Infer key skills from role or courses mentioned.",
                    parameters={},
                )
            )
            steps.append(
                OrchestrationStep(
                    pipeline="course_recommender",
                    reason="Map skills to concrete course recommendations.",
                    parameters={"skill_gaps": []},
                )
            )
        elif "job market" in q or "salary" in q:
            steps.append(
                OrchestrationStep(
                    pipeline="job_market_analyzer",
                    reason="Analyze market and salary trends for the role.",
                    parameters={"role": "UNKNOWN"},
                )
            )
            steps.append(
                OrchestrationStep(
                    pipeline="claude_fallback",
                    reason="Fallback search using model knowledge for missing info (salary, trends, niche topics).",
                    parameters={"query": q, "missing_info_type": "job_market_data"},
                )
            )
        else:
            steps.append(
                OrchestrationStep(
                    pipeline="intent_detector",
                    reason="Clarify user's high-level intent.",
                    parameters={"user_activities": [], "recent_query": query},
                )
            )

        raw = {
            "query": query,
            "suggested_pipelines": [s.dict() for s in steps],
            "expected_coverage": (
                "Plan covers skill analysis, learning path, market insights, and/or "
                "intent clarification depending on the query."
            ),
            "confidence": 0.76,
        }

        validated = OrchestrationPlan(**raw)
        return validated.dict()

    # ------------------------------------------------------------------ #
    # 7. CAREER & JOB MARKET ANALYZER
    # ------------------------------------------------------------------ #

    @staticmethod
    def career_path_analyzer(current_role: str, target_role: str) -> Dict:
        """
        Simulates career path analysis
        """
        print("    [PIPE] Running career_path_analyzer...")
        time.sleep(0.5)

        raw = {
            "transition_difficulty": "Medium",
            "typical_timeline": "12-18 months",
            "intermediate_roles": ["Junior ML Engineer", "ML Engineer"],
            "success_factors": [
                "Strong portfolio projects",
                "Relevant certifications",
                "Open source contributions",
            ],
            "salary_progression": {
                "current": "$70k",
                "intermediate": "$90k",
                "target": "$120k",
            },
            "confidence": 0.75,
        }

        validated = CareerPathAnalysis(**raw)
        return validated.dict()

    @staticmethod
    def job_market_analyzer(role: str, location: str = "Remote") -> Dict:
        """
        Simulates job market analysis
        """
        print("    [PIPE] Running job_market_analyzer...")
        time.sleep(0.5)

        raw = {
            "total_openings": 1247,
            "demand_trend": "High",
            "top_companies": ["Google", "Microsoft", "Meta", "Amazon"],
            "average_salary": "$125k",
            "required_experience": "3-5 years",
            "top_skills": ["Python", "TensorFlow", "PyTorch", "MLOps"],
            "confidence": 0.90,
        }

        validated = JobMarketAnalysis(**raw)
        return validated.dict()

    @staticmethod
    def summary_retriever(session_id: str = "default_session", user_id: str = "default_user") -> Dict:
        """
        Retrieves the conversation summary and history directly from memory.
        """
        print("    [PIPE] Running summary_retriever...")
        time.sleep(0.5)
        
        history_turns = ""
        rolling_summary = ""
        
        if VIRAAI_MEMORY_SERVICE:
            mem_state = VIRAAI_MEMORY_SERVICE._get_state(session_id, user_id)
            if mem_state:
                history_turns = "\n".join([f"{t.role.capitalize()}: {t.content}" for t in mem_state.recent_turns])
                rolling_summary = mem_state.rolling_summary
        
        return {
            "conversation_history": history_turns if history_turns else "No history available.",
            "conversation_summary": rolling_summary if rolling_summary else "No summary available.",
            "confidence": 0.95
        }


# ============================================================================
# PIPELINE REGISTRY (Maps user queries to pipelines)
# ============================================================================


class PipelineRegistry:
    """
    Registry of available pipelines with semantic descriptions.
    The ReAct loop uses these descriptions to select appropriate pipelines.

    Each pipeline has:
      - function
      - description (for LLM planner)
      - required_params
      - optional_params
      - output_type
    """

    def __init__(self):
        self.pipelines = {
            "skill_gap_analyzer": {
                "function": DummyPipelines.skill_gap_analyzer,
                "description": (
                    "Analyzes skill gaps between a learner's current skills and "
                    "target role requirements. "
                    "Use when: user wants to know what skills they're missing, "
                    "planning a career transition, or assessing readiness for a role. "
                    "Returns: detailed skill gap analysis with proficiency levels."
                ),
                "required_params": ["user_profile", "target_role"],
                "optional_params": [],
                "output_type": "skill_analysis",
            },
            "course_recommender": {
                "function": DummyPipelines.course_recommender,
                "description": (
                    "Recommends relevant courses to fill skill gaps using the "
                    "platform's course catalog (backed by MXBAI embedded courses). "
                    "Use when: user needs learning resources, wants course suggestions, "
                    "or is planning an upskilling roadmap. "
                    "Returns: curated list of courses with details and learning path."
                ),
                "required_params": ["skill_gaps"],
                "optional_params": ["learning_style", "user_query"],
                "output_type": "course_recommendations",
            },
            "skills_fetcher": {
                "function": DummyPipelines.skills_fetcher,
                "description": (
                    "Fetches or infers skills from user profile, job roles, or selected "
                    "courses. Use when: you need a normalized list of skills for a user "
                    "or role before recommending content."
                ),
                "required_params": [],
                "optional_params": ["user_profile", "role", "course_ids"],
                "output_type": "skills_list",
            },
            "intent_detector": {
                "function": DummyPipelines.intent_detector,
                "description": (
                    "Analyzes user activity patterns and recent queries to infer intent "
                    "(e.g., career_transition, upskilling, cloud_devops_upskilling). "
                    "Use when: you want to personalize next steps based on behavior."
                ),
                "required_params": ["user_activities", "recent_query"],
                "optional_params": [],
                "output_type": "intent",
            },
            "context_preserver": {
                "function": DummyPipelines.context_preserver,
                "description": (
                    "Maintains and updates a user's learning session context (recent "
                    "queries, skills, and courses). Use when: you want to persist "
                    "state across multiple interactions."
                ),
                "required_params": ["user_id", "session_id", "new_event"],
                "optional_params": ["previous_context"],
                "output_type": "context_state",
            },
            "multi_pipeline_orchestrator": {
                "function": DummyPipelines.multi_pipeline_orchestrator,
                "description": (
                    "Given a free-form query, suggests an ordered set of pipelines to "
                    "run (skill analysis, course recommendations, intent detection, "
                    "market analysis, etc.). Use when: you want a planning hint for "
                    "multi-pipeline flows."
                ),
                "required_params": ["query"],
                "optional_params": [],
                "output_type": "orchestration_plan",
            },
            "career_path_analyzer": {
                "function": DummyPipelines.career_path_analyzer,
                "description": (
                    "Analyzes career progression paths and transition strategies for "
                    "moving from current role to target role. "
                    "Use when: user is planning a career transition, wants to "
                    "understand progression, or needs a timeline estimate. "
                    "Returns: detailed career path with milestones."
                ),
                "required_params": ["current_role", "target_role"],
                "optional_params": [],
                "output_type": "career_path",
            },
            "job_market_analyzer": {
                "function": DummyPipelines.job_market_analyzer,
                "description": (
                    "Analyzes job market conditions, demand, and salary trends for a role. "
                    "Use when: user wants market insights, salary information, "
                    "or demand trends. Returns: comprehensive market analysis "
                    "with statistics."
                ),
                "required_params": ["role"],
                "optional_params": ["location"],
                "output_type": "market_analysis",
            },
            "claude_fallback": {
                "function": DummyPipelines.claude_fallback_pipeline,
                "description": (
                    "Fallback search for missing information using model knowledge."
                ),
                "required_params": ["query", "missing_info_type"],
                "optional_params": [],
                "output_type": "json",
            },
            "summary_retriever": {
                "function": DummyPipelines.summary_retriever,
                "description": (
                    "Retrieves the conversation history and a rolling summary of all past interactions. "
                    "Use when: user asks to summarize, recap, or review the conversation; "
                    "asks what was discussed, what we talked about, what we covered, or what happened so far; "
                    "asks for a summary or partial summary of any previous exchange; "
                    "asks 'what do you know about me' based on conversation context; "
                    "or references previous interactions in any way. "
                    "This pipeline MUST be called for ANY query involving conversation recall or history review."
                ),
                "required_params": [],
                "optional_params": ["session_id", "user_id"],
                "output_type": "conversation_summary",
            },
        }

    def get_pipeline(self, name: str) -> Optional[Dict[str, Any]]:
        return self.pipelines.get(name)

    def get_all_descriptions(self) -> str:
        desc = "Available Pipelines:\n\n"
        for name, info in self.pipelines.items():
            desc += f"â€¢ {name}: {info['description']}\n"
        return desc


# ============================================================================
# REACT AGENT - THE CORE REASONING ENGINE
# ============================================================================


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent for EdTech Deep Thinking
    """

    def __init__(self, model: UnifiedModel, pipeline_registry: PipelineRegistry):
        self.model = model
        self.registry = pipeline_registry
        self.iteration_count = 0
        self.learning_memory = []  # Stores lessons from past failures

    def _retry_with_backoff(self, func, *args, max_retries=None, initial_delay=None, timeout=None, **kwargs):
        """
        Helper to retry LLM calls with exponential backoff on ResourceExhausted errors.
        ENHANCED: Now includes rate limiting, timeout handling, and metrics collection.
        """
        max_retries = max_retries or ViraAIConfig.MAX_RETRIES
        delay = initial_delay or ViraAIConfig.INITIAL_RETRY_DELAY
        timeout = timeout or ViraAIConfig.LLM_CALL_TIMEOUT
        
        for i in range(max_retries):
            try:
                # Rate limiting check before making request
                RATE_LIMITER.wait_if_needed()
                
                # Execute with timing for metrics
                start_time = time.time()
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                # Record successful call
                METRICS_COLLECTOR.record_llm_call(latency)
                
                return result
                
            except Exception as e:
                # Handle error with detailed tracking
                error_info = ERROR_HANDLER.handle_error(e, context=f"{func.__name__} (attempt {i+1}/{max_retries})")
                
                # Check if error is retryable
                is_retryable = error_info["is_retryable"]
                
                if not is_retryable or i == max_retries - 1:
                    # Non-retryable error or final attempt - raise it
                    if ViraAIConfig.VERBOSE_ERRORS:
                        print(f"   [ERROR] {error_info['type']}: {error_info['message']}")
                    raise e
                
                # Retryable error - log and retry
                ERROR_HANDLER.total_retries += 1
                
                # Calculate backoff with jitter
                import random
                jitter = random.uniform(0.8, 1.2)  # ±20% jitter
                backoff_time = delay * jitter
                
                print(f"   [WARN] {error_info['type']} (attempt {i+1}/{max_retries}). Retrying in {backoff_time:.1f}s...")
                time.sleep(backoff_time)
                
                # Exponential backoff
                delay *= ViraAIConfig.BACKOFF_FACTOR


    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------ #

    def _classify_query(self, query: str) -> str:
        """
        Classifies user query to determine how to handle it.
        
        Returns:
            "knowledge" - simple factual question, use search only
            "pipeline" - needs pipeline processing
            "hybrid" - needs both pipelines and search
        """
        try:
            classification_prompt = f"""Classify this user query into ONE category:

CATEGORIES:
- "knowledge": Simple factual/informational question (What is X? Define Y? Tell me about Z?)
- "pipeline": Needs career/course/skills analysis (career paths, course recommendations, skill gaps),
             OR needs conversation recall/summary (summarize, what did we discuss, recap, review our chat)
- "hybrid": Needs both factual info AND career analysis

IMPORTANT: If the user is asking to recall, summarize, or review a previous conversation or interaction,
that is ALWAYS "pipeline" — NOT "knowledge". Conversation recall requires the summary_retriever pipeline.

QUERY: {query}

Respond with ONLY ONE WORD: knowledge, pipeline, or hybrid
"""
            response = self._retry_with_backoff(self.model.generate_content, classification_prompt)
            classification = response.text.strip().lower()
            
            #Validate response
            if classification in ["knowledge", "pipeline", "hybrid"]:
                print(f"[CLASSIFY] Query classified as: {classification}")
                return classification
            else:
                # Default to hybrid if unclear
                print("[CLASSIFY] Classification unclear, defaulting to hybrid")
                return "hybrid"
                
        except Exception as e:
            print(f"[WARN] Query classification failed: {e}. Defaulting to pipeline")
            return "pipeline"

    def _parse_intent(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        LLM-based intent parser that serves as the single decision-maker for:
          - Role extraction (query roles always override profile)
          - Continuation / filter / follow-up detection
          - Response scope (how much detail to give)
          - Context switching (when to use profile vs conversation context)
        """
        history = context.get("conversation_history", [])
        history_text = ""
        if history:
            history_text = "PREVIOUS CONVERSATION:\n" + "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:]]
            )

        # Surface previous answer + courses so the LLM can detect filter/continuation
        last_answer_summary = context.get("_last_answer_summary", "")
        last_courses_json = ""
        last_pipeline = context.get("_last_pipeline_results", {})
        if last_pipeline:
            cr = last_pipeline.get("course_recommender", {})
            courses = cr.get("recommended_courses", [])
            if courses:
                last_courses_json = json.dumps(courses[:10], indent=2)

        last_context_section = ""
        if last_answer_summary or last_courses_json:
            last_context_section = f"""
LAST ASSISTANT RESPONSE (summary, max 400 chars):
{last_answer_summary}

LAST RECOMMENDED COURSES (from previous turn):
{last_courses_json if last_courses_json else 'None'}
"""

        prompt = f"""
You are an expert intent parser for an EdTech platform.
Analyze the USER QUERY carefully. The query might reference the previous conversation.

IMPORTANT ROLE EXTRACTION RULES:
- If the query explicitly mentions roles (e.g. "from Data Analyst to Data Architect"),
  extract those as current_role and target_role, IGNORING the user profile below.
- "from X to Y" means current_role=X, target_role=Y.
- If the query does NOT mention any role, return null for current_role and target_role
  (the system will fall back to the user profile).
- If the query references a role from the previous conversation ("that role", "the one
  you mentioned"), resolve it using the conversation history.

USER PROFILE (IMMUTABLE SNAPSHOT - query roles always override these):
{json.dumps(context.get('user_profile_snapshot', dict()), indent=2)}

{history_text}
{last_context_section}

{f"VIRAAI MEMORY LAYER:\nTreat this memory as factual, user-provided context. Do NOT question or override it unless explicitly contradicted by the user.\n{context.get('_viraai_memory', '')}\n" if context.get('_viraai_memory') else ""}
{f"### Conversation History (Retrieved Memory)\nUse this conversation history to answer. Do NOT assume this is the first interaction.\n{context.get('_viraai_full_history', '')}\n" if context.get('_viraai_full_history') else ""}
USER QUERY:
"{query}"

Analyze the query and return a JSON object:
{{
    "target_role": "Role user wants to achieve, extracted from QUERY (null if not mentioned in query)",
    "current_role": "Starting role, extracted from QUERY (null if not stated in query)",
    "constraints": ["list of constraints e.g. 6 months, free, budget"],
    "requested_info": ["skills", "courses", "salary", "market_demand", "learning_path"],
    "intent_type": "career_transition | upskilling | exploration | specific_query",
    "resolved_query": "Fully contextualized query with pronouns resolved",
    "query_type": "new_topic | continuation | filter_refine | follow_up",
    "response_scope": "focused | moderate | comprehensive",
    "use_previous_results": true or false,
    "filter_instruction": "Describe what to filter/narrow/compare from previous results, or null if not applicable"
}}

DEFINITIONS for query_type:
- "new_topic": A completely new question unrelated to previous conversation.
- "continuation": Continues the same topic (e.g. "what about its salary?", "tell me more").
- "filter_refine": Asks to narrow, filter, compare, or select from previously given results (e.g. "show only the top 2", "remove Python courses", "compare those").
- "follow_up": A related but different angle on the same topic (e.g. asking about job market after asking about courses).

DEFINITIONS for response_scope:
- "focused": User asked a specific question — answer ONLY that (e.g. "what is the salary?").
- "moderate": User wants some detail but not exhaustive (e.g. "what courses should I take?").
- "comprehensive": User explicitly wants full detail (e.g. "give me a complete roadmap").

DEFINITIONS for use_previous_results:
- true: The query references, filters, or continues from the last response's data.
- false: The query is about something new and needs fresh pipeline runs.
"""
        try:
            response = self._retry_with_backoff(self.model.generate_content, prompt)
            parsed = self._parse_json_response(response.text)
            if parsed:
                print(f"   [INTENT] query_type={parsed.get('query_type', '?')}, "
                      f"scope={parsed.get('response_scope', '?')}, "
                      f"use_prev={parsed.get('use_previous_results', '?')}")
                if parsed.get('filter_instruction'):
                    print(f"   [INTENT] filter_instruction: {parsed['filter_instruction']}")
                return parsed
            return {}
        except Exception as e:
            print(f"   [WARN] Intent parsing failed: {e}. Falling back to defaults.")
            return {}

    def process_query(self, user_query: str, user_context: Dict) -> Dict:
        # ---- TASK 5: Global execution timer ----
        _global_start_time = time.time()

        print("\n" + "=" * 80)
        print("[BOT] ViraAI EdTech ReAct Agent - Deep Thinking Mode")
        print("=" * 80)
        print(f"Query: {user_query}")
        print("=" * 80 + "\n")

        # ------------------------------------------------------------------ #
        # VIRAAI MEMORY LAYER INJECTION
        # ------------------------------------------------------------------ #
        session_id = user_context.get("session_id", "default_session")
        user_id = user_context.get("user_id", "default_user")
        
        if VIRAAI_MEMORY_SERVICE:
            print("[MEMORY] Retrieving compiled episodic memory...")
            try:
                compiled_memory = VIRAAI_MEMORY_SERVICE.get_compiled_context(session_id, user_id, current_query=user_query)
                user_context["_viraai_memory"] = compiled_memory
                # ---- TASK 2: Log FULL memory contents (never truncated) ----
                print("[MEMORY] === FULL COMPILED MEMORY START ===")
                print(compiled_memory)
                print("[MEMORY] === FULL COMPILED MEMORY END ===")
                
                # Always inject full conversation history when memory has turns
                if VIRAAI_MEMORY_SERVICE:
                    mem_state = VIRAAI_MEMORY_SERVICE._get_state(session_id, user_id)
                    if mem_state and mem_state.recent_turns:
                        history_turns = "\n".join([f"{t.role.capitalize()}: {t.content}" for t in mem_state.recent_turns])
                        user_context["_viraai_full_history"] = history_turns
                        print(f"[MEMORY] Injected {len(mem_state.recent_turns)} conversation turns into context.")
            except Exception as mem_e:
                print(f"[WARN] Failed to retrieve memory {mem_e}")

        # 0. Intent Parsing
        # Explicitly parse intent from query to override profile defaults if needed
        print("[INTENT] Parsing user query for explicit intent...")
        parsed_intent = self._parse_intent(user_query, user_context)

        enhanced_context = user_context.copy()
        enhanced_context["parsed_intent"] = parsed_intent

        # ------------------------------------------------------------------ #
        # ROLE OVERRIDE: query-stated roles take precedence over profile
        # ------------------------------------------------------------------ #
        # Store originals so downstream can reference profile if needed
        enhanced_context["_profile_current_role"] = enhanced_context.get("current_role")
        enhanced_context["_profile_target_role"] = enhanced_context.get("target_role")

        if parsed_intent.get("target_role"):
            enhanced_context["target_role"] = parsed_intent["target_role"]
            print(f"   [ROLE] target_role: {enhanced_context['_profile_target_role']} → {parsed_intent['target_role']}")

        if parsed_intent.get("current_role"):
            enhanced_context["current_role"] = parsed_intent["current_role"]
            print(f"   [ROLE] current_role: {enhanced_context['_profile_current_role']} → {parsed_intent['current_role']}")

        # ------------------------------------------------------------------ #
        # CONTEXT CONTINUITY: restore persisted context from previous turn
        # ------------------------------------------------------------------ #
        if "previous_context" not in enhanced_context:
            prev_ctx = user_context.get("_persisted_context")
            if prev_ctx:
                enhanced_context["previous_context"] = prev_ctx

        # ------------------------------------------------------------------ #
        # CACHE INJECTION: for filter/refine queries, inject previous results
        # ------------------------------------------------------------------ #
        # The LLM decided whether this query should operate on previous data
        _use_prev = parsed_intent.get("use_previous_results", False)
        _query_type = parsed_intent.get("query_type", "new_topic")
        _filter_instruction = parsed_intent.get("filter_instruction")
        _response_scope = parsed_intent.get("response_scope", "moderate")

        # Store scope + type in context for _synthesize_answer to read
        enhanced_context["_response_scope"] = _response_scope
        enhanced_context["_query_type"] = _query_type
        enhanced_context["_filter_instruction"] = _filter_instruction

        # Pre-seed pipeline_results with cached data when LLM says to use previous
        _cached_pipeline_seed: Dict[str, Any] = {}
        if _use_prev and _query_type in ("filter_refine", "continuation", "follow_up"):
            prev_pipeline = user_context.get("_last_pipeline_results", {})
            if prev_pipeline:
                _cached_pipeline_seed = dict(prev_pipeline)
                print(f"[CACHE] Injecting previous pipeline results for {_query_type} query.")

        # Classify query type and optionally run universal search
        query_type = self._classify_query(user_query)

        search_results = None
        if query_type in ["knowledge", "hybrid"]:
            print(f"[SEARCH] Performing universal search...")
            if UNIVERSAL_SEARCH:
                search_results = UNIVERSAL_SEARCH.search(user_query, enhanced_context)
                print(f"   [OK] Search complete (confidence: {search_results.get('confidence', 0):.2f})")
            else:
                print("   [WARN] Universal search not initialized")

        # EdTech-related 'knowledge' queries are promoted to 'hybrid' so pipelines always run.
        _is_edtech_knowledge = any(
            kw in user_query.lower() for kw in [
                "course", "skill", "learn", "career", "job", "salary", "role",
                "path", "transition", "recommend", "market", "gap",
                "discuss", "discussed", "discussion", "summary", "summarise", "summarize",
                "conversation", "recap", "review", "talked", "chatted", "covered",
                "went over", "brief", "gist",
                "my profile", "my role", "my skills", "tell me about me"
            ]
        )
        if query_type == "knowledge" and search_results and search_results.get("found") and not _is_edtech_knowledge:
            print("[KNOWLEDGE] Non-EdTech knowledge query. Returning search results directly.")
            return {
                "answer": search_results["summary"],
                "quality_score": search_results.get("confidence", 0.8),
                "complexity": "simple",
                "iterations": 0,
                "pipelines_used": [],
                "reasoning_trace": [],
                "validation_history": [],
                "success": True,
                "search_results": search_results,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": ViraAIConfig.MODEL_NAME,
                    "query_type": "knowledge"
                },
                "final_context": enhanced_context
            }
        if query_type == "knowledge" and _is_edtech_knowledge:
            print("[KNOWLEDGE] EdTech query detected. Promoting to hybrid for pipeline execution.")
            query_type = "hybrid"

        state: Dict[str, Any] = {
            "query": user_query,
            "context": enhanced_context,
            "pipeline_results": dict(_cached_pipeline_seed),
            "search_results": search_results,
            "current_answer": None,
            "quality_score": 0.0,
            "iterations": 0,
            "reasoning_trace": [],
            "validation_history": [],
            "pipeline_errors": [],
        }

        COURSE_STORE.clear_cache()
        state["complexity"] = self._assess_complexity(state["query"])

        try:
            max_iters = ViraAIConfig.MAX_ITERATIONS
            if state["complexity"] == "simple":
                max_iters = 2
                print(f"[CONFIG] Simple query detected. Capping iterations at {max_iters}.")

            while (
                state["quality_score"] < ViraAIConfig.QUALITY_THRESHOLD
                and state["iterations"] < max_iters
            ):
                self.iteration_count = state["iterations"] + 1
                print(f"\n--------------------------------------------------------------------------------")
                print(f"[ITER] ITERATION {self.iteration_count}")
                print(f"--------------------------------------------------------------------------------")

                # Explore alternative paths on first iteration or when quality is very low
                should_explore = (self.iteration_count == 1 or state["quality_score"] < 0.4)
                if state["complexity"] == "simple" and state["quality_score"] > 0.1:
                    should_explore = False

                if should_explore:
                    exploration_result = self.explore_alternative_paths(state)
                    state["reasoning_trace"].append({"exploration": exploration_result})
                    state["context"]["selected_strategy"] = exploration_result["selected_path"]
                    state["context"]["strategy_reasoning"] = exploration_result["reasoning"]

                # Use self-consistency voting for first 2 iterations on complex queries
                if state["complexity"] == "simple":
                    reasoning = self._reason(state)
                elif self.iteration_count <= 2:
                    reasoning = self._reason_with_voting(state)
                else:
                    reasoning = self._reason(state)
                state["reasoning_trace"].append(reasoning)

                # Execute all planned pipeline actions, with error recovery per action
                actions_taken = []
                for action_plan in reasoning.get("planned_actions", []):
                    try:
                        result = self._execute_action(action_plan, state)
                        actions_taken.append(result)
                    except Exception as e:
                        error_msg = str(e)
                        recovery_strategy = self.recover_from_error(error_msg, state, action_plan)
                        state["reasoning_trace"].append({"recovery": recovery_strategy})
                        actions_taken.append({
                            "pipeline": action_plan.get("pipeline"),
                            "params": action_plan.get("parameters"),
                            "result": {"error": error_msg, "recovery_plan": recovery_strategy["action_plan"]},
                            "success": False
                        })

                self._observe(actions_taken, state)
                for action in actions_taken:
                    if action.get("success"):
                        self._validate_intermediate(action["pipeline"], action["result"])

                reflection = self._reflect(state, actions_taken)
                state["reasoning_trace"].append({"reflection": reflection})

                raw_answer = self._synthesize_answer(state)
                refined_answer = self._critique_and_refine(raw_answer, state)
                state["current_answer"] = refined_answer

                validation = self._validate_answer(state)
                state["quality_score"] = validation.get("quality_score", 0.0)
                state["validation_history"].append(validation)

                if state["quality_score"] >= ViraAIConfig.QUALITY_THRESHOLD:
                    print(
                        f"\n[OK] Quality threshold met! Score: {state['quality_score']:.2%}"
                    )
                    break
                else:
                    print(f"\n[WARN] Quality below threshold: {state['quality_score']:.2%}")
                    issues = validation.get("issues") or []
                    print(f"   Issues: {issues}")
                    print("   -> Planning next iteration to address gaps...")

                state["iterations"] += 1
                time.sleep(3)  # Reduced from 5s — still respects rate limits

        finally:
            COURSE_STORE.clear_cache()

        # Persist context_preserver result so next query can access accumulated history.
        final_ctx = state["context"].copy()
        cp_result = state["pipeline_results"].get("context_preserver")
        if cp_result:
            final_ctx["_persisted_context"] = cp_result
            final_ctx["recent_queries"] = cp_result.get("recent_queries", [])
            final_ctx["recent_skills"] = cp_result.get("recent_skills", [])
            final_ctx["last_recommended_courses"] = cp_result.get("last_recommended_courses", [])

        # ------------------------------------------------------------------ #
        # CACHE: store last answer + pipeline results for follow-up queries
        # ------------------------------------------------------------------ #
        # These are read by _parse_intent on the NEXT query to detect
        # filter/refine intent and provide the LLM with previous results.
        answer_text = state.get("current_answer", "")
        final_ctx["_last_answer_summary"] = (answer_text or "")[:400]
        final_ctx["_last_pipeline_results"] = state.get("pipeline_results", {})
        # Keep the conversation-level roles (query-overridden) for continuity
        final_ctx["_conversation_current_role"] = final_ctx.get("current_role")
        final_ctx["_conversation_target_role"] = final_ctx.get("target_role")

        if VIRAAI_MEMORY_SERVICE:
            print("[MEMORY] Updating episodic memory asynchronously...")
            try:
                VIRAAI_MEMORY_SERVICE.update_memory_after_turn(
                    session_id=final_ctx.get("session_id", "default_session"),
                    user_id=final_ctx.get("user_id", "default_user"),
                    user_msg=user_query,
                    assistant_msg=state.get("current_answer", "")
                )
                print("[MEMORY] Memory updated successfully.")
                # ---- TASK 2: Log full memory state after update ----
                try:
                    post_update_memory = VIRAAI_MEMORY_SERVICE.get_compiled_context(
                        final_ctx.get("session_id", "default_session"),
                        final_ctx.get("user_id", "default_user"),
                        current_query=user_query
                    )
                    print("[MEMORY] === POST-UPDATE MEMORY DUMP START ===")
                    print(post_update_memory)
                    print("[MEMORY] === POST-UPDATE MEMORY DUMP END ===")
                except Exception as dump_e:
                    print(f"[MEMORY][WARN] Could not dump post-update memory: {dump_e}")
            except Exception as e:
                print(f"[WARN] Failed to update memory: {e}")

        response = self._prepare_final_response(state)
        response["final_context"] = final_ctx

        # ---- TASK 4: Enforce course labels in final answer ----
        response["answer"] = self._enforce_course_labels(response.get("answer", ""), state)

        # ---- TASK 5: Append execution timer ----
        _global_elapsed = time.time() - _global_start_time
        response["answer"] = response["answer"].rstrip() + f"\n\nExecution Time: {_global_elapsed:.2f} seconds"
        print(f"[TIMER] Total execution time: {_global_elapsed:.2f} seconds")

        return response

    # ------------------------------------------------------------------ #
    # CLAUDE API CONTEXT ENGINE — PARALLEL ENTRY POINT (NEW)
    # ------------------------------------------------------------------ #



    # ------------------------------------------------------------------ #
    # INTERNAL STEPS
    # ------------------------------------------------------------------ #

    def _assess_complexity(self, query: str) -> str:
        """
        Determines the complexity of the query to adjust validation strictness.
        Returns: 'simple' or 'complex'
        """
        # Quick heuristic for clearly simple queries
        if len(query.split()) < 5:
            return "simple"
            
        prompt = f"""
        Analyze the complexity of this user query for an EdTech agent.
        
        QUERY: {query}
        
        CLASSIFICATION RULES:
        - "simple": Direct requests, definitions, list requests (e.g., "List courses for Python", "What is a PM?").
                    ALSO INCLUDES specific roadmap requests (e.g. "Give me a 12-month roadmap", "Learning path for X") where the task is clear.
        - "complex": Ambiguous goals, "help me figure out my life", queries requiring complex user history analysis or multi-step deducing of user intent.
        
        Respond with STRICT JSON: {{"complexity": "simple" | "complex", "reason": "why"}}
        """
        try:
            response = self.model.generate_content(prompt)
            raw = self._parse_json_response(response.text)
            complexity = raw.get("complexity", "complex")
            print(f"[PLAN] Query classified as: {complexity.upper()} ({raw.get('reason', 'no reason')})")
            return complexity
        except:
            print("[WARN] Complexity assessment failed, defaulting to 'complex'")
            return "complex"


    def _reason(self, state: Dict) -> Dict:
        print("[REASONING] Analyzing situation and planning next steps...")

        available_pipelines = self.registry.get_all_descriptions()
        last_validation = (
            state["validation_history"][-1] if state["validation_history"] else {}
        )
        pipeline_errors = state.get("pipeline_errors", [])
        course_meta = state["pipeline_results"].get("course_recommender", {})
        
        # Extract parsed intent for the prompt
        parsed_intent = state["context"].get("parsed_intent", {})

        # [CONTEXT] Adaptive Context Injection
        # For simple queries, do not force deep user profile context unless relevant
        complexity = state.get("complexity", "complex")
        is_simple = complexity == "simple"
        
        user_profile_context = state['context']
        conversation_history = state['context'].get("conversation_history", [])
        history_context = ""
        
        if conversation_history:
            # Include last 2 turns for context maintenance
            recent_history = conversation_history[-4:]
            history_context = "RECENT CONVERSATION:\n" + "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]
            )

        if is_simple:
            # Include essential profile fields so the planner always knows user context
            # even for simple profile-retrieval queries like "tell me my role"
            user_profile_context = {
                "user_id": state['context'].get('user_id'),
                "current_role": state['context'].get('current_role'),
                "target_role": state['context'].get('target_role'),
                "current_skills": state['context'].get('current_skills'),
                "experience_level": state['context'].get('experience_level'),
                "user_profile_snapshot": state['context'].get('user_profile_snapshot', {})
            }

        reasoning_prompt = f"""
You are ViraAI's EdTech reasoning engine. You help learners on an education platform.
You are in "deep thinking" mode.

**UNIFIED REASONING FLOW:**
1. **Interpret**: Analyze the user's explicit intent (see PARSED INTENT below).
2. **Plan**: Determine which pipelines are needed. Ensure you cover:
   - **Role -> Skills**: Use `skill_gap_analyzer` or `skills_fetcher` to find required skills.
   - **Course -> Skills**: Use `skills_fetcher` to validate what skills courses teach.
   - **Skills -> Courses**: Use `course_recommender` to find courses for skill gaps.
   - **Market/Context**: Use `job_market_analyzer` or `career_path_analyzer` if requested.
   - **Fallback**: Use `claude_fallback` ONLY if internal pipelines cannot answer specific questions (e.g., niche salary data, external trends).
3. **Execute**: Call the pipelines in the correct dependency order.
4. **Integrate**: Ensure you have all data to answer: Skills, Gaps, Courses, Timeline, Salary (if asked).

**IMPORTANT RULES:**
- **QUERY ROLES OVERRIDE PROFILE**: If PARSED INTENT has target_role or current_role extracted from the query, those are the authoritative roles. USER PROFILE roles are fallbacks only.
- **Prioritize PARSED INTENT**: If the parsed intent specifies a target role (e.g., "Data Scientist"), USE IT, even if the user profile says something else.
- **QUERY TYPE AWARENESS**: Check parsed_intent.query_type:
  - "filter_refine": Previous pipeline results are already loaded. You may skip re-running pipelines if the data is already available. Focus on filtering/narrowing.
  - "continuation"/"follow_up": Previous results may be pre-loaded. Run additional pipelines only if needed for the new angle.
  - "new_topic": Run all necessary pipelines from scratch.
- **Do NOT skip steps**: If the user asks for a career path, you MUST run `skill_gap_analyzer` AND `course_recommender` AND `career_path_analyzer`.
- **Check for "Hallucinations"**: If `course_recommender` uses "llm_fallback", be transparent.
- **Ask for Clarification**: If the query is truly ambiguous (e.g., "I want to learn"), call `ask_user`.
- **Pipeline Validation**: Verify that you have selected the best possible pipeline combination. Do not skip any necessary pipeline.
- **CONVERSATION RECALL RULE**: If the user is asking about what was discussed, requesting a summary, gist, recap,
  or review of previous interactions, you MUST explicitly output an action to call the `summary_retriever` pipeline. This applies to ALL queries, regardless of complexity. Do NOT try to answer from your context alone.

**CONTEXT:**
USER QUERY: {state['query']}
PARSED INTENT: {json.dumps(parsed_intent, indent=2)}
ORIGINAL USER PROFILE (IMMUTABLE SNAPSHOT):
{json.dumps(state['context'].get('user_profile_snapshot', dict()), indent=2)}

{history_context}
{f"\nVIRAAI CORE MEMORY (DYNAMIC):\nTreat this memory as factual, user-provided context. Do NOT question or override it unless explicitly contradicted by the user.\nIf the user asks about their profile, skills, or background, rely PRIMARILY on the 'ORIGINAL USER PROFILE (IMMUTABLE SNAPSHOT)' above. Only use this dynamic memory if additional information is requested that is not in the snapshot.\n{state['context'].get('_viraai_memory', '')}\n" if state['context'].get('_viraai_memory') else ""}
{f"\n### Conversation History (Retrieved Memory)\nUse this conversation history to answer. Do NOT assume this is the first interaction.\n{state['context'].get('_viraai_full_history', '')}\n" if state['context'].get('_viraai_full_history') else ""}

**HISTORY:**
PREVIOUS RESULTS: {json.dumps(list(state['pipeline_results'].keys()))}
LAST VALIDATION: {json.dumps(last_validation, indent=2)}
ERRORS: {json.dumps(pipeline_errors, indent=2)}
MEMORY: {json.dumps(self.learning_memory, indent=2)}

**AVAILABLE PIPELINES:**
{available_pipelines}

**TASK:**
Decide the next set of pipelines to run. Return a STRICT JSON object.

JSON FORMAT:
{{
    "reasoning": "Step-by-step logic (Hypothesis -> Evidence -> Plan)...",
    "planned_actions": [
        {{
            "pipeline": "pipeline_name",
            "why": "Reason for calling this pipeline",
            "parameters": {{"param_name": "value"}}
        }}
    ],
    "strategy": "High-level strategy for this iteration"
}}
"""

        response = self._retry_with_backoff(self.model.generate_content, reasoning_prompt)
        raw = self._parse_json_response(response.text, retry_on_fail=True)
        
        # Retry once if parsing completely failed
        if raw is None:
            print("   [RETRY] JSON parse failed. Retrying with enhanced prompt...")
            retry_prompt = reasoning_prompt + "\n\n**CRITICAL**: Respond with ONLY valid JSON. Start with { and end with }. No extra text."
            response = self._retry_with_backoff(self.model.generate_content, retry_prompt)
            raw = self._parse_json_response(response.text, retry_on_fail=False)

        try:
            parsed = ReasoningOutput.parse_obj(raw)
            reasoning_dict = parsed.dict()
        except ValidationError as e:
            print("   [WARN] Reasoning JSON failed validation, using fallback")
            print(f"      Validation error: {e}")
            # Create minimal valid fallback
            reasoning_dict = {
                "reasoning": "LLM validation failed. Using fallback strategy.",
                "strategy": "Default course recommendation",
                "planned_actions": [
                    {
                        "pipeline": "course_recommender",
                        "parameters": {"query": state["query"]},
                        "rationale": "Fallback action"
                    }
                ],
                "confidence": 0.5
            }

        print(
            f"   [PLAN] Planned actions: "
            f"{len(reasoning_dict.get('planned_actions', []))} pipelines"
        )

        return reasoning_dict

    def _reason_with_voting(self, state: Dict) -> Dict:
        """
        Implements Self-Consistency Voting.
        Runs reasoning 3 times and selects the most consistent/comprehensive plan.
        ENHANCED: Now uses parallel execution for faster voting (3x speedup).
        """
        print("[VOTING] Running self-consistency voting (3 passes)...")
        
        # ENHANCED: Try parallel execution first
        if ASYNC_EXECUTOR.enabled and ViraAIConfig.ENABLE_PARALLEL_EXECUTION:
            try:
                print("   [PARALLEL] Executing 3 reasoning passes in parallel...")
                
                # Create tasks for parallel execution
                tasks = [
                    (self._reason, (state,), {})
                    for _ in range(3)
                ]
                
                # Execute in parallel
                start_time = time.time()
                candidates = ASYNC_EXECUTOR.execute_parallel(tasks)
                elapsed = time.time() - start_time
                
                # Record parallel execution
                METRICS_COLLECTOR.record_execution_mode(is_parallel=True)
                
                print(f"   [PARALLEL] Completed 3 passes in {elapsed:.2f}s (parallel)")
                
            except Exception as e:
                print(f"   [WARN] Parallel voting failed: {e}. Falling back to sequential.")
                candidates = []
        else:
            candidates = []
        
        # Fallback to sequential if parallel failed or disabled
        if not candidates:
            METRICS_COLLECTOR.record_execution_mode(is_parallel=False)
            for i in range(3):
                print(f"   [VOTE] Generating reasoning path {i+1}...")
                try:
                    candidate = self._reason(state)
                    candidates.append(candidate)
                except Exception as e:
                    print(f"   [WARN] Reasoning pass {i+1} failed: {e}")
        
        if not candidates:
            # Fallback if all fail
            return {"reasoning": "Voting failed, using default fallback.", "planned_actions": [], "strategy": "Fallback"}
        
        if len(candidates) == 1:
            return candidates[0]
            
        # Selection Logic:
        # 1. Prefer plans with actions over empty plans
        # 2. Prefer plans that don't repeat recent errors (checked in _reason implicitly via prompt)
        # 3. Simple consensus: pick the one with the most actions (heuristic for comprehensiveness)
        #    or use LLM to select the best one (more robust but expensive)
        
        # We'll use a lightweight LLM selection step
        selection_prompt = f"""
You are a judge selecting the best reasoning plan from 3 candidates.
Choose the plan that is most logical, comprehensive, and likely to solve the user's query.

QUERY: {state['query']}

CANDIDATE 1:
{json.dumps(candidates[0], indent=2)}

CANDIDATE 2:
{json.dumps(candidates[1] if len(candidates) > 1 else {}, indent=2)}

CANDIDATE 3:
{json.dumps(candidates[2] if len(candidates) > 2 else {}, indent=2)}

Respond with the index of the best candidate (0, 1, or 2) and a brief reason.
JSON format: {{"best_index": 0, "reason": "..."}}
"""
        try:
            response = self._retry_with_backoff(self.model.generate_content, selection_prompt)
            raw = self._parse_json_response(response.text)
            best_index = int(raw.get("best_index", 0))
            if best_index < 0 or best_index >= len(candidates):
                best_index = 0
            print(f"   [VOTE] Selected Candidate {best_index+1}: {raw.get('reason', 'No reason provided')}")
            return candidates[best_index]
        except Exception as e:
            print(f"   [WARN] Voting selection failed: {e}. Defaulting to Candidate 1.")
            return candidates[0]

    def _execute_actions_parallel(self, actions: List[Dict], state: Dict) -> List[Dict]:
        """
        Execute multiple pipeline actions in parallel when they have no dependencies.
        ENHANCED: New method for parallel pipeline execution.
        
        Args:
            actions: List of action dictionaries with pipeline and parameters
            state: Current agent state
            
        Returns:
            List of execution results in the same order as input actions
        """
        if not ASYNC_EXECUTOR.enabled or len(actions) <= 1:
            # Fall back to sequential execution
            return [self._execute_action(action, state) for action in actions]
        
        # Detect dependencies - if any action depends on results from another, execute sequentially
        # For now, we use a simple heuristic: if any action is skill_gap_analyzer or skills_fetcher,
        # and another is course_recommender, they have dependencies
        has_dependencies = False
        pipeline_names = [a.get("pipeline") for a in actions]
        
        if "course_recommender" in pipeline_names:
            if "skill_gap_analyzer" in pipeline_names or "skills_fetcher" in pipeline_names:
                has_dependencies = True
        
        if has_dependencies:
            print("   [INFO] Detected dependencies between actions, executing sequentially.")
            return [self._execute_action(action, state) for action in actions]
        
        # No dependencies - execute in parallel
        print(f"   [PARALLEL] Executing {len(actions)} independent actions in parallel...")
        
        try:
            # Create tasks for parallel execution
            tasks = [
                (self._execute_action, (action, state), {})
                for action in actions
            ]
            
            # Execute in parallel
            start_time = time.time()
            results = ASYNC_EXECUTOR.execute_parallel(tasks)
            elapsed = time.time() - start_time
            
            METRICS_COLLECTOR.record_execution_mode(is_parallel=True)
            print(f"   [PARALLEL] Completed {len(actions)} actions in {elapsed:.2f}s")
            
            return results
            
        except Exception as e:
            print(f"   [WARN] Parallel action execution failed: {e}. Falling back to sequential.")
            METRICS_COLLECTOR.record_execution_mode(is_parallel=False)
            return [self._execute_action(action, state) for action in actions]

    def _execute_action(self, action: Dict, state: Dict) -> Dict:
        """
        Executes a single planned action with robust error handling.
        ENHANCED: Includes metrics tracking, structured error handling, and
        injects user_query into course_recommender parameters so the internal
        catalog search benefits from query expansion and semantic reranking.
        """
        pipeline_name = action.get("pipeline")
        suggested_params = action.get("parameters", {}) or {}

        pipeline_info = self.registry.get_pipeline(pipeline_name)

        if pipeline_info:
            try:
                # [FIX] Inject user_query into course_recommender so the COURSE_STORE
                # hybrid search uses the real question for query expansion & reranking.
                if pipeline_name == "course_recommender" and "user_query" not in suggested_params:
                    suggested_params = dict(suggested_params)
                    suggested_params["user_query"] = state.get("query", "")
                    
                if pipeline_name == "summary_retriever":
                    suggested_params = dict(suggested_params)
                    suggested_params["session_id"] = state["context"].get("session_id", "default_session")
                    suggested_params["user_id"] = state["context"].get("user_id", "default_user")

                # Parameter resolution
                final_params = self._build_pipeline_parameters(
                    pipeline_name, pipeline_info, suggested_params, state
                )

                fn = pipeline_info["function"]
                start_time = time.time()
                result = fn(**final_params)
                latency = time.time() - start_time

                METRICS_COLLECTOR.record_pipeline(pipeline_name, latency)
                state["pipeline_results"][pipeline_name] = result

                # After summary_retriever runs, promote its conversation_history
                # into the state context so the synthesis prompt's dedicated
                # CRITICAL INSTRUCTION block picks it up prominently.
                if pipeline_name == "summary_retriever" and isinstance(result, dict):
                    conv_hist = result.get("conversation_history", "")
                    if conv_hist and conv_hist != "No history available.":
                        state["context"]["_viraai_full_history"] = conv_hist
                        print(f"   [MEMORY] Promoted summary_retriever history into context ({len(conv_hist)} chars)")

                return {
                    "pipeline": pipeline_name,
                    "parameters": final_params,
                    "result": result,
                    "success": True,
                }

            except Exception as e:
                error_info = ERROR_HANDLER.handle_error(
                    e,
                    context=f"Pipeline '{pipeline_name}' with params {final_params}"
                )
                error_detail = {
                    "pipeline": pipeline_name,
                    "error_type": error_info["type"],
                    "error_message": error_info["message"],
                    "timestamp": error_info["timestamp"],
                    "is_retryable": error_info["is_retryable"],
                }
                state["pipeline_errors"].append(error_detail)
                
                if ViraAIConfig.VERBOSE_ERRORS:
                    print(f"      [ERROR] Pipeline '{pipeline_name}' failed: {error_info['message']}")

                return {
                    "pipeline": pipeline_name,
                    "parameters": suggested_params,
                    "result": {"error": "Internal execution error", "details": str(e)},
                    "success": False,
                }

        elif pipeline_name == "ask_user":
            # Special handling for asking the user
            question = suggested_params.get("question", "Can you clarify?")
            result = self._act_ask_user(question)
            state["pipeline_results"]["ask_user"] = result
            return {
                "pipeline": "ask_user",
                "parameters": {"question": question},
                "result": result,
                "success": True,
            }
        else:
            # ENHANCED: Structured error for unknown pipeline
            error_msg = f"Pipeline not found in registry: {pipeline_name}"
            error_detail = {
                "pipeline": pipeline_name,
                "error_type": "PipelineNotFound",
                "error_message": error_msg,
                "timestamp": datetime.now().isoformat(),
                "is_retryable": False,
            }
            state["pipeline_errors"].append(error_detail)
            
            return {
                "pipeline": pipeline_name,
                "parameters": suggested_params,
                "result": {"error": "Pipeline not found"},
                "success": False,
            }

    def _act_ask_user(self, question: str) -> Dict[str, Any]:
        """
        Simulates asking the user a question.
        """
        # print(f"      [INTERACTION] Agent asks user: '{question}'")
        return {
            "question": question,
            "status": "waiting_for_user",
            "simulated_response": "User provides clarification (simulated)",
        }

    def _build_pipeline_parameters(
        self,
        pipeline_name: str,
        pipeline_info: Dict[str, Any],
        suggested_params: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        required = pipeline_info.get("required_params", [])
        optional = pipeline_info.get("optional_params", [])

        final_params: Dict[str, Any] = {}

        for param in required:
            if param in suggested_params and suggested_params[param] not in (
                None,
                "",
                [],
            ):
                final_params[param] = suggested_params[param]
            else:
                auto_val = self._auto_fill_param(
                    pipeline_name, param, state, suggested_params
                )
                if auto_val is not None:
                    final_params[param] = auto_val

        for param in optional:
            if param in suggested_params:
                final_params[param] = suggested_params[param]
            else:
                auto_val = self._auto_fill_param(
                    pipeline_name, param, state, suggested_params, optional=True
                )
                if auto_val is not None:
                    final_params[param] = auto_val

        return final_params

    def _auto_fill_param(
        self,
        pipeline_name: str,
        param: str,
        state: Dict[str, Any],
        suggested_params: Dict[str, Any],
        optional: bool = False,
    ) -> Any:
        ctx = state.get("context", {})
        parsed_intent = ctx.get("parsed_intent", {}) # Prioritize parsed intent
        results = state.get("pipeline_results", {})

        # Prioritize explicit intent from query
        if param == "target_role":
            return (
                parsed_intent.get("target_role")
                or suggested_params.get("target_role")
                or ctx.get("target_role")
                or "UNKNOWN"
            )

        if param == "current_role":
            return (
                parsed_intent.get("current_role")
                or suggested_params.get("current_role") 
                or ctx.get("current_role")
            )

        if param == "role":
            return (
                parsed_intent.get("target_role") # Often 'role' implies target
                or suggested_params.get("role")
                or ctx.get("target_role")
                or ctx.get("current_role")
            )

        if param in ("target_role", "current_role", "role"):
            if param in suggested_params and suggested_params[param]:
                return suggested_params[param]

        if param == "user_profile":
            profile = {
                "current_role": parsed_intent.get("current_role") or ctx.get("current_role"),
                "target_role": parsed_intent.get("target_role") or ctx.get("target_role"),
                "current_skills": ctx.get("current_skills", []),
                "experience_years": ctx.get("experience_years"),
                "location": ctx.get("location"),
                "proficiency": ctx.get("proficiency", {}),
            }
            return profile

        if param == "user_activities":
            return ctx.get("user_activities", [])

        if param == "recent_query":
            return state.get("query")

        if param == "skill_gaps":
            if "skill_gap_analyzer" in results:
                gaps = results["skill_gap_analyzer"].get("skill_gaps", [])
                if gaps:
                    return gaps
            if "skills_fetcher" in results:
                return results["skills_fetcher"].get("skills", [])
            return suggested_params.get("skill_gaps", [])

        if param == "user_query":
            return suggested_params.get("user_query") or state.get("query", "")

        if param == "user_id":
            return ctx.get("user_id", "unknown_user")

        if param == "session_id":
            return ctx.get("session_id", "session_1")

        if param == "new_event":
            new_event: Dict[str, Any] = {"query": state.get("query")}
            if "skill_gap_analyzer" in results:
                new_event["skills"] = results["skill_gap_analyzer"].get(
                    "skill_gaps", []
                )
            if "course_recommender" in results:
                courses = results["course_recommender"].get(
                    "recommended_courses", []
                )
                new_event["courses"] = [c["course_name"] for c in courses]
            return new_event

        if param == "previous_context":
            # Priority: session-persisted context > within-query context_preserver output.
            if "_persisted_context" in ctx:
                return ctx["_persisted_context"]
            if "previous_context" in ctx:
                return ctx["previous_context"]
            if "context_preserver" in results:
                return results["context_preserver"]
            return None

        if param == "query":
            return state.get("query")

        if param == "location":
            return ctx.get("location", "Remote")

        if param == "learning_style":
            return (
                suggested_params.get("learning_style")
                or ctx.get("preferred_learning_style", "online")
            )

        if param == "course_ids":
            return suggested_params.get("course_ids", [])

        return None

    def _observe(self, actions_taken: List[Dict], state: Dict):
        print("\n[OBSERVE] Collecting and storing pipeline results...")

        for action in actions_taken:
            if action.get("success"):
                pipeline_name = action["pipeline"]
                state["pipeline_results"][pipeline_name] = action["result"]
                print(f"   [OK] Stored result from: {pipeline_name}")

    def _build_scope_and_filter_block(self, state: Dict) -> str:
        """
        Build dynamic instruction block for _synthesize_answer based on:
          - response_scope (focused / moderate / comprehensive)
          - query_type (new_topic / continuation / filter_refine / follow_up)
          - filter_instruction (LLM's description of what to filter)
        All decisions are made by the LLM in _parse_intent — no keyword heuristics here.
        """
        ctx = state.get("context", {})
        scope = ctx.get("_response_scope", "moderate")
        query_type = ctx.get("_query_type", "new_topic")
        filter_instruction = ctx.get("_filter_instruction")
        parsed_intent = ctx.get("parsed_intent", {})

        blocks = []

        # ---- Role priority ----
        blocks.append("""
        ROLE PRIORITY: If the user's query mentions specific roles (e.g. "from Data Analyst
        to Data Architect"), the answer MUST be about those roles. The user profile is
        background context only — never override query-stated roles with profile roles.""")

        # ---- Response scope ----
        if scope == "focused":
            blocks.append("""
        RESPONSE SCOPE: FOCUSED — The user asked a specific, narrow question.
        Answer ONLY what was asked. Do NOT add unsolicited career roadmaps, skill gap analyses,
        or course recommendations unless the user explicitly requested them.
        Keep the answer concise and directly targeted.""")
        elif scope == "comprehensive":
            blocks.append("""
        RESPONSE SCOPE: COMPREHENSIVE — The user explicitly wants full, detailed coverage.
        Provide a thorough answer with roadmaps, timelines, course lists, skill breakdowns,
        and any other relevant details.""")
        else:  # moderate
            blocks.append("""
        RESPONSE SCOPE: MODERATE — Provide a helpful, reasonably detailed answer.
        Cover the main points the user asked about with some supporting detail,
        but don't overload with information they didn't request.""")

        # ---- Filter / Refine mode ----
        if query_type == "filter_refine" and filter_instruction:
            # Get cached courses from context for the LLM to work with
            last_pipeline = ctx.get("_last_pipeline_results", {})
            cr = last_pipeline.get("course_recommender", {}) if last_pipeline else {}
            cached_courses = cr.get("recommended_courses", [])

            if cached_courses:
                blocks.append(f"""
        FILTER / REFINE MODE: The user wants to filter, narrow, compare, or select from
        previously given results. Do NOT generate a brand-new list of courses.
        Instead, operate on the CACHED COURSES below and apply this instruction:
        "{filter_instruction}"

        CACHED COURSES FROM PREVIOUS RESPONSE:
        {json.dumps(cached_courses[:10], indent=2)}

        Rules for filtering:
        - Select, rank, compare, or remove courses ONLY from the cached list above.
        - Preserve original course metadata (name, provider, duration, rating, price).
        - Still apply [Internal Course] / [External Course] labels to every course.""")

        elif query_type == "continuation":
            blocks.append("""
        CONTINUATION MODE: This query continues the same topic as the previous response.
        Build upon what was already discussed. Reference the previous answer's context
        naturally. Do NOT repeat information that was already given.""")

        elif query_type == "follow_up":
            blocks.append("""
        FOLLOW-UP MODE: This is a related question on the same general topic.
        Use the context from the previous conversation but address the new angle
        the user is asking about.""")

        return "\n".join(blocks)

    def _synthesize_answer(self, state: Dict) -> str:
        print("\n[SYNTHESIZE] Creating learner-facing answer...")

        # Prepare search results section if available
        search_section = ""
        if state.get("search_results") and state["search_results"].get("found"):
            search_data = state["search_results"]
            search_section = f"""

UNIVERSAL SEARCH RESULTS:
Summary: {search_data.get('summary', 'N/A')}
Snippets: {json.dumps(search_data.get('snippets', []), indent=2)}
Confidence: {search_data.get('confidence', 0)}

IMPORTANT: Integrate search results with pipeline data for a complete answer.
If search provides general context, use it to enhance the pipeline recommendations.
"""

        complexity = state.get("complexity", "complex")
        is_simple = complexity == "simple"

        user_context_str = json.dumps(state['context'], indent=2)
        history_context = ""
        conversation_history = state['context'].get("conversation_history", [])

        if conversation_history:
            recent_history = conversation_history[-4:]
            history_context = "PREVIOUS CONVERSATION:\n" + "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]
            )

        if is_simple:
            simple_context = {
                "user_id": state['context'].get('user_id'),
                "current_role": state['context'].get('current_role'),
                "target_role": state['context'].get('target_role'),
                "current_skills": state['context'].get('current_skills'),
                "experience_level": state['context'].get('experience_level'),
                "user_profile_snapshot": state['context'].get('user_profile_snapshot', {}),
                "conversation_history_available": "YES - see Retrieve Memory and Pipeline Results below" if state['context'].get('_viraai_full_history') else "NO"
            }
            user_context_str = json.dumps(simple_context, indent=2)
        cp_result = state["pipeline_results"].get("context_preserver", {})
        context_summary = ""
        if cp_result:
            recent_qs = cp_result.get("recent_queries", [])
            recent_skills = cp_result.get("recent_skills", [])
            last_courses = cp_result.get("last_recommended_courses", [])
            if recent_qs or recent_skills or last_courses:
                context_summary = "\nPREVIOUS SESSION CONTEXT (from context_preserver):"
                if recent_qs:
                    context_summary += f"\n- Previous queries: {', '.join(recent_qs[-3:])}"
                if recent_skills:
                    context_summary += f"\n- Skills discussed: {', '.join(recent_skills[:10])}"
                if last_courses:
                    context_summary += f"\n- Last recommended courses: {', '.join(last_courses[:5])}"

        # ---- TASK: EXPLICITLY CATEGORIZE COURSES FOR SYNTHESIS ----
        pr = state.get("pipeline_results", {})
        internal_platform_courses = []
        
        for pipe_name, pipe_data in pr.items():
            if not isinstance(pipe_data, dict): continue
            
            # 1. Standard course recommender
            if "recommended_courses" in pipe_data:
                for c in pipe_data["recommended_courses"]:
                    if c.get("course_origin") == "internal":
                        internal_platform_courses.append(c)

        courses_prompt_block = f"""
        INTERNAL CAREERVIRA COURSES (STRICTLY USE ONLY THESE):
        {json.dumps(internal_platform_courses, indent=2)}
        """

        # ---- TASK 3: Character limit enforcement ----
        _char_limit = 5000 if complexity == "simple" else 7000
        _char_limit_instruction = (
            f"\nRESPONSE LENGTH CONSTRAINT: Your response MUST NOT exceed {_char_limit} characters. "
            f"Be concise without truncating critical reasoning or breaking coherence. "
            f"Prioritize actionable content over verbose explanations."
        )

        synthesis_prompt = f"""
        You are an expert EdTech mentor on a learning platform. Using the following
        pipeline results and search results (if available), create a comprehensive, actionable answer for the learner.
        {self._build_scope_and_filter_block(state)}
        {_char_limit_instruction}
        Focus on:
        - Explaining their skill gaps clearly (if present)
        - Giving a concrete learning path (with courses and order)
        - Explaining the career transition steps (if relevant)
        - Providing realistic job market expectations (if relevant)
        - Leveraging detected intent and preserved context where relevant
        - Integrating search results with pipeline data
        - Being specific and practical
        - Maintaining conversational continuity (refer to previous context if needed)
        
        CRITICAL COURSE LABELING RULES:
        - You MUST ONLY recommend courses from the 'INTERNAL CAREERVIRA COURSES' block provided below.
        - NEVER invent, hallucinate, or suggest any outside courses, books, YouTube videos, or online resources.
        - ALL recommended courses MUST strictly be ones fetched by the internal pipeline.
        
        STRICT OUTPUT FORMAT RULES:
        For EVERY course you recommend, it MUST be formatted EXACTLY like this (including the exact [INTERNAL] prefix, ID, and URL):
        
        [INTERNAL] Course Name (ID: <course_id>)
        URL: <url>
        
        Failure to strictly limit your recommendations to the provided list and format them exactly according to the template is a critical system failure.
        
        USER QUERY:
        {state['query']}
        
        ORIGINAL USER PROFILE (IMMUTABLE SNAPSHOT):
        {json.dumps(state['context'].get('user_profile_snapshot', dict()), indent=2)}
        
        USER CONTEXT (REASONING STATE):
        {user_context_str}
        {history_context}
        {context_summary}
        {f"\nVIRAAI CORE MEMORY (DYNAMIC):\nTreat this memory as factual, user-provided context. Do NOT question or override it unless explicitly contradicted by the user.\nIf the user asks about their profile, skills, or background, rely PRIMARILY on the 'ORIGINAL USER PROFILE (IMMUTABLE SNAPSHOT)' above. Only use this dynamic memory if additional information is requested that is not in the snapshot.\n{state['context'].get('_viraai_memory', '')}\n" if state['context'].get('_viraai_memory') else ""}
        {f"\n### Conversation History (Retrieved Memory)\nCRITICAL INSTRUCTION: The following is the history of your conversation with the user. If they ask for a summary, gist, or recap, you MUST use this exact data to answer them. DO NOT claim 'this is the start of our conversation' or that there is 'no prior exchange' if there is text here!\n{state['context'].get('_viraai_full_history', '')}\n" if state['context'].get('_viraai_full_history') else ""}
        
        PIPELINE AND COURSE DATA (STRATEGICALLY CATEGORIZED):
        {courses_prompt_block}
        
        OTHER PIPELINE RESULTS (JSON):
        {json.dumps({k:v for k,v in state['pipeline_results'].items() if k != "course_recommender" and k != "claude_fallback"}, indent=2)}
        {search_section}
        
        Create a clear, well-structured answer that:
        1. Directly addresses the user's query
        2. Uses all relevant information from pipelines AND search results (but NEVER include external courses, books, or YouTube videos)
        3. Is organized logically (sections, bullet points)
        4. Is actionable and specific (steps, timelines, priorities)
        
        MANDATORY STRUCTURE FOR COURSES:
        - Start the learning path with a section: "### Recommended CareerVira Courses"
        - List EVERY internal course you recommend under this section using the strict format:
          [INTERNAL] Course Name (ID: <course_id>)
          URL: <url>
        - DO NOT create any sections for external courses or external resources.
        
        5. Labels EVERY course as [INTERNAL] — no exceptions. Includes the exact ID and URL provided.
        6. Do not include or hallucinate any external links or outside courses.
        7. Stays within the {_char_limit}-character limit.
        
        Answer:
        """

        response = self._retry_with_backoff(self.model.generate_content, synthesis_prompt)
        answer = response.text

        # ---- TASK 3: Hard character limit enforcement (trim if LLM exceeded) ----
        if len(answer) > _char_limit:
            print(f"   [CHAR_LIMIT] Answer exceeded {_char_limit} chars ({len(answer)}). Trimming...")
            # Trim at the last complete sentence before the limit
            trimmed = answer[:_char_limit]
            last_period = trimmed.rfind('.')
            last_newline = trimmed.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > _char_limit * 0.7:  # Only if we can keep 70%+
                answer = trimmed[:cut_point + 1]
            else:
                answer = trimmed  # Fallback: hard cut
            print(f"   [CHAR_LIMIT] Trimmed to {len(answer)} characters.")

        print(f"   [OK] Answer synthesized ({len(answer)} characters)")
        return answer

    def _critique_and_refine(self, answer: str, state: Dict) -> str:
        """
        Critiques the synthesized answer and refines it.
        Asks: "What is weak here? What could fail?"
        """
        print("\n[CRITIQUE] Critiquing and refining answer...")
        
        complexity = state.get("complexity", "complex")
        is_simple = complexity == "simple"
        
        if is_simple:
            print("   [CRITIQUE] Using lightweight validation (Accuracy, Relevance, Context)...")
            critique_prompt = f"""
            You are a helpful editor. Briefly review this answer.
            
            USER QUERY: {state['query']}
            
            TRUSTED USER PROFILE (SOURCE OF TRUTH — do NOT flag this data as hallucinated or fabricated):
            {json.dumps(state['context'].get('user_profile_snapshot', dict()), indent=2)}
            
            ANSWER:
            {answer}
            
            IMPORTANT: The user profile above is verified, real data provided by the user themselves.
            If the answer references the user's role, skills, experience, or target role from the profile above,
            that is CORRECT and TRUSTED — do NOT remove, question, or flag it as assumed or hallucinated.
            
            Only refine if there are genuinely wrong facts unrelated to the user's own profile data.
            If the answer correctly uses the profile data to respond, return it UNCHANGED.
            
            Respond in STRICT JSON:
            {{
                "critique": "Brief feedback...",
                "refined_answer": "Improved text (or original if good)"
            }}
            """
        else:
            print("   [CRITIQUE] Using deep validation (Accuracy, Completeness, Logic)...")
            critique_prompt = f"""
            You are a critical editor. Deeply review the following answer.
            
            USER QUERY: {state['query']}
            CURRENT ANSWER:
            {answer}
            
            TRUSTED USER PROFILE (SOURCE OF TRUTH — do NOT flag this data as hallucinated or fabricated):
            {json.dumps(state['context'].get('user_profile_snapshot', dict()), indent=2)}
            
            CONTEXT:
            {json.dumps(state['context'], indent=2)}
            
            CRITIQUE INSTRUCTIONS:
            1. Check if the answer directly addresses the user's intent.
            2. Check for hallucinations or unsupported claims — BUT any data matching the TRUSTED USER PROFILE
               above (role, skills, experience, target role, name) is REAL and must NOT be flagged.
            3. Ensure tone is helpful and professional.
            4. Verify logical flow and completeness for complex requests.
            
            Respond in STRICT JSON:
            {{
                "critique": "Detailed critique...",
                "refined_answer": "Refined answer text (or original if no changes needed)"
            }}
            """
        try:
            # Single attempt with one retry limit built into generate_content logic usually, 
            # but here we manual handle parsing failure gracefully.
            response = self.model.generate_content(critique_prompt)
            
            # Attempt to parse
            raw = self._parse_json_response(response.text, retry_on_fail=False)
            
            if raw is None:
                # One single retry with stricter formatting instruction
                print("   [RETRY] Critique JSON parse failed. One retry attempt...")
                retry_prompt = critique_prompt + "\n\n**CRITICAL**: Respond with ONLY valid JSON. No markdown blocks."
                response = self.model.generate_content(retry_prompt)
                raw = self._parse_json_response(response.text, retry_on_fail=True)
            
            if raw is None:
                # FALLBACK: Return original answer instead of looping or crashing
                print("   [WARN] Critique JSON parse failed after retry. Returning original answer.")
                return answer
            
            # [CRITIQUE] FAST PASS for Simple/High-Score Queries
            # If complexity is simple and answer seems robust, skip detailed critique loop
            complexity = state.get("complexity", "complex")
            if complexity == "simple" and raw.get("critique") == "No critique provided." and raw.get("refined_answer") == answer:
                 # The LLM already decided it's fine.
                 pass
            elif complexity == "simple":
                 # Use a more lenient acceptance criteria
                 # If the only changes are stylistic, minimal, reject them to save time?
                 # Actually, let's just trust the LLM's "refined_answer" but ensure we don't loop forever.
                 pass
                 
            critique = raw.get("critique", "No critique provided.")
            refined_answer = raw.get("refined_answer", answer)
            
            print(f"   [CRITIQUE] Feedback: {critique[:100]}...")
            
            if refined_answer != answer:
                print("   [CRITIQUE] Answer refined based on feedback.")
                return refined_answer
            else:
                print("   [CRITIQUE] Answer accepted without changes.")
                return answer
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"   [WARN] Critique process failed: {e}. Returning original answer.")
            return answer

    def _validate_answer(self, state: Dict) -> Dict:
        print("\n[VALIDATE] Checking answer quality via self-evaluation...")

        complexity = state.get("complexity", "complex")
        is_simple = complexity == "simple"
        
        if is_simple:
            print("   [VALIDATE] Lightweight validation (simple query)...")
            validation_prompt = f"""
            Evaluate this answer for a SIMPLE query. Be generous — if the answer is relevant and accurate, it PASSES.
            
            USER QUERY: {state['query']}
            ANSWER: {state['current_answer']}
            
            TRUSTED USER PROFILE (SOURCE OF TRUTH):
            {json.dumps(state['context'].get('user_profile_snapshot', dict()), indent=2)}
            
            IMPORTANT: Any information in the answer that matches the TRUSTED USER PROFILE above
            (such as the user's role, skills, experience, or target role) is REAL, VERIFIED data —
            do NOT flag it as fabricated or contradictory. It was provided by the user themselves.
            
            ONLY FAIL (scores below 0.7) if:
            - The answer is completely irrelevant to the query
            - The answer contains clearly fabricated information that does NOT come from the user profile above
            - The answer fails to address the core question at all
            
            Otherwise, rate HIGH (0.85-0.95). Do NOT penalize for style, length, or minor omissions.
            
            Respond in STRICT JSON:
            {{
                "completeness": 0.9,
                "accuracy": 0.9,
                "relevance": 0.9,
                "actionability": 0.85,
                "clarity": 0.95,
                "overall_quality": 0.92,
                "passes_threshold": true,
                "issues": [],
                "missing_information": [],
                "suggestions_for_improvement": []
            }}
            """
        else:
            print("   [VALIDATE] Standard validation (complex query)...")
            validation_prompt = f"""
            Evaluate this EdTech answer. Your role is to catch REAL problems, not nitpick.
            
            USER QUERY: {state['query']}
            ANSWER: {state['current_answer']}
            
            TRUSTED USER PROFILE (SOURCE OF TRUTH):
            {json.dumps(state['context'].get('user_profile_snapshot', dict()), indent=2)}
            
            AVAILABLE PIPELINE RESULTS (JSON):
            {json.dumps(state['pipeline_results'], indent=2)}
            
            IMPORTANT: Any information in the answer that matches the TRUSTED USER PROFILE above
            is REAL, VERIFIED data provided by the user — do NOT flag it as hallucinated or fabricated.
            
            PASS the answer (scores 0.82+) UNLESS any of these critical issues exist:
            1. RELEVANCE FAILURE: Answer does not address the user's actual question
            2. HALLUCINATION: Answer fabricates data not supported by pipeline results, memory, OR the user profile above
            3. COVERAGE GAP: Answer ignores major pipeline data that directly answers the query
            4. CONTEXT MISMATCH: Answer contradicts the user's stated role, skills, or goals
            
            Do NOT fail for:
            - Minor style or formatting preferences
            - Not mentioning every single pipeline result
            - Length being slightly short or long
            - Missing supplementary details that weren't asked for
            - Using profile data that matches the TRUSTED USER PROFILE
            
            Respond in STRICT JSON:
            {{
                "completeness": 0.85,
                "accuracy": 0.9,
                "relevance": 0.88,
                "actionability": 0.82,
                "clarity": 0.9,
                "overall_quality": 0.87,
                "passes_threshold": true,
                "issues": ["only list genuine problems"],
                "missing_information": ["only critical missing items"],
                "suggestions_for_improvement": ["only high-impact suggestions"]
            }}
            """

        response = self.model.generate_content(validation_prompt)
        raw = self._parse_json_response(response.text)

        try:
            parsed = ValidationOutput.parse_obj(raw)
        except ValidationError as e:
            print("   [WARN] Validation JSON failed validation, using raw dict")
            print(f"      Validation error: {e}")
            parsed = ValidationOutput()

        # Weighted scoring: accuracy and relevance dominate
        accuracy_weight = 0.30
        relevance_weight = 0.30
        completeness_weight = 0.15
        actionability_weight = 0.15
        clarity_weight = 0.10

        dynamic_quality = (
            parsed.accuracy * accuracy_weight
            + parsed.relevance * relevance_weight
            + parsed.completeness * completeness_weight
            + parsed.actionability * actionability_weight
            + parsed.clarity * clarity_weight
        )

        if parsed.overall_quality is None:
            parsed.overall_quality = dynamic_quality
        else:
            # Trust the LLM's own assessment more (60/40 blend)
            parsed.overall_quality = 0.6 * parsed.overall_quality + 0.4 * dynamic_quality

        parsed.quality_score = parsed.overall_quality or 0.0

        # Penalize only when a critical pipeline failed AND it was directly requested
        penalty = 0.0
        query_lower = state.get("query", "").lower()

        if state.get("pipeline_errors"):
            for err in state["pipeline_errors"]:
                failed_pipeline = err.get("pipeline", "")
                is_critical = False
                if failed_pipeline == "skill_gap_analyzer" and any(
                    kw in query_lower for kw in ["skill", "gap", "missing"]
                ):
                    is_critical = True
                elif failed_pipeline == "course_recommender" and any(
                    kw in query_lower for kw in ["course", "recommend"]
                ):
                    is_critical = True
                elif failed_pipeline == "job_market_analyzer" and any(
                    kw in query_lower for kw in ["salary", "job", "market"]
                ):
                    is_critical = True
                elif failed_pipeline == "career_path_analyzer" and any(
                    kw in query_lower for kw in ["career", "path", "timeline"]
                ):
                    is_critical = True

                if is_critical:
                    print(f"   [WARN] Critical pipeline failed: {failed_pipeline}")
                    penalty = max(penalty, 0.15)
                else:
                    print(f"   [INFO] Non-critical pipeline failed (no penalty): {failed_pipeline}")

        pr = state.get("pipeline_results", {})
        cr = pr.get("course_recommender")
        if COURSE_STORE.available and isinstance(cr, dict):
            if cr.get("vector_courses_used") is False and any(
                kw in query_lower for kw in ["course", "recommend"]
            ):
                print("   [WARN] No internal catalog courses used; small penalty applied.")
                penalty = max(penalty, 0.10)

        if penalty > 0.0:
            parsed.quality_score = max(0.0, parsed.quality_score - penalty)
            print(f"   [VALIDATE] Applied penalty of {penalty:.2f}. Adjusted score: {parsed.quality_score:.2%}")

        # Dynamic thresholding: be generous for queries where accuracy + relevance are solid
        complexity = state.get("complexity", "complex")
        dynamic_threshold = ViraAIConfig.QUALITY_THRESHOLD

        if complexity == "simple":
            if parsed.accuracy > 0.75 and parsed.relevance > 0.75:
                dynamic_threshold = 0.70
                print(f"   [VALIDATE] Simple query: relaxed threshold -> {dynamic_threshold}")
        else:
            # For complex queries, also relax slightly if core metrics are good
            if parsed.accuracy > 0.80 and parsed.relevance > 0.80:
                dynamic_threshold = min(dynamic_threshold, 0.78)
                print(f"   [VALIDATE] Complex query with solid metrics: threshold -> {dynamic_threshold}")

        if parsed.quality_score >= dynamic_threshold:
            parsed.passes_threshold = True

        # Boost simple queries that meet the dynamic threshold
        if complexity == "simple" and parsed.quality_score >= dynamic_threshold:
            parsed.quality_score = max(parsed.quality_score, ViraAIConfig.QUALITY_THRESHOLD + 0.01)

        print("   [STATS] Quality Scores:")
        print(f"      Completeness:  {parsed.completeness:.1%}")
        print(f"      Accuracy:      {parsed.accuracy:.1%}")
        print(f"      Relevance:     {parsed.relevance:.1%}")
        print(f"      Actionability: {parsed.actionability:.1%}")
        print(f"      Clarity:       {parsed.clarity:.1%}")
        print("      -------------------------")
        print(f"      OVERALL (after penalties): {parsed.quality_score:.1%}")

        return parsed.dict()

    def _validate_intermediate(self, pipeline_name: str, result: Dict) -> Dict:
        """
        Validates the output of a single pipeline step using a fast schema-based check.
        Avoids LLM calls for obvious pass/fail cases.
        """
        print(f"   [VALIDATE] Checking intermediate result from {pipeline_name}...")

        if not result or not isinstance(result, dict):
            print(f"      [WARN] Empty or non-dict result from {pipeline_name}")
            return {"is_valid": False, "issues": ["Empty result"], "confidence": 0.0}

        # Known required keys per pipeline
        required_keys_map = {
            "skill_gap_analyzer": ["skill_gaps", "required_skills"],
            "course_recommender": ["recommended_courses"],
            "skills_fetcher": ["skills"],
            "intent_detector": ["primary_intent"],
            "context_preserver": ["recent_queries"],
            "career_path_analyzer": ["typical_timeline"],
            "job_market_analyzer": ["average_salary", "demand_trend"],
        }
        required_keys = required_keys_map.get(pipeline_name, [])
        missing = [k for k in required_keys if k not in result]
        if missing:
            print(f"      [WARN] Missing required keys in {pipeline_name}: {missing}")
            # Only fail validation if all required keys are absent
            if len(missing) == len(required_keys):
                return {"is_valid": False, "issues": [f"Missing keys: {missing}"], "confidence": 0.1}
            # Partial data - still valid but with a note
            print(f"      [INFO] Partial data ({len(missing)} missing fields). Allowing output.")
            return {"is_valid": True, "issues": [f"Optional fields missing: {missing}"], "confidence": 0.7}

        print(f"      [OK] Schema check passed for {pipeline_name}")
        return {"is_valid": True, "issues": [], "confidence": 0.9}

    def _validate_intermediate_llm(self, pipeline_name: str, result: Dict) -> Dict:
        """
        Full LLM-based intermediate validation (kept for reference, not called in main flow).
        """
        print(f"   [VALIDATE-LLM] Checking intermediate result from {pipeline_name}...")
        time.sleep(1)  # Reduced from 2s

        validation_prompt = f"""
You are a strict data validator. Check if the following JSON output from the '{pipeline_name}' pipeline
is valid, complete, and makes sense given the context of an EdTech agent.

PIPELINE: {pipeline_name}
RESULT JSON:
{json.dumps(result, indent=2)}

CRITERIA:
1. Must not be empty or null.
2. Must contain relevant fields for {pipeline_name}.
3. Values should not be "N/A" or "Unknown" unless absolutely unavoidable.
4. No obvious hallucinations or broken text.
5. OPTIONAL FIELDS: Do NOT reject if fields like 'rating', 'reviews_count', 'duration', or 'price' are missing, unless the user EXPLICITLY asked for them.
6. ALLOW PARTIAL DATA: It is better to return a course with missing metadata than no course at all.

Respond in STRICT JSON:

Respond in STRICT JSON:
{{
    "is_valid": true/false,
    "issues": ["list of specific issues if any"],
    "confidence": 0.0 to 1.0
}}
"""
        response = self.model.generate_content(validation_prompt)
        raw = self._parse_json_response(response.text)

        try:
            parsed = IntermediateValidationOutput.parse_obj(raw)
        except ValidationError:
            # Fallback if LLM returns bad JSON
            parsed = IntermediateValidationOutput(is_valid=True, confidence=0.5, issues=["Validation JSON parsing failed"])

        if not parsed.is_valid:
            print(f"      [WARN] Intermediate validation failed for {pipeline_name}: {parsed.issues}")
        else:
            print(f"      [OK] Intermediate result valid (Confidence: {parsed.confidence:.2f})")

        return parsed.dict()

    def _reflect(self, state: Dict, last_actions: List[Dict]) -> Dict:
        """
        Reflects on the success/failure of the last iteration's actions.
        Saves lessons to learning_memory if mistakes were made.
        """
        print("\n[REFLECTION] Analyzing recent actions...")
        time.sleep(1)  # Reduced from 2s

        reflection_prompt = f"""
You are ViraAI's reflective conscience. Analyze the last set of actions taken by the agent.
Did they succeed? Did they provide useful information? What should be done differently next time?

LAST ACTIONS:
{json.dumps(last_actions, indent=2)}

CURRENT STATE SUMMARY:
- Query: {state['query']}
- Iteration: {state['iterations']}
- Pipeline Errors: {state.get('pipeline_errors', [])}

Respond in STRICT JSON:
{{
    "success": true/false,
    "mistake": "Description of any mistake or inefficiency (or null)",
    "correction": "What to do differently in the next reasoning step (or null)",
    "lesson_learned": "A general rule to avoid this mistake in future (or null)",
    "why_success_or_fail": "Brief explanation"
}}
"""
        response = self._retry_with_backoff(self.model.generate_content, reflection_prompt)
        raw = self._parse_json_response(response.text)

        try:
            parsed = ReflectionOutput.parse_obj(raw)
        except ValidationError:
             # Handle extended fields manually if Pydantic model isn't updated yet
             parsed = ReflectionOutput(success=True, why_success_or_fail="Reflection parsing failed, assuming success.")

        print(f"   [REFLECT] Success: {parsed.success}")
        if parsed.mistake:
            print(f"   [REFLECT] Mistake: {parsed.mistake}")
        if parsed.correction:
            print(f"   [REFLECT] Correction: {parsed.correction}")
            
        # Save lesson to memory if provided (and not already present)
        lesson = raw.get("lesson_learned")
        if lesson and lesson not in self.learning_memory:
            self.learning_memory.append(lesson)
            print(f"   [MEMORY] Learned new lesson: {lesson}")

        return parsed.dict()

    def explore_alternative_paths(self, state: Dict) -> Dict:
        """
        Explores multiple reasoning paths when faced with ambiguity or complexity.
        """
        print("\n[EXPLORE] Brainstorming alternative solution paths...")
        time.sleep(1)  # Reduced from 2s

        exploration_prompt = f"""
You are ViraAI's strategic planner. The current situation requires creative problem solving.
Brainstorm 2-3 distinct approaches to answer the user's query effectively.

USER QUERY:
{state['query']}

CONTEXT:
{json.dumps(state['context'], indent=2)}

Generate 2-3 alternative paths. For each path, provide a name, description, pros, cons, and a score (0.0-1.0).
Then select the best path and explain why.

Respond in STRICT JSON:
{{
    "paths": [
        {{
            "name": "Path A Name",
            "description": "Description...",
            "pros": "...",
            "cons": "...",
            "score": 0.9
        }},
        ...
    ],
    "selected_path": "Path A Name",
    "reasoning": "Why this path is best..."
}}
"""
        response = self.model.generate_content(exploration_prompt)
        raw = self._parse_json_response(response.text)

        try:
            parsed = PathExplorationOutput.parse_obj(raw)
        except ValidationError:
            # Fallback
            parsed = PathExplorationOutput(
                paths=[],
                selected_path="Default Path",
                reasoning="Exploration failed, proceeding with standard reasoning."
            )

        print(f"   [EXPLORE] Generated {len(parsed.paths)} paths.")
        print(f"   [EXPLORE] Selected: {parsed.selected_path}")
        print(f"   [EXPLORE] Reasoning: {parsed.reasoning[:100]}...")

        return parsed.dict()

    def recover_from_error(self, error: str, state: Dict, failed_action: Dict) -> Dict:
        """
        Proposes a recovery strategy when an action fails.
        """
        print(f"\n[RECOVERY] Handling error in {failed_action.get('pipeline')}: {error}")
        time.sleep(1)  # Reduced from 2s

        recovery_prompt = f"""
You are ViraAI's crisis manager. An action has failed. Propose a recovery strategy.

FAILED ACTION:
{json.dumps(failed_action, indent=2)}

ERROR:
{error}

CONTEXT:
{json.dumps(state['context'], indent=2)}

Respond in STRICT JSON:
{{
    "strategy_name": "Retry / Fallback / Skip",
    "description": "What we will do",
    "action_plan": "Specific instructions for the next step (e.g., 'retry with param X', 'call pipeline Y instead')",
    "confidence": 0.0 to 1.0
}}
"""
        response = self.model.generate_content(recovery_prompt)
        raw = self._parse_json_response(response.text)

        try:
            parsed = RecoveryStrategy.parse_obj(raw)
        except ValidationError:
            parsed = RecoveryStrategy(
                strategy_name="Safe Mode Fallback",
                description="LLM recovery failed, defaulting to safe mode.",
                action_plan="Skip this step and proceed with best effort.",
                confidence=0.5
            )

        print(f"   [RECOVERY] Strategy: {parsed.strategy_name}")
        print(f"   [RECOVERY] Plan: {parsed.action_plan}")

        return parsed.dict()

    def _enforce_course_labels(self, answer: str, state: Dict) -> str:
        """
        TASK 4: Post-synthesis enforcement pass (STRICT).
        Scans the final answer text for any course names from ALL pipeline results
        that are missing their [Internal Course], [External Course], or [External - Web Search] label,
        and appends the correct label. This is a robust fail-safe.
        """
        if not answer:
            return answer

        pr = state.get("pipeline_results", {})
        # GATHER ALL COURSES FROM ALL POTENTIAL PIPELINE SOURCES
        all_courses_found = []
        
        for pipe_name, pipe_data in pr.items():
            if not isinstance(pipe_data, dict):
                continue
                
            # 1. Standard course recommender format
            if "recommended_courses" in pipe_data:
                for c in pipe_data["recommended_courses"]:
                    name = c.get("course_name") or c.get("title")
                    if name:
                        all_courses_found.append({
                            "name": name,
                            "origin": c.get("course_origin", "internal"),
                            "source": c.get("source", "vector_catalog"),
                            "url": c.get("url", ""),
                            "course_id": c.get("course_id", "")
                        })
            
            # 2. Fallback search results are entirely disabled to block external sources
            pass

        # 3. Universal search results are entirely disabled for course logic
        pass

        if not all_courses_found:
            return answer

        # Deduplicate and sort by name length descending (to match longest names first)
        unique_courses = {}
        for c in all_courses_found:
            name = str(c["name"]).strip()
            if not name or len(name) < 4: continue # Skip very short or empty names
            if name not in unique_courses:
                unique_courses[name] = c

        sorted_names = sorted(unique_courses.keys(), key=len, reverse=True)
        
        modified = answer
        labels_added = 0
        import re

        for name in sorted_names:
            course = unique_courses[name]
            course_url = course.get("url", "")
            course_id = course.get("course_id", "")
            
            if not course_url or not course_id:
                continue

            label = "[INTERNAL]"
            
            # Use regex to find the matching course name in lines
            lines = modified.split('\n')
            new_lines = []
            
            for line in lines:
                line_lower = line.lower()
                name_lower = name.lower()
                
                if name_lower in line_lower:
                    pattern = re.compile(re.escape(name), re.IGNORECASE)
                    match = pattern.search(line)
                    
                    if match:
                        actual_name = match.group(0)
                        
                        # Only apply styling if it isn't already perfectly formatted
                        if f"{label} {actual_name} (ID: {course_id})" not in line:
                            # Reformat the line to ensure it fits the exact struct:
                            # [INTERNAL] Course Name (ID: XXXX)\nURL: <url>
                            replacement_text = f"{label} {actual_name} (ID: {course_id})\nURL: {course_url}"
                            
                            # Clean up old tags if any
                            clean_line = line.replace("[INTERNAL]", "").replace(f"(ID: {course_id})", "").replace(f"URL: {course_url}", "").strip()
                            if not clean_line: clean_line = actual_name # if line was just the course name
                            
                            line = clean_line.replace(actual_name, replacement_text)
                            labels_added += 1

                new_lines.append(line)
            
            modified = '\n'.join(new_lines)

        if labels_added > 0:
            print(f"   [LABEL_ENFORCE] Added {labels_added} strict internal course label(s) to final response.")

        return modified

    def _prepare_final_response(self, state: Dict) -> Dict:
        """
        ENHANCED: Now includes metrics, error summary, and configuration details.
        """
        # Get metrics and error summaries
        metrics_summary = METRICS_COLLECTOR.get_summary()
        error_summary = ERROR_HANDLER.get_error_summary()
        
        response = {
            "answer": state["current_answer"],
            "quality_score": state["quality_score"],
            "complexity": state.get("complexity", "unknown"),
            "iterations": state["iterations"] + 1,
            "pipelines_used": list(state["pipeline_results"].keys()),
            "reasoning_trace": state["reasoning_trace"],
            "validation_history": state["validation_history"],
            "success": state["quality_score"] >= ViraAIConfig.QUALITY_THRESHOLD,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": ViraAIConfig.MODEL_NAME,
                "threshold": ViraAIConfig.QUALITY_THRESHOLD,
                # ENHANCED: Add metrics and error tracking
                "metrics": metrics_summary,
                "error_summary": error_summary,
                "configuration": ViraAIConfig.get_config(),
            },
        }
        
        # Add search results if available
        if state.get("search_results"):
            response["search_results"] = state["search_results"]
        
        return response

    def _parse_json_response(self, text: str, retry_on_fail: bool = False) -> Any:
        """
        Parses JSON from LLM response with multiple fallback strategies.
        
        Args:
            text: Raw LLM response text
            retry_on_fail: If True, return None on parse failure to trigger upstream retry
        """
        raw = text.strip()

        # Remove markdown code block markers
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find first complete JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try to find JSON array
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass

        # All strategies failed
        print("   [WARN] Failed to parse JSON from LLM after multiple strategies.")
        if retry_on_fail:
            return None  # Signal caller to retry
        return {"error": "parse_failed", "raw": raw}


# ============================================================================
# MAIN DEMO
# ============================================================================


def main():
    print("\n" + "=" * 80)
    print("[DEMO] ViraAI EdTech Deep-Thinking ReAct Loop Demo")
    print("=" * 80)
    print("\nThis demo shows:")
    print("  1. Deep reasoning (explicit thought process via planning JSON)")
    print("  2. ReAct loop (Reason -> Act -> Observe -> Synthesize -> Validate)")
    print("  3. Self-validation with structural penalties (forces re-tries on failures)")
    print("  4. Course recommendations backed by your MXBAI catalog (with source flags)")
    print("  5. Multiple varied EdTech scenarios (6) using different pipelines")
    print("\n" + "=" * 80 + "\n")

# ============================================================================
# USER DATABASE (Sample User Profiles)
# ============================================================================

# Centralized user database for Mode B manual query execution
USER_DATABASE = {
    "user_001": {
        "user_id": "user_001",
        "session_id": "sess_001",
        "name": "Alex Chen",
        "current_role": "Data Analyst",
        "target_role": "Machine Learning Engineer",
        "current_skills": ["SQL", "Excel", "Basic Python"],
        "experience_years": 2,
        "location": "Remote",
        "budget": "medium",
        "preferred_learning_style": "online",
        "user_activities": [],
    },
    "user_002": {
        "user_id": "user_002",
        "session_id": "sess_002",
        "name": "Priya Sharma",
        "current_role": "Business Analyst",
        "target_role": "Senior Data Analyst",
        "current_skills": ["Excel", "Basic SQL"],
        "experience_years": 3,
        "location": "India",
        "budget": "low",
        "preferred_learning_style": "weekend_intensive",
        "user_activities": [],
    },
    "user_003": {
        "user_id": "user_003",
        "session_id": "sess_003",
        "name": "Marcus Johnson",
        "current_role": "Software Engineer",
        "target_role": None,
        "current_skills": ["Python", "Linux"],
        "experience_years": 4,
        "location": "Europe",
        "budget": "flexible",
        "preferred_learning_style": "hands_on",
        "user_activities": [
            {"page": "course/aws-fundamentals", "action": "view"},
            {"page": "course/kubernetes-basics", "action": "click"},
            {"page": "course/devops-pipeline", "action": "add_to_wishlist"},
        ],
    },
    "user_004": {
        "user_id": "user_004",
        "session_id": "sess_004",
        "name": "Sarah Kim",
        "current_role": "Backend Developer",
        "target_role": "Data Engineer",
        "current_skills": ["Python", "APIs", "Basic SQL"],
        "experience_years": 3,
        "location": "Remote",
        "budget": "medium",
        "preferred_learning_style": "online",
        "user_activities": [],
    },
    "user_005": {
        "user_id": "user_005",
        "session_id": "sess_005",
        "name": "David Martinez",
        "current_role": "Learner",
        "target_role": "Data Scientist",
        "current_skills": ["Python", "Basic Statistics"],
        "experience_years": 1,
        "location": "Remote",
        "budget": "flexible",
        "preferred_learning_style": "mixed",
        "user_activities": [
            {"page": "track/data-science", "action": "in_progress"},
            {"page": "course/stats-basics", "action": "completed"},
        ],
        "previous_context": {
            "recent_queries": [
                "how to build ds portfolio",
                "statistics for ml",
            ],
            "recent_skills": ["Python", "Statistics"],
            "last_recommended_courses": ["stats-basics"],
        },
    },
    "user_006": {
        "user_id": "user_006",
        "session_id": "sess_006",
        "name": "Aisha Patel",
        "current_role": "Student",
        "target_role": "AI Engineer",
        "current_skills": [],
        "experience_years": 0,
        "location": "India",
        "budget": "low",
        "preferred_learning_style": "visual",
        "user_activities": [],
    },
    "user_007": {
        "user_id": "user_007",
        "session_id": "sess_007",
        "name": "Tom Wilson",
        "current_role": "Student",
        "target_role": None,
        "current_skills": [],
        "experience_years": 0,
        "location": "Remote",
        "budget": "low",
        "preferred_learning_style": "video",
        "user_activities": [],
    },
}


# ============================================================================
# SAMPLE SCENARIOS (Predefined Test Cases)
# ============================================================================

def get_sample_scenarios():
    """
    Returns a list of predefined test scenarios for Mode A execution.
    Each scenario contains a query and associated user context.
    """
    return [
        {
            "name": "Top 3 ML courses for Data Analyst (explicit count)",
            "query": (
                "I'm a data analyst and want to start with machine learning. "
                "Recommend exactly 3 beginner-friendly courses from the platform."
            ),
            "context": USER_DATABASE["user_001"].copy(),
        },
        {
            "name": "Weekend-only SQL & analytics plan (time constrained)",
            "query": (
                "I can only study on weekends for about 5 hours. "
                "I want to strengthen my SQL and analytics skills over 3 months. "
                "How should I plan this and which courses should I take?"
            ),
            "context": USER_DATABASE["user_002"].copy(),
        },
        {
            "name": "DevOps / Cloud browsing -> intent detection & next 2 courses",
            "query": (
                "Based on my recent browsing on the platform about AWS, Kubernetes, "
                "and CI/CD, what do I think I'm trying to achieve and which 2 "
                "courses should I take next?"
            ),
            "context": USER_DATABASE["user_003"].copy(),
        },
        {
            "name": "Remote Data Engineer: skills + market + path",
            "query": (
                "I want to become a remote data engineer. "
                "What skills do I need, what does the job market look like, and "
                "which courses on the platform should I follow in order?"
            ),
            "context": USER_DATABASE["user_004"].copy(),
        },
        {
            "name": "Continue DS track using preserved context",
            "query": (
                "I'm halfway through the data science track. "
                "Can you remind me what I've learned so far and suggest the next "
                "steps for this month using my previous activity?"
            ),
            "context": USER_DATABASE["user_005"].copy(),
        },
        {
            "name": "High school student exploring AI path (2-year horizon)",
            "query": (
                "I'm in 12th grade and interested in AI but have no background. "
                "How should I start on this platform for the next 2 years?"
            ),
            "context": USER_DATABASE["user_006"].copy(),
        },
        {
            "name": "Vague query requiring clarification",
            "query": "I want to learn something new.",
            "context": USER_DATABASE["user_007"].copy(),
        },
    ]


# ============================================================================
# MODE SELECTION & EXECUTION FUNCTIONS
# ============================================================================

def display_user_database():
    """
    Displays all available users in the database with their key information.
    Used in Mode B to help the user select a profile.
    """
    print("\n" + "=" * 80)
    print("AVAILABLE USER PROFILES")
    print("=" * 80)
    
    for user_id, user_data in USER_DATABASE.items():
        print(f"\n[{user_id}] {user_data.get('name', 'Unknown')}")
        print(f"  Current Role: {user_data.get('current_role', 'N/A')}")
        print(f"  Target Role: {user_data.get('target_role', 'N/A')}")
        print(f"  Skills: {', '.join(user_data.get('current_skills', [])) or 'None'}")
        print(f"  Experience: {user_data.get('experience_years', 0)} years")
        print(f"  Location: {user_data.get('location', 'N/A')}")
        print(f"  Budget: {user_data.get('budget', 'N/A')}")
    
    print("\n" + "=" * 80)


def select_execution_mode():
    """
    Prompts the user to select between Mode A (Sample Queries) and Mode B (Manual Query).
    Returns the selected mode as a string: 'A' or 'B'.
    """
    print("\n" + "=" * 80)
    print("VIRAAI AGENT - EXECUTION MODE SELECTION")
    print("=" * 80)
    print("\nPlease select an execution mode:\n")
    print("  [A] Run Sample Queries")
    print("      - Executes all predefined test scenarios")
    print("      - No user input required")
    print("      - Useful for testing and demonstration\n")
    print("  [B] Run Manual Query")
    print("      - Select a user profile from the database")
    print("      - Enter a custom query")
    print("      - Interactive execution mode\n")
    print("=" * 80)
    
    while True:
        choice = input("\nEnter your choice (A or B): ").strip().upper()
        if choice in ['A', 'B']:
            return choice
        print("[ERROR] Invalid choice. Please enter 'A' or 'B'.")


def execute_mode_a(agent):
    """
    Mode A: Executes all predefined sample scenarios sequentially.
    
    Args:
        agent: Initialized ReActAgent instance
    """
    print("\n" + "#" * 80)
    print("MODE A: RUNNING SAMPLE QUERIES")
    print("#" * 80)
    
    sample_scenarios = get_sample_scenarios()
    overall_start = time.time()
    
    for idx, scenario in enumerate(sample_scenarios, 1):
        print("\n" + "#" * 80)
        print(f"SCENARIO {idx}/{len(sample_scenarios)}: {scenario['name']}")
        print("#" * 80)

        user_query = scenario["query"]
        user_context = scenario["context"]

        start_time = time.time()
        result = agent.process_query(user_query, user_context)
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print(f"[RES] FINAL RESULTS - {scenario['name']}")
        print("=" * 80)

        print("\n[SUM] Summary:")
        print(f"   Success: {'[YES]' if result['success'] else '[NO]'}")
        print(f"   Quality Score: {result['quality_score']:.1%}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Pipelines Used: {', '.join(result['pipelines_used']) or 'None'}")
        print(f"   Time Elapsed: {elapsed_time:.2f}s")

        print("\n[ANS] Final Answer:")
        print("-" * 80)
        safe_answer = result["answer"].encode("ascii", "replace").decode("ascii")
        print(safe_answer)
        print("-" * 80)

        print("\n[TRACE] Reasoning Trace (strategies per iteration):")
        for i, reasoning in enumerate(result["reasoning_trace"], 1):
            print(f"\n   Iteration {i}:")
            print(f"   Strategy: {str(reasoning.get('strategy', 'N/A'))[:200]}...")

        print("\n" + "=" * 80)
        print(f"[DONE] Scenario '{scenario['name']}' Complete!")
        print("=" * 80 + "\n")
        time.sleep(10)  # Rate limit protection

    overall_elapsed = time.time() - overall_start
    print("\n" + "#" * 80)
    print(f"[ALL] All {len(sample_scenarios)} scenarios processed in {overall_elapsed:.2f}s")
    print("#" * 80 + "\n")


def execute_mode_b(agent):
    """
    Mode B: Interactive mode where the user selects a profile and enters a custom query.
    
    Args:
        agent: Initialized ReActAgent instance
    """
    print("\n" + "#" * 80)
    print("MODE B: MANUAL QUERY EXECUTION")
    print("#" * 80)
    
    # Display available user profiles
    display_user_database()
    
    # Get user selection
    while True:
        user_id = input("\nEnter the User ID to assume (e.g., user_001): ").strip()
        if user_id in USER_DATABASE:
            break
        print(f"[ERROR] User ID '{user_id}' not found. Please select from the list above.")
    
    # Fetch user data
    user_context = USER_DATABASE[user_id].copy()
    user_name = user_context.get('name', 'Unknown')
    
    print(f"\n[OK] Selected profile: {user_name} ({user_id})")
    print(f"     Role: {user_context.get('current_role', 'N/A')}")
    print(f"     Skills: {', '.join(user_context.get('current_skills', [])) or 'None'}")
    
    # [CONTINUATION] Conversation Loop
    user_query = "" # Initialize for loop
    conversation_history = [] # Maintain history for this session

    while True:
        # Get custom query if first turn, else prompt for follow-up
        if not user_query:
             print("\n" + "-" * 80)
             print("Enter the query (or 'exit'/'quit' to stop):")
             print("-" * 80)
             
             query_lines = []
             while True:
                 line = input()
                 if line == "" and query_lines:
                     break
                 if line:
                     query_lines.append(line)
             
             user_query = " ".join(query_lines).strip()
        
        if not user_query or user_query.lower() in ['exit', 'quit', 'no']:
            print("[INFO] Exiting conversation.")
            break
        
        print(f"\n[OK] Query received: {user_query[:100]}...")
        
        # Inject history into context
        user_context["conversation_history"] = conversation_history

        # Execute query
        print("\n" + "#" * 80)
        print(f"EXECUTING QUERY FOR: {user_name} ({user_id})")
        print("#" * 80)
        
        start_time = time.time()
        result = agent.process_query(user_query, user_context)
        elapsed_time = time.time() - start_time
        
        # Update history with current turn
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": result["answer"]})
        
        # Display results
        print("\n" + "=" * 80)
        print("[RES] FINAL RESULTS")
        print("=" * 80)
        
        print("\n[SUM] Summary:")
        print(f"   User: {user_name} ({user_id})")
        print(f"   Success: {'[YES]' if result['success'] else '[NO]'}")
        print(f"   Quality Score: {result['quality_score']:.1%}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Pipelines Used: {', '.join(result['pipelines_used']) or 'None'}")
        print(f"   Time Elapsed: {elapsed_time:.2f}s")
        
        print("\n[ANS] Final Answer:")
        print("-" * 80)
        safe_answer = result["answer"].encode("ascii", "replace").decode("ascii")
        print(safe_answer)
        print("-" * 80)
        
        print("\n[TRACE] Reasoning Trace (strategies per iteration):")
        for i, reasoning in enumerate(result["reasoning_trace"], 1):
            print(f"\n   Iteration {i}:")
            print(f"   Strategy: {str(reasoning.get('strategy', 'N/A'))[:200]}...")
        
        # [CONTINUATION] Update context and Prompt for next turn
        if "final_context" in result:
             user_context = result["final_context"]
             print("\n[CONTEXT] Context updated for follow-up.")
        
        print("\n" + "=" * 80)
        print("Would you like to ask a follow-up or know more? (type query or 'no'/'exit')")
        user_query = "" # Reset for next input loop
        
    print("\n" + "=" * 80)
    print("[DONE] Manual Query Execution Complete!")
    print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================

def main():
    """
    Main entry point for the ViraAI EdTech Deep-Thinking ReAct Agent.
    
    Workflow:
        1. Initialize Claude model and agent components
        2. Prompt user to select execution mode (A or B)
        3. Execute selected mode:
           - Mode A: Run all predefined sample scenarios
           - Mode B: Interactive query with user profile selection
    """
    print("\n" + "=" * 80)
    print("[DEMO] ViraAI EdTech Deep-Thinking ReAct Loop Demo")
    print("=" * 80)
    print("\nThis demo shows:")
    print("  1. Deep reasoning (explicit thought process via planning JSON)")
    print("  2. ReAct loop (Reason -> Act -> Observe -> Synthesize -> Validate)")
    print("  3. Self-validation with structural penalties (forces re-tries on failures)")
    print("  4. Course recommendations backed by MXBAI catalog (with source flags)")
    print("  5. Multiple varied EdTech scenarios using different pipelines")
    print("\n" + "=" * 80 + "\n")

    # Initialize system components
    print("\n" + "=" * 80)
    print("[BOT] Initializing ViraAI system...")
    model = initialize_model()
    
    # Set model reference for COURSE_STORE to enable query expansion
    set_course_store_model(model)
    
    # Initialize Universal Search system
    set_universal_search_model(model)
    print("[OK] Universal Search initialized.")
    
    pipeline_registry = PipelineRegistry()
    agent = ReActAgent(model, pipeline_registry)
    print("[OK] System initialization complete.")
    print("=" * 80)
    
    # Mode selection
    mode = select_execution_mode()
    
    # Execute selected mode
    if mode == 'A':
        execute_mode_a(agent)
    elif mode == 'B':
        execute_mode_b(agent)
    
    print("\n[EXIT] ViraAI Agent session terminated.")


if __name__ == "__main__":
    main()