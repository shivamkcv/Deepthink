import os
import json
import uuid
import math
import re
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ViraAIMemory")

# ============================================================================
# CONFIGURATION
# ============================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

def _ensure_api_key():
    """Ensure API key is loaded, trying to import from main demo if missing."""
    global ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        try:
            from viraai_react_demo import ViraAIConfig
            ANTHROPIC_API_KEY = ViraAIConfig.ANTHROPIC_API_KEY
        except (ImportError, AttributeError):
            pass
            
    # Final Streamlit secrets fallback
    if not ANTHROPIC_API_KEY:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "ANTHROPIC_API_KEY" in st.secrets:
                ANTHROPIC_API_KEY = str(st.secrets["ANTHROPIC_API_KEY"]).strip()
            else:
                ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
        except:
            ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

import anthropic

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_SEMANTIC = True
    logger.info("Semantic embeddings loaded successfully.")
except ImportError:
    HAS_SEMANTIC = False
    logger.warning("sentence-transformers or scikit-learn missing. Falling back to keyword/decay retrieval.")

# ============================================================================
# CONSTANTS & BUDGETS
# ============================================================================
MAX_EPISODES_PER_USER = 50
MAX_PROFILE_FACTS_PER_USER = 30
TOKEN_BUDGET = 2500  
MAX_RECENT_TURNS = 6
HIGH_RES_EXCHANGES = 2
HIGH_RES_TURNS = HIGH_RES_EXCHANGES * 2

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text.split()) * 1.3)

# ============================================================================
# DATA MODELS
# ============================================================================
class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class AnswerSnapshot(BaseModel):
    answer_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Episode(BaseModel):
    episode_id: str
    summary: str
    importance: float
    turn_created: int
    timestamp: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    embedding: Optional[List[float]] = None
    
    def get_decayed_score(self, current_turn: int) -> float:
        turns_passed = current_turn - self.turn_created
        decay_lambda = 0.05
        base_score = self.importance * math.exp(-decay_lambda * turns_passed)
        refresh_boost = min(3.0, self.access_count * 0.5) 
        return base_score + refresh_boost

class UserProfileFact(BaseModel):
    fact_id: str
    category: str
    content: str
    importance: float
    timestamp: datetime = Field(default_factory=datetime.now)

class ActiveTask(BaseModel):
    goal: str = ""
    constraints: List[str] = Field(default_factory=list)
    selected_items: List[str] = Field(default_factory=list)

class MemoryState(BaseModel):
    session_id: str
    user_id: str
    current_turn: int = 0
    recent_turns: List[ConversationTurn] = Field(default_factory=list)
    last_answers: List[AnswerSnapshot] = Field(default_factory=list)
    rolling_summary: str = ""
    task_state: ActiveTask = Field(default_factory=ActiveTask)
    episodes: List[Episode] = Field(default_factory=list)
    profile_facts: List[UserProfileFact] = Field(default_factory=list)

# ============================================================================
# MEMORY SERVICE
# ============================================================================
class ViraAIMemoryService:
    def __init__(self, anthropic_client=None):
        if not anthropic_client:
            _ensure_api_key()
        self.client = anthropic_client or anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = CLAUDE_MODEL
        self.memory_store: Dict[str, MemoryState] = {}
        self._store_lock = threading.Lock()
        self._session_locks: Dict[str, threading.Lock] = {}
        self._executor = ThreadPoolExecutor(max_workers=5)

    def _get_lock(self, session_id: str) -> threading.Lock:
        with self._store_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]

    def _get_state(self, session_id: str, user_id: str) -> MemoryState:
        with self._get_lock(session_id):
            if session_id not in self.memory_store:
                self.memory_store[session_id] = MemoryState(session_id=session_id, user_id=user_id)
            return self.memory_store[session_id]

    def _save_state(self, state: MemoryState):
        with self._get_lock(state.session_id):
            if len(state.episodes) > MAX_EPISODES_PER_USER:
                logger.debug(f"Pruning episodes for {state.user_id}")
                state.episodes.sort(key=lambda ep: ep.get_decayed_score(state.current_turn), reverse=True)
                state.episodes = state.episodes[:MAX_EPISODES_PER_USER]
            
            if len(state.profile_facts) > MAX_PROFILE_FACTS_PER_USER:
                logger.debug(f"Pruning profile facts for {state.user_id}")
                state.profile_facts.sort(key=lambda f: f.importance, reverse=True)
                state.profile_facts = state.profile_facts[:MAX_PROFILE_FACTS_PER_USER]
            
            self.memory_store[state.session_id] = state

    # --- RETRIEVAL & ASSEMBLY ---
    
    def _retrieve_episodes(self, state: MemoryState, query: str = "") -> List[Tuple[Episode, float]]:
        if not state.episodes:
            return []
            
        scored = []
        query_emb = None
        if HAS_SEMANTIC and query.strip():
            try:
                query_emb = embedder.encode([query])[0]
            except Exception as e:
                logger.error(f"Failed to encode query: {e}")
        
        for ep in state.episodes:
            decay_score = ep.get_decayed_score(state.current_turn)
            similarity = 0.0
            
            if HAS_SEMANTIC and query_emb is not None and ep.embedding is not None:
                try:
                    ep_emb = np.array(ep.embedding).reshape(1, -1)
                    q_emb = np.array(query_emb).reshape(1, -1)
                    similarity = cosine_similarity(q_emb, ep_emb)[0][0]
                except Exception:
                    pass
            
            final_score = (similarity * 0.7) + (min(decay_score, 10.0) / 10.0 * 0.3) if similarity > 0 else decay_score
            scored.append((ep, float(final_score)))
            
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def get_compiled_context(self, session_id: str, user_id: str, current_query: str = "") -> str:
        state = self._get_state(session_id, user_id)
        current_budget = TOKEN_BUDGET
        compiled_parts = {}

        task_text = (
            f"Goal: {state.task_state.goal or 'None'}\n"
            f"Constraints: {', '.join(state.task_state.constraints) or 'None'}\n"
            f"Selected: {', '.join(state.task_state.selected_items) or 'None'}"
        )
        compiled_parts['active_task'] = task_text
        current_budget -= estimate_tokens(task_text)

        recent_text = "\n".join([f"{t.role.capitalize()}: {t.content}" for t in state.recent_turns])
        compiled_parts['recent_turns'] = recent_text or "No turns."
        current_budget -= estimate_tokens(recent_text)

        facts_sorted = sorted(state.profile_facts, key=lambda x: x.importance, reverse=True)
        profile_lines = []
        for f in facts_sorted:
            line = f"- [{f.category.upper()}] {f.content}"
            if estimate_tokens(line) < current_budget:
                profile_lines.append(line)
                current_budget -= estimate_tokens(line)
        compiled_parts['user_profile'] = "\n".join(profile_lines) if profile_lines else "No profile data yet."

        retrieved = self._retrieve_episodes(state, current_query)
        epi_lines = []
        for list_idx, (ep, score) in enumerate(retrieved):
            line = f"- {ep.summary} [Relevance: {score:.2f}]"
            if estimate_tokens(line) < current_budget:
                epi_lines.append(line)
                current_budget -= estimate_tokens(line)
                if list_idx < 3: 
                    ep.access_count += 1 
        compiled_parts['past_episodes_retrieved'] = "\n".join(epi_lines) if epi_lines else "No past episodes."

        summary_tokens = estimate_tokens(state.rolling_summary)
        if summary_tokens < current_budget:
            compiled_parts['conversation_summary'] = state.rolling_summary or "No older context summarized yet."
        else:
            compiled_parts['conversation_summary'] = "Omitted."

        return f"""<memory>
<user_profile>
{compiled_parts.get('user_profile')}
</user_profile>

<active_task>
{compiled_parts.get('active_task')}
</active_task>

<past_episodes_retrieved>
{compiled_parts.get('past_episodes_retrieved')}
</past_episodes_retrieved>

<conversation_summary>
{compiled_parts.get('conversation_summary')}
</conversation_summary>

<recent_turns>
{compiled_parts.get('recent_turns')}
</recent_turns>
</memory>"""

    # --- INGESTION & PIPELINE ---
    
    def update_memory_after_turn(self, session_id: str, user_id: str, user_msg: str, assistant_msg: str):
        state = self._get_state(session_id, user_id)
        
        with self._get_lock(session_id):
            state.current_turn += 1
            state.recent_turns.append(ConversationTurn(role="user", content=user_msg))
            state.recent_turns.append(ConversationTurn(role="assistant", content=assistant_msg))
            
            state.last_answers.append(AnswerSnapshot(answer_id=uuid.uuid4().hex, content=assistant_msg))
            if len(state.last_answers) > 5:
                state.last_answers = state.last_answers[-5:]
            
            needs_compression = len(state.recent_turns) > MAX_RECENT_TURNS
            turns_to_compress = []
            if needs_compression:
                turns_to_compress = state.recent_turns[:-HIGH_RES_TURNS]
                state.recent_turns = state.recent_turns[-HIGH_RES_TURNS:]

        self._save_state(state)
        
        if needs_compression:
            self._executor.submit(self._compress_short_term, session_id, user_id, turns_to_compress)
            
        self._executor.submit(self._extract_insights, session_id, user_id, user_msg, assistant_msg)

    def _compress_short_term(self, session_id: str, user_id: str, old_turns: List[ConversationTurn]):
        state = self._get_state(session_id, user_id)
        turns_text = "\n".join([f"{t.role}: {t.content}" for t in old_turns])
        
        prompt = f"""You are compressing conversation logs.
Current Summary: {state.rolling_summary}
New turns to incorporate:
{turns_text}
Write a concise, updated running summary. Output ONLY a strict JSON object:
{{
    "summary": "your updated narrative text",
    "facts": ["list of user facts/preferences observed"],
    "decisions": ["list of important insights/decisions made"]
}}"""
        
        try:
            res = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = res.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                with self._get_lock(session_id):
                    state.rolling_summary = data.get("summary", state.rolling_summary)
                    
                    for fact in data.get("facts", []):
                        state.profile_facts.append(UserProfileFact(
                            fact_id=uuid.uuid4().hex,
                            category="preference",
                            content=fact,
                            importance=3.0
                        ))
                    
                    for decision in data.get("decisions", []):
                        embedding = None
                        if HAS_SEMANTIC:
                            try:
                                embedding = embedder.encode([decision])[0].tolist()
                            except Exception:
                                pass
                        state.episodes.append(Episode(
                            episode_id=uuid.uuid4().hex,
                            summary=decision,
                            importance=5.0,
                            turn_created=state.current_turn,
                            embedding=embedding
                        ))    
                self._save_state(state)
                logger.info(f"[{session_id}] Rolling summary, facts, and decisions updated.")
        except Exception as e:
            logger.error(f"[{session_id}] Compression failed: {e}")

    def _extract_insights(self, session_id: str, user_id: str, user_msg: str, assistant_msg: str):
        state = self._get_state(session_id, user_id)
        
        with self._get_lock(session_id):
            current_profile = [{"id": f.fact_id, "cat": f.category, "content": f.content} for f in state.profile_facts]
            current_task = state.task_state.model_dump()
            current_turn = state.current_turn

        prompt = f"""Analyze the latest conversation turn to update the memory map. Output ONLY valid JSON.
        
Current Task State: {json.dumps(current_task)}
Current Profile Facts: {json.dumps(current_profile)}

Latest Turn:
User: {user_msg}
Assistant: {assistant_msg}

Instructions:
1. "importance": Rate importance (1.0 to 10.0). Ignore formatting/greetings.
2. "episode_summary": A 1-sentence summary if importance >= 4.0, else null.
3. "task_updates": Identify new goals, constraints, or selections. Pivot if needed.
4. "profile_facts": Extract user facts. To UPDATE an existing fact, output its specific string "id". Else use "NEW". Let category be "role", "preference", "goal", or "experience".

EXPECTED JSON SCHEMA:
{{
    "importance": float,
    "episode_summary": "string or null",
    "task_updates": {{
        "goal": "string (or null)",
        "constraints": ["additions string text"],
        "selected_items": ["additions string text"]
    }},
    "profile_facts": [
        {{"id": "id or NEW", "category": "category", "content": "fact"}}
    ]
}}"""
        try:
            res = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = res.content[0].text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if not json_match:
                raise ValueError("No JSON block found in LLM response.")
            data = json.loads(json_match.group())
            
            with self._get_lock(session_id):
                importance = float(data.get("importance", 1.0))
                summary = data.get("episode_summary")
                if importance >= 4.0 and summary:
                    embedding = None
                    if HAS_SEMANTIC:
                        try:
                            embedding = embedder.encode([summary])[0].tolist()
                        except Exception as e:
                            logger.error(f"Semantic encoding failed: {e}")
                    
                    state.episodes.append(Episode(
                        episode_id=str(uuid.uuid4()),
                        summary=summary,
                        importance=importance,
                        turn_created=current_turn,
                        embedding=embedding
                    ))

                task_up = data.get("task_updates", {})
                if task_up.get("goal"):
                    if state.task_state.goal and task_up["goal"] != state.task_state.goal:
                        state.task_state.constraints = []
                        state.task_state.selected_items = []
                        logger.info(f"[{session_id}] Detected Task Pivot. Target locked: {task_up['goal']}")
                    state.task_state.goal = task_up["goal"]
                    
                if task_up.get("constraints"):
                    state.task_state.constraints = list(set(state.task_state.constraints + task_up["constraints"]))
                if task_up.get("selected_items"):
                    state.task_state.selected_items = list(set(state.task_state.selected_items + task_up["selected_items"]))

                for pf in data.get("profile_facts", []):
                    f_id = pf.get("id")
                    if f_id and f_id != "NEW":
                        for f in state.profile_facts:
                            if f.fact_id == f_id:
                                f.content = pf["content"]
                                f.importance = importance
                                f.timestamp = datetime.now()
                    else:
                        state.profile_facts.append(UserProfileFact(
                            fact_id=str(uuid.uuid4()),
                            category=pf.get("category", "preference"),
                            content=pf["content"],
                            importance=importance
                        ))
                        
            self._save_state(state)
            
        except Exception as e:
            logger.error(f"[{session_id}] Insight extraction failed: {e}")

# ============================================================================
# MODULE GUARD — No side effects on import
# ============================================================================

if __name__ == "__main__":
    print("[INFO] viraai_memory.py loaded as standalone — no interactive REPL. Import from viraai_react_demo.py instead.")
