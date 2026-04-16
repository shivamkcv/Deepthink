"""
ViraAI ReAct Agent - Streamlit UI Wrapper
==========================================

Pure UI wrapper around viraai_react_demo.py.
Claude-only. In-house memory only (viraai_memory.py).

Run with: streamlit run app.py
"""

import streamlit as st
import sys
import io
import threading
import queue
import time
from contextlib import redirect_stdout
from typing import Dict, Any

from viraai_react_demo import (
    initialize_model,
    ReActAgent,
    PipelineRegistry,
    set_course_store_model,
    set_universal_search_model,
    COURSE_STORE,
    ViraAIConfig,
    USER_DATABASE,
)


class OutputCapture:
    """Captures stdout in real-time for streaming effect"""

    def __init__(self):
        self.output_queue = queue.Queue()
        self.stop_flag = threading.Event()

    def write(self, text):
        if text and text.strip():
            self.output_queue.put(text)
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    def get_output(self):
        output = []
        while not self.output_queue.empty():
            try:
                output.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return "".join(output)


def initialize_system(provider: str = "claude", model_name: str = None):
    """Initialize the ViraAI system (Claude-only, in-house memory)."""
    cache_key = f"{provider}_{model_name or 'default'}"

    if "cache_key" not in st.session_state or st.session_state.cache_key != cache_key:
        with st.spinner(f"🔧 Initializing ViraAI with {provider}..."):
            if provider == "claude" and model_name:
                ViraAIConfig.CLAUDE_MODEL = model_name

            model = initialize_model(provider=provider, model_name=model_name)
            set_course_store_model(model)
            set_universal_search_model(model)
            pipeline_registry = PipelineRegistry()
            agent = ReActAgent(model, pipeline_registry)

            st.session_state.agent = agent
            st.session_state.model = model
            st.session_state.cache_key = cache_key
            st.session_state.current_provider = provider
            st.session_state.current_model = model_name
            st.session_state.initialized = True

    return st.session_state.agent


# Monkey-patch explore_alternative_paths for robust error handling
def safe_explore_alternative_paths(self, state):
    try:
        return self._original_explore_paths(state)
    except Exception as e:
        print(f"   [WARN] Path exploration failed validation: {e}")
        print("   [RECOVERY] Using fallback path strategy to continue execution...")
        return {
            "selected_path": "Standard ReAct Execution",
            "reasoning": "Path exploration failed due to format error. Proceeding with standard execution flow.",
            "paths": [],
        }


if not hasattr(ReActAgent, "_original_explore_paths"):
    ReActAgent._original_explore_paths = ReActAgent.explore_alternative_paths
    ReActAgent.explore_alternative_paths = safe_explore_alternative_paths
    print("[INIT] Patched ReActAgent.explore_alternative_paths for robust error handling")


def format_response(result: Dict[str, Any]) -> str:
    """Format the agent's response for display"""
    answer = result.get("answer", "No answer generated")

    output = f"### 📝 Answer\n\n{answer}\n\n"
    output += "---\n\n"
    output += "### 📊 Query Metrics\n\n"
    output += f"- **Complexity:** {result.get('complexity', 'N/A').title()}\n"
    output += f"- **Quality Score:** {result.get('quality_score', 0):.1%}\n"
    output += f"- **Iterations:** {result.get('iterations', 0)}\n"
    output += f"- **Pipelines Used:** {', '.join(result.get('pipelines_used', [])) or 'None'}\n"
    output += f"- **Success:** {'✅ Yes' if result.get('success') else '❌ No'}\n"

    if result.get("reasoning_trace"):
        output += "\n\n"
        with st.expander("🧠 View Reasoning Trace"):
            for i, trace in enumerate(result["reasoning_trace"], 1):
                st.markdown(f"**Iteration {i}:**")
                st.json(trace)

    return output


def main():
    """Main Streamlit app"""

    st.set_page_config(
        page_title="ViraAI ReAct Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message { background-color: #e3f2fd; }
        .assistant-message { background-color: #f5f5f5; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🤖 ViraAI EdTech ReAct Agent")
    st.caption("Deep Thinking AI Assistant for Career Learning Paths")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("👤 User Profile")

        selected_user_id = st.selectbox(
            "Select User Profile",
            options=list(USER_DATABASE.keys()),
            format_func=lambda x: f"{USER_DATABASE[x]['name']} ({x})",
        )

        if selected_user_id:
            user_profile = USER_DATABASE[selected_user_id]
            st.markdown("---")
            st.subheader(user_profile["name"])
            st.markdown(f"**Current Role:** {user_profile['current_role']}")
            st.markdown(f"**Target Role:** {user_profile['target_role']}")
            st.markdown(f"**Experience:** {user_profile['experience_years']} years")
            st.markdown(f"**Location:** {user_profile['location']}")

            with st.expander("📚 Current Skills"):
                for skill in user_profile["current_skills"]:
                    proficiency = user_profile.get("proficiency", {}).get(skill, 0.5)
                    st.progress(proficiency, text=f"{skill} ({proficiency:.0%})")

        st.markdown("---")
        st.markdown("### 🧠 Memory Engine")
        st.info(
            "**In-House Memory** (viraai_memory.py)\n\n"
            "- Episodic memory with semantic retrieval\n"
            "- User profile fact extraction\n"
            "- Rolling conversation summaries\n"
            "- Full memory logged on each turn"
        )

        st.markdown("---")
        st.markdown("### 🤖 Model Selection")

        # Claude-only: two model choices
        model_options = {
            "claude-opus-4-6": "Claude Opus 4.6 (Deep Reasoning)",
            "claude-sonnet-4-6": "Claude Sonnet 4.6 (Fast)",
        }
        default_model = ViraAIConfig.CLAUDE_MODEL
        provider = "claude"

        try:
            default_index = list(model_options.keys()).index(default_model)
        except ValueError:
            default_index = 0

        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=default_index,
        )

        st.info(f"**Active:** {model_options[selected_model]}")

        st.markdown("---")
        st.markdown("### ⚙️ System Config")
        st.markdown(f"**Quality Threshold:** {ViraAIConfig.QUALITY_THRESHOLD:.0%}")
        st.markdown(f"**Max Iterations:** {ViraAIConfig.MAX_ITERATIONS}")

        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.session_state.user_context = None
            st.rerun()

    # Initialize system (Claude-only)
    agent = initialize_system(provider=provider, model_name=selected_model)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    import copy
    # Define user profile snapshot once if not exists
    if "user_profile_snapshot" not in st.session_state:
        st.session_state.user_profile_snapshot = copy.deepcopy(USER_DATABASE[selected_user_id])

    if "user_context" not in st.session_state:
        st.session_state.user_context = copy.deepcopy(USER_DATABASE[selected_user_id])
        # Inject the baseline snapshot into context
        st.session_state.user_context["user_profile_snapshot"] = copy.deepcopy(st.session_state.user_profile_snapshot)

    # Check if user swapped
    if st.session_state.user_context.get("user_id") != selected_user_id:
        st.session_state.user_profile_snapshot = copy.deepcopy(USER_DATABASE[selected_user_id])
        st.session_state.user_context = copy.deepcopy(USER_DATABASE[selected_user_id])
        st.session_state.user_context["user_profile_snapshot"] = copy.deepcopy(st.session_state.user_profile_snapshot)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about your career learning path..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            thinking_placeholder = st.empty()
            result_placeholder = st.empty()

            try:
                status_placeholder.info("🤔 Deep thinking in progress...")
                thinking_container = thinking_placeholder.container()
                # Safe stream capture via standard contextlib
                f = io.StringIO()
                result = None
                error_encountered = None
                captured_text = ""

                try:
                    current_history = st.session_state.messages[:-1]
                    st.session_state.user_context["conversation_history"] = current_history
                    
                    with redirect_stdout(f):
                        result = agent.process_query(
                            prompt, st.session_state.user_context
                        )
                except Exception as e:
                    error_encountered = e
                finally:
                    captured_text = f.getvalue()
                    captured_output = captured_text.split("\n") if captured_text else []

                if error_encountered:
                    import traceback

                    error_type = type(error_encountered).__name__
                    error_msg_full = str(error_encountered)

                    status_placeholder.empty()
                    thinking_placeholder.empty()

                    if (
                        "validation error" in error_msg_full.lower()
                        or "pydantic" in error_msg_full.lower()
                    ):
                        friendly_msg = f"""⚠️ **Processing Issue Detected**

The AI encountered a data validation issue during deep reasoning.

**Suggestions:**
- Try rephrasing your question
- Ask a more specific question
- Try a different user profile

**Technical details:** {error_type}
"""
                        result_placeholder.warning(friendly_msg)
                        with st.expander("🔧 Technical Error Details"):
                            st.code(error_msg_full, language="text")
                        if captured_output:
                            with st.expander("📋 Processing Log (before error)"):
                                st.code("\n".join(captured_output), language="text")
                        response_msg = (
                            f"**Processing Error:** {error_type}\n\n{friendly_msg}"
                        )
                    else:
                        friendly_msg = f"**Error during processing:** {error_type}\n\n{error_msg_full[:500]}"
                        result_placeholder.error(friendly_msg)
                        if captured_output:
                            with st.expander("📋 Processing Log"):
                                st.code("\n".join(captured_output), language="text")
                        response_msg = friendly_msg

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_msg}
                    )

                elif result:
                    if "final_context" in result:
                        st.session_state.user_context = result["final_context"]

                    status_placeholder.empty()
                    thinking_placeholder.empty()

                    formatted_response = format_response(result)
                    result_placeholder.markdown(formatted_response)

                    if captured_output:
                        with st.expander("🔍 View Processing Log"):
                            st.code("\n".join(captured_output), language="text")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": formatted_response}
                    )
                else:
                    status_placeholder.empty()
                    thinking_placeholder.empty()
                    result_placeholder.warning(
                        "⚠️ Processing completed but no result was returned. Please try again."
                    )

            except Exception as e:
                status_placeholder.empty()
                thinking_placeholder.empty()
                error_msg = f"❌ **Unexpected Error:** {str(e)}"
                result_placeholder.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

    st.markdown("---")
    st.caption("💡 ViraAI ReAct Agent — Claude-only, In-house Memory")


if __name__ == "__main__":
    main()
