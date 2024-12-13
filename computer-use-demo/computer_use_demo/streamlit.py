"""
Entrypoint for streamlit, see https://docs.streamlit.io/
"""

import asyncio
import base64
import cProfile
import glob
import logging
import os
import pstats
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import StrEnum
from functools import partial, wraps
from pathlib import Path, PosixPath
from typing import cast

import httpx
import streamlit as st
from anthropic import RateLimitError
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from streamlit.delta_generator import DeltaGenerator

from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop,
)
from computer_use_demo.tools import ToolResult

CONFIG_DIR = PosixPath("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"
STREAMLIT_STYLE = """
<style>
    /* Highlight the stop button in red */
    button[kind=header] {
        background-color: rgb(255, 75, 75);
        border: 1px solid rgb(255, 75, 75);
        color: rgb(255, 255, 255);
    }
    button[kind=header]:hover {
        background-color: rgb(255, 51, 51);
    }
     /* Hide the streamlit deploy button */
    .stAppDeployButton {
        visibility: hidden;
    }
</style>
"""

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"
INTERRUPT_TEXT = "(user stopped or interrupted and wrote the following)"
INTERRUPT_TOOL_ERROR = "human stopped or interrupted tool execution"

# Set up logging at the top of the file
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def profile_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            return profile.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profile)
            stats.sort_stats("cumulative")
            # Create a string buffer to capture the output
            import io

            stream = io.StringIO()
            stats.print_stats(20)  # Remove file parameter
            output = stream.getvalue()
            # Print to terminal
            logger.info(f"\nProfile for {func.__name__}:\n{output}")

    return wrapper


def time_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        # Print to terminal
        logger.info(f"{func.__name__} took {duration:.2f} seconds")
        return result

    return wrapper


# @time_decorator
def get_state_name():
    """Get state name input without triggering heavy operations"""
    logger.info("Starting get_state_name")

    with st.sidebar:
        with st.container():
            result = st.text_input(
                "State name",
                key="state_name_input",
                on_change=None,  # Disable automatic callbacks
            )
            # logger.info(f"State name input value: {result}")
            return result


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


# @time_decorator
def setup_state():
    """Initialize the session state with empty values"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tools" not in st.session_state:
        st.session_state.tools = {}
    if "last_saved_timestamp" not in st.session_state:
        st.session_state.last_saved_timestamp = None
    if "api_key" not in st.session_state:
        # Try to load API key from file first, then environment
        st.session_state.api_key = load_from_storage("api_key") or os.getenv(
            "ANTHROPIC_API_KEY", ""
        )
    if "provider" not in st.session_state:
        st.session_state.provider = (
            os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
        )
    if "provider_radio" not in st.session_state:
        st.session_state.provider_radio = st.session_state.provider
    if "model" not in st.session_state:
        _reset_model()
    if "auth_validated" not in st.session_state:
        st.session_state.auth_validated = False
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "prompt_caching" not in st.session_state:
        st.session_state.prompt_caching = False
    if "only_n_most_recent_images" not in st.session_state:
        st.session_state.only_n_most_recent_images = 3
    if "only_n_most_recent_messages" not in st.session_state:
        st.session_state.only_n_most_recent_messages = 500
    if "visible_messages" not in st.session_state:
        st.session_state.visible_messages = 20
    if "custom_system_prompt" not in st.session_state:
        st.session_state.custom_system_prompt = load_from_storage("system_prompt") or ""
    if "hide_images" not in st.session_state:
        st.session_state.hide_images = False
    if "in_sampling_loop" not in st.session_state:
        st.session_state.in_sampling_loop = False
    if "render_api_responses" not in st.session_state:
        st.session_state.render_api_responses = False


def _reset_model():
    st.session_state.model = PROVIDER_TO_DEFAULT_MODEL_NAME[
        cast(APIProvider, st.session_state.provider)
    ]


def is_message_visible(idx: int, total_messages: int) -> bool:
    """Show only the most recent N messages"""
    if st.session_state.visible_messages == -1:
        return True
    return idx >= (total_messages - st.session_state.visible_messages)


# @profile_decorator
async def main():
    """Render loop for streamlit"""
    setup_state()

    st.markdown(STREAMLIT_STYLE, unsafe_allow_html=True)

    st.title("Claude Computer Use Demo")

    if not os.getenv("HIDE_WARNING", False):
        st.warning(WARNING_TEXT)

    with st.sidebar:

        def _reset_api_provider():
            if st.session_state.provider_radio != st.session_state.provider:
                _reset_model()
                st.session_state.provider = st.session_state.provider_radio
                st.session_state.auth_validated = False

        provider_options = [option.value for option in APIProvider]
        st.radio(
            "API Provider",
            options=provider_options,
            key="provider_radio",
            format_func=lambda x: x.title(),
            on_change=_reset_api_provider,
        )

        st.text_input("Model", key="model")

        if st.session_state.provider == APIProvider.ANTHROPIC:
            st.text_input(
                "Anthropic API Key",
                type="password",
                key="api_key",
                on_change=lambda: save_to_storage("api_key", st.session_state.api_key),
            )
        st.text_area(
            "Custom System Prompt Suffix",
            key="custom_system_prompt",
            help="Additional instructions to append to the system prompt. see computer_use_demo/loop.py for the base system prompt.",
            on_change=lambda: save_to_storage(
                "system_prompt", st.session_state.custom_system_prompt
            ),
        )
        if st.session_state.provider == APIProvider.ANTHROPIC:
            st.checkbox(
                "Cache prompt",
                key="prompt_caching",
                help="Use prompt caching. Only available for Anthropic API.",
            )
        st.number_input(
            "Only send N most recent images",
            min_value=0,
            key="only_n_most_recent_images",
            help="To decrease the total tokens sent, remove older screenshots from the conversation. Only available if prompt caching is disabled.",
            disabled=st.session_state.provider == APIProvider.ANTHROPIC
            and st.session_state.prompt_caching,
        )
        # st.number_input(
        #     "Only send N most recent messages",
        #     min_value=-1,
        #     key="only_n_most_recent_messages",
        #     help="To decrease the total tokens sent, remove older messages from the conversation",
        # )
        st.number_input(
            "Only render N most recent messages",
            min_value=-1,
            key="visible_messages",
            help="Only show the most recent N messages in the chat to decrease render time. Set to -1 to show all messages.",
        )
        st.checkbox(
            "Render API Responses",
            key="render_api_responses",
            help="Render the API responses in the HTTP Exchange Logs tab (this is slow)",
        )
        st.checkbox("Hide screenshots", key="hide_images")

        st.divider()  # Add visual separation

        # Save State Section
        st.subheader("Save/Load State")

        new_state_name = get_state_name()
        if st.button(
            "Save",
            type="secondary",
            disabled=not new_state_name,
            key="save_state_button",
        ):
            try:
                with st.spinner("Saving state..."):
                    # Move the serialization logic inside the button click handler
                    serializable_tools = {}
                    for tool_id, tool_result in st.session_state.tools.items():
                        serializable_tools[tool_id] = {
                            "output": tool_result.output,
                            "error": tool_result.error,
                            "base64_image": tool_result.base64_image,
                            "system": tool_result.system,
                        }

                    save_to_storage_json(
                        f"state_{new_state_name}",
                        {
                            "messages": st.session_state.messages,
                            "tools": serializable_tools,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    st.session_state.last_saved_timestamp = datetime.now()
                st.success(f"Saved as '{new_state_name}'!")
            except Exception as e:
                st.error(f"Error saving state: {str(e)}")

        # Load existing state
        saved_states = get_saved_states()
        if saved_states:
            st.write("Load saved state:")
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_state = st.selectbox(
                    "Select state to load", saved_states, label_visibility="collapsed"
                )
            with col2:
                if st.button("Load", type="primary"):
                    with st.spinner("Loading state..."):
                        if load_state_with_name(selected_state):
                            st.success(f"Loaded '{selected_state}'!")
                        else:
                            st.error("Failed to load state!")

            # Delete state option
            if st.button("Delete Selected State", type="secondary"):
                try:
                    (CONFIG_DIR / f"state_{selected_state}.json").unlink()
                    st.success(f"Deleted '{selected_state}'!")
                except Exception as e:
                    st.error(f"Failed to delete state: {e}")

        if st.session_state.last_saved_timestamp:
            st.caption(
                f"Last saved: {st.session_state.last_saved_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # Reset button in its own section
        st.divider()
        if st.button("Reset Current Session", type="primary"):
            with st.spinner("Resetting..."):
                st.session_state.messages = []
                st.session_state.tools = {}
                st.session_state.last_saved_timestamp = None
                subprocess.run("pkill Xvfb; pkill tint2", shell=True)  # noqa: ASYNC221
                await asyncio.sleep(1)
                subprocess.run("./start_all.sh", shell=True)  # noqa: ASYNC221

    if not st.session_state.auth_validated:
        if auth_error := validate_auth(
            st.session_state.provider, st.session_state.api_key
        ):
            st.warning(f"Please resolve the following auth issue:\n\n{auth_error}")
            return
        else:
            st.session_state.auth_validated = True

    chat, http_logs = st.tabs(["Chat", "HTTP Exchange Logs"])
    new_message = st.chat_input(
        "Type a message to send to Claude to control the computer..."
    )

    with chat:
        total_messages = len(st.session_state.messages)

        # logger.info(f"\nTotal messages: {total_messages}")

        # # Show message count if some are hidden
        # if total_messages > 10:
        #     st.caption(f"Showing most recent 10 of {total_messages} messages")

        # with st.container():
        for idx, message in enumerate(st.session_state.messages):
            if is_message_visible(idx, total_messages):
                if isinstance(message["content"], str):
                    _render_message(message["role"], message["content"])
                elif isinstance(message["content"], list):
                    for block in message["content"]:
                        # the tool result we send back to the Anthropic API isn't sufficient to render all details,
                        # so we store the tool use responses
                        if isinstance(block, dict) and block["type"] == "tool_result":
                            _render_message(
                                Sender.TOOL,
                                st.session_state.tools[block["tool_use_id"]],
                            )
                        else:
                            _render_message(
                                message["role"],
                                cast(BetaContentBlockParam | ToolResult, block),
                            )

        # render past http exchanges
        # if http_logs.id == st.tabs(['Chat', 'HTTP Exchange Logs'])[1].id:
        if st.session_state.render_api_responses:
            for identity, (request, response) in st.session_state.responses.items():
                _render_api_response(request, response, identity, http_logs)

        # render past chats
        if new_message:
            st.session_state.messages.append(
                {
                    "role": Sender.USER,
                    "content": [
                        *maybe_add_interruption_blocks(),
                        BetaTextBlockParam(type="text", text=new_message),
                    ],
                }
            )
            _render_message(Sender.USER, new_message)

        try:
            most_recent_message = st.session_state["messages"][-1]
        except IndexError:
            return

        if most_recent_message["role"] is not Sender.USER:
            # we don't have a user message to respond to, exit early
            return

        logger.info(
            f"N most recent images: {st.session_state.only_n_most_recent_images}"
        )

        with track_sampling_loop():
            # run the agent sampling loop with the newest message
            # st.session_state.messages = st.session_state.messages[-st.session_state.only_n_most_recent_messages:]
            st.session_state.messages = await sampling_loop(
                system_prompt_suffix=st.session_state.custom_system_prompt,
                model=st.session_state.model,
                provider=st.session_state.provider,
                messages=st.session_state.messages,
                output_callback=partial(_render_message, Sender.BOT),
                tool_output_callback=partial(
                    _tool_output_callback, tool_state=st.session_state.tools
                ),
                api_response_callback=partial(
                    _api_response_callback,
                    tab=http_logs,
                    response_state=st.session_state.responses,
                ),
                api_key=st.session_state.api_key,
                only_n_most_recent_images=st.session_state.only_n_most_recent_images,
                prompt_caching=st.session_state.prompt_caching,
                # only_n_most_recent_messages=st.session_state.only_n_most_recent_messages,
            )


def maybe_add_interruption_blocks():
    if not st.session_state.in_sampling_loop:
        return []
    # If this function is called while we're in the sampling loop, we can assume that the previous sampling loop was interrupted
    # and we should annotate the conversation with additional context for the model and heal any incomplete tool use calls
    result = []
    last_message = st.session_state.messages[-1]
    previous_tool_use_ids = [
        block["id"] for block in last_message["content"] if block["type"] == "tool_use"
    ]
    for tool_use_id in previous_tool_use_ids:
        st.session_state.tools[tool_use_id] = ToolResult(error=INTERRUPT_TOOL_ERROR)
        result.append(
            BetaToolResultBlockParam(
                tool_use_id=tool_use_id,
                type="tool_result",
                content=INTERRUPT_TOOL_ERROR,
                is_error=True,
            )
        )
    result.append(BetaTextBlockParam(type="text", text=INTERRUPT_TEXT))
    return result


@contextmanager
def track_sampling_loop():
    st.session_state.in_sampling_loop = True
    yield
    st.session_state.in_sampling_loop = False


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key in the sidebar to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


def load_from_storage_json(filename: str) -> dict | list | None:
    """Load JSON data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / f"{filename}.json"
        if file_path.exists():
            import json

            with open(file_path) as f:
                return json.load(f)
    except Exception as e:
        st.write(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage_json(filename: str, data: dict | list) -> None:
    """Save JSON data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / f"{filename}.json"
        import json

        with open(file_path, "w") as f:
            json.dump(data, f)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        st.write(f"Debug: Error saving {filename}: {e}")


# @time_decorator
def _api_response_callback(
    request: httpx.Request,
    response: httpx.Response | object | None,
    error: Exception | None,
    tab: DeltaGenerator,
    response_state: dict[str, tuple[httpx.Request, httpx.Response | object | None]],
):
    """
    Handle an API response by storing it to state and rendering it.
    """
    response_id = datetime.now().isoformat()
    response_state[response_id] = (request, response)
    if error:
        _render_error(error)
    # Only render if we're actually on the HTTP logs tab
    if st.session_state.render_api_responses:
        _render_api_response(request, response, response_id, tab)


def _tool_output_callback(
    tool_output: ToolResult, tool_id: str, tool_state: dict[str, ToolResult]
):
    """Handle a tool output by storing it to state and rendering it."""
    tool_state[tool_id] = tool_output
    _render_message(Sender.TOOL, tool_output)


# @time_decorator
def _render_api_response(
    request: httpx.Request,
    response: httpx.Response | object | None,
    response_id: str,
    tab: DeltaGenerator,
):
    """Render an API response to a streamlit tab"""
    with tab:
        with st.expander(f"Request/Response ({response_id})"):
            newline = "\n\n"
            st.markdown(
                f"`{request.method} {request.url}`{newline}{newline.join(f'`{k}: {v}`' for k, v in request.headers.items())}"
            )
            st.json(request.read().decode())
            st.markdown("---")
            if isinstance(response, httpx.Response):
                st.markdown(
                    f"`{response.status_code}`{newline}{newline.join(f'`{k}: {v}`' for k, v in response.headers.items())}"
                )
                st.json(response.text)
            else:
                st.write(response)


def _render_error(error: Exception):
    if isinstance(error, RateLimitError):
        body = "You have been rate limited."
        if retry_after := error.response.headers.get("retry-after"):
            body += f" **Retry after {str(timedelta(seconds=int(retry_after)))} (HH:MM:SS).** See our API [documentation](https://docs.anthropic.com/en/api/rate-limits) for more details."
        body += f"\n\n{error.message}"
    else:
        body = str(error)
        body += "\n\n**Traceback:**"
        lines = "\n".join(traceback.format_exception(error))
        body += f"\n\n```{lines}```"
    save_to_storage(f"error_{datetime.now().timestamp()}.md", body)
    st.error(f"**{error.__class__.__name__}**\n\n{body}", icon=":material/error:")


# @time_decorator
def _render_message(
    sender: Sender,
    message: str | BetaContentBlockParam | ToolResult,
):
    """Convert input from the user or output from the agent to a streamlit message."""

    # @st.cache_data(show_spinner=False)
    # def render_cached_message(sender, message_hash, message):
    #     with st.chat_message(sender):

    # # streamlit's hotreloading breaks isinstance checks, so we need to check for class names
    is_tool_result = not isinstance(message, str | dict)
    if not message or (
        is_tool_result
        and st.session_state.hide_images
        and not hasattr(message, "error")
        and not hasattr(message, "output")
    ):
        return
    with st.chat_message(sender):
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                if message.__class__.__name__ == "CLIResult":
                    st.code(message.output)
                else:
                    st.markdown(message.output)
            if message.error:
                st.error(message.error)
            if message.base64_image and not st.session_state.hide_images:
                st.image(base64.b64decode(message.base64_image))
        elif isinstance(message, dict):
            if message["type"] == "text":
                st.write(message["text"])
            elif message["type"] == "tool_use":
                st.code(f'Tool Use: {message["name"]}\nInput: {message["input"]}')
            else:
                # only expected return types are text and tool_use
                raise Exception(f'Unexpected response type {message["type"]}')
        else:
            st.markdown(message)

    # @st.cache_data(show_spinner=False)
    # def render_cached_message(sender, message_hash, message):
    #     with st.chat_message(sender):
    #         if is_tool_result:
    #             message = cast(ToolResult, message)
    #             if message.output:
    #                 if message.__class__.__name__ == "CLIResult":
    #                     st.code(message.output)
    #                 else:
    #                     st.markdown(message.output)
    #             if message.error:
    #                 st.error(message.error)
    #             if message.base64_image and not st.session_state.hide_images:
    #                 st.image(base64.b64decode(message.base64_image))
    #         elif isinstance(message, dict):
    #             if message["type"] == "text":
    #                 st.write(message["text"])
    #             elif message["type"] == "tool_use":
    #                 st.code(f'Tool Use: {message["name"]}\nInput: {message["input"]}')
    #             else:
    #                 # only expected return types are text and tool_use
    #                 raise Exception(f'Unexpected response type {message["type"]}')
    #         else:
    #             st.markdown(message)
    #     # Create a hash of the message content for caching
    # message_hash = hash(str(message))
    # render_cached_message(sender, message_hash, message)


def get_saved_states() -> list[str]:
    """Get list of all saved state files, sorted by modification time (newest first)"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    files = glob.glob(str(CONFIG_DIR / "state_*.json"))
    # Create list of tuples with (filename, modification_time)
    files_with_times = [(f, os.path.getmtime(f)) for f in files]
    # Sort by modification time, newest first
    sorted_files = sorted(files_with_times, key=lambda x: x[1], reverse=True)
    # Extract just the state names from the sorted files
    return [Path(f[0]).stem.replace("state_", "") for f in sorted_files]


def load_state_with_name(name: str) -> bool:
    """Load a specific named state"""
    data = load_from_storage_json(f"state_{name}")
    if data and isinstance(data, dict):  # Type guard to help pyright
        st.session_state["messages"] = data["messages"]
        tools_data = data.get("tools", {})
        st.session_state["tools"] = {
            tool_id: ToolResult(
                output=tool_data.get("output"),
                error=tool_data.get("error"),
                base64_image=tool_data.get("base64_image"),
                system=tool_data.get("system"),
            )
            for tool_id, tool_data in tools_data.items()
        }
        st.session_state["last_saved_timestamp"] = datetime.fromisoformat(
            data["timestamp"]
        )
        return True
    return False


if __name__ == "__main__":
    asyncio.run(main())
