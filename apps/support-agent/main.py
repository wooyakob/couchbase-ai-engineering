"""Interactive support-agent CLI.

Setup (once):
    agentc init
    git add -A && git commit -m "..."   # commit FIRST: `agentc index` records
    agentc index .                      # whether the tree was dirty at index time,
    agentc publish                      # baking that into the snapshot `publish`
                                         # checks — index on a dirty tree and
                                         # `publish` refuses even after a later commit.

Run:
    python main.py [user_id]
"""

import os
import sys
import uuid

import agentc
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv(os.getenv("ENV_FILE", ".env"), usecwd=True))  # ENV_FILE lets .env.server / .env.capella run side by side

from agent.graph import build_graph  # noqa: E402 — also applies the shared warning filters
from agent.memory import AgentMemory  # noqa: E402


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else "demo-user"
    session_id = f"{user_id}::{uuid.uuid4().hex[:8]}"

    try:
        catalog = agentc.Catalog()
    except Exception as e:
        raise RuntimeError(
            f"Couldn't load the Agent Catalog: {e}. Have you run the one-time setup "
            "in this file's docstring (agentc init / index / commit / publish)? "
            "Also check AGENT_CATALOG_CONN_STRING/USERNAME/PASSWORD/BUCKET — these "
            "are a separate credential namespace from CB_*. "
            "See docs/troubleshooting.md."
        ) from e
    span = catalog.Span(name="support_agent", session=session_id)
    graph = build_graph(catalog, span)

    # This conversation is one Agent Memory session; recall spans all of the
    # user's prior sessions, and durable facts live in their profile session.
    try:
        memory = AgentMemory(user_id, conversation_id=session_id)
    except Exception as e:  # noqa: BLE001 — CLI keeps working without the memory server
        print(f"(agent memory unavailable: {e}; continuing without persistence)")
        memory = None

    config = {"configurable": {"thread_id": session_id}}
    print(f"support-agent ready (user={user_id}, thread={session_id}). Ctrl-D to exit.")

    with span:
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue

            state = graph.invoke(
                {"messages": [("user", user_input)], "user_id": user_id,
                 "needs_human": False, "is_last_step": False, "previous_node": None},
                config,
            )
            reply = state["messages"][-1].content
            if memory is not None:
                try:
                    memory.add_exchange(user_input, reply)
                except Exception as e:  # noqa: BLE001 — never let a memory write kill the chat
                    print(f"(couldn't save this exchange to memory: {e})")
            print(f"\nagent> {reply}")

    print("\nbye — memory persisted via the Agent Memory server, "
          "trace in ai.agent_activity.logs")


if __name__ == "__main__":
    main()
