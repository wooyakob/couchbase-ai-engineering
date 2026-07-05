"""Interactive support-agent CLI.

Setup (once):
    agentc init
    agentc index .
    git add -A && git commit -m "..."   # snapshots are keyed to commits
    agentc publish

Run:
    python main.py [user_id]
"""

import sys
import uuid

import agentc
import dotenv

dotenv.load_dotenv()

from agent.graph import build_graph
from agent.memory import SessionStore


def main():
    user_id = sys.argv[1] if len(sys.argv) > 1 else "demo-user"
    session_id = f"{user_id}::{uuid.uuid4().hex[:8]}"

    catalog = agentc.Catalog()
    span = catalog.Span(name="support_agent", session=session_id)
    graph = build_graph(catalog, span)
    sessions = SessionStore()

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

            sessions.append_turn(session_id, "user", user_input)
            state = graph.invoke(
                {"messages": [("user", user_input)], "user_id": user_id,
                 "needs_human": False, "is_last_step": False, "previous_node": None},
                config,
            )
            reply = state["messages"][-1].content
            sessions.append_turn(session_id, "assistant", reply)
            print(f"\nagent> {reply}")

    print("\nbye — transcript in ai.agent.sessions, trace in ai.agent_activity.logs")


if __name__ == "__main__":
    main()
