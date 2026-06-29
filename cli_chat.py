"""
Interactive CLI Chat Interface
================================
Run: python cli_chat.py
"""

import os
import sys
import time
import textwrap

# ── Optional color support ────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORS = True
except ImportError:
    COLORS = False
    class Fore:
        GREEN = CYAN = YELLOW = RED = MAGENTA = BLUE = WHITE = ""
    class Style:
        BRIGHT = RESET_ALL = DIM = ""

from matching_agent import MatchingAgent


# ── UI helpers ────────────────────────────────────────────────────────────────

WIDTH = 80

def banner():
    print(Fore.CYAN + Style.BRIGHT + "=" * WIDTH)
    print(" 🤖  AGENTIC PROFILE MATCHING SYSTEM  ".center(WIDTH))
    print("     LangGraph-powered AI Recruiter Agent     ".center(WIDTH))
    print("=" * WIDTH + Style.RESET_ALL)
    print()

def section(title: str):
    print()
    print(Fore.YELLOW + Style.BRIGHT + f"  ── {title} " + "─" * max(0, WIDTH - len(title) - 6) + Style.RESET_ALL)

def agent_say(message: str, prefix: str = "🤖 Agent"):
    wrapped = textwrap.fill(message, width=WIDTH - 10)
    print(Fore.GREEN + f"\n{prefix}:" + Style.RESET_ALL)
    for line in wrapped.split('\n'):
        print(f"  {line}")

def user_input(prompt: str = "You") -> str:
    print(Fore.BLUE + f"\n{prompt} > " + Style.RESET_ALL, end="")
    return input().strip()

def thinking(message: str = "Thinking"):
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    print(Fore.YELLOW + f"  {message} ", end="", flush=True)
    for frame in frames * 2:
        print(f"\r  {frame} {message}...", end="", flush=True)
        time.sleep(0.05)
    print(f"\r  ✓ Done{' ' * 20}" + Style.RESET_ALL)

def show_candidates_table(state: dict):
    """Print a ranked table of candidates."""
    candidates = state.get("all_candidates", [])
    if not candidates:
        return
    
    section("CANDIDATE RANKINGS")
    header = f"  {'#':<4} {'Name':<18} {'Score':<8} {'Exp':<6} {'Skills':<25} {'Decision'}"
    print(Fore.WHITE + Style.BRIGHT + header + Style.RESET_ALL)
    print("  " + "-" * (WIDTH - 2))
    
    for cand in sorted(candidates, key=lambda x: x["score"], reverse=True):
        score = cand["score"]
        decision_color = (
            Fore.GREEN if score >= 75 else
            Fore.CYAN if score >= 60 else
            Fore.YELLOW if score >= 45 else
            Fore.RED
        )
        decision = (
            "✅ Strong Hire" if score >= 75 else
            "👍 Hire" if score >= 60 else
            "⚠️  Borderline" if score >= 45 else
            "❌ No Hire"
        )
        skills_preview = ", ".join(cand["skills"][:4]) if cand["skills"] else "—"
        skills_preview = skills_preview[:24]
        exp = f"{cand['experience_years']:.0f}yr"
        rank = cand.get("rank", "?")
        
        print(
            f"  {str(rank):<4} "
            f"{cand['name']:<18} "
            + Fore.WHITE + Style.BRIGHT + f"{score:<8}" + Style.RESET_ALL +
            f" {exp:<6} "
            f"{skills_preview:<25} "
            + decision_color + decision + Style.RESET_ALL
        )


def show_requirements(state: dict):
    req = state.get("job_requirements", {})
    section("JOB REQUIREMENTS PARSED")
    print(f"  📋 Title: {req.get('title', 'N/A')}")
    print(f"  ⏱️  Experience: {req.get('min_experience', 0):.0f}+ years")
    print(f"\n  ✅ Must-Have:")
    for item in req.get("must_have", [])[:5]:
        print(f"     • {item}")
    print(f"\n  ⭐ Nice-to-Have:")
    for item in req.get("nice_to_have", [])[:4]:
        print(f"     • {item}")


def show_final_recommendations(state: dict):
    recs = state.get("final_recommendations", [])
    section("FINAL HIRE RECOMMENDATIONS")
    for i, rec in enumerate(recs, 1):
        color = Fore.GREEN if "HIRE" in rec["hire_decision"] else Fore.RED
        print(color + f"  {i}. {rec['name']:<20} {rec['hire_decision']:<15} Score: {rec['score']}/100" + Style.RESET_ALL)


def show_conversation_flow(state: dict):
    section("AGENT REASONING LOG")
    for entry in state.get("conversation_history", []):
        node = entry.get("node", "?")
        msg = entry.get("message", "")
        print(Fore.MAGENTA + f"  [{node}]" + Style.RESET_ALL + f" {msg}")


# ── Predefined Test Scenarios ─────────────────────────────────────────────────

TEST_SCENARIOS = [
    {
        "name": "Scenario 1: Find React + 3yr experience",
        "query": "Find me candidates with React and 3+ years experience"
    },
    {
        "name": "Scenario 2: Compare top 3 side by side",
        "query": "Compare the top 3 matches side by side"
    },
    {
        "name": "Scenario 3: Why did John rank higher?",
        "query": "Why did John Doe rank higher than Jane Smith?"
    },
    {
        "name": "Scenario 4: Generate interview questions for top candidate",
        "query": "Give me interview questions for John Doe"
    },
    {
        "name": "Scenario 5: Show top 2 candidates",
        "query": "Show me the top 2 candidates"
    },
]


# ── Main Chat Loop ─────────────────────────────────────────────────────────────

def main():
    banner()
    
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resume_dir = os.path.join(base_dir, "resumes")
    jd_dir = os.path.join(base_dir, "jd")
    report_dir = os.path.join(base_dir, "reports")
    
    # Check for resumes
    if not os.path.exists(resume_dir) or not os.listdir(resume_dir):
        print(Fore.RED + "  ⚠️  No resumes found in ./resumes/ directory." + Style.RESET_ALL)
        print("  Add .txt resume files to the resumes/ folder and restart.")
        sys.exit(1)
    
    # Load JD
    section("STEP 1 — LOAD JOB DESCRIPTION")
    jd_files = [f for f in os.listdir(jd_dir)] if os.path.exists(jd_dir) else []
    
    if jd_files:
        print(f"  Found JD files: {jd_files}")
        jd_path = os.path.join(jd_dir, jd_files[0])
        with open(jd_path, 'r') as f:
            jd_text = f.read()
        print(f"  Loaded: {jd_files[0]}")
    else:
        print("  No JD file found. Paste your job description below (type END on a new line to finish):")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        jd_text = '\n'.join(lines)
    
    # Initialize and run agent
    section("STEP 2 — RUNNING SCREENING PIPELINE")
    agent = MatchingAgent(resume_dir=resume_dir, report_dir=report_dir)
    
    thinking("Parsing JD and indexing resumes")
    state = agent.run(jd_text)
    
    show_requirements(state)
    show_candidates_table(state)
    show_final_recommendations(state)
    
    report_path = state.get("report_path", "")
    if report_path:
        print(Fore.GREEN + f"\n  📄 Full report saved: {report_path}" + Style.RESET_ALL)
    
    show_conversation_flow(state)
    
    # ── Interactive Chat ──────────────────────────────────────────────────────
    section("STEP 3 — INTERACTIVE CHAT")
    agent_say(
        "Screening complete! You can now ask me anything about the candidates.\n\n"
        "Examples:\n"
        "  • 'Find candidates with React and 3+ years experience'\n"
        "  • 'Compare the top 3 matches side by side'\n"
        "  • 'Why did John rank higher than Jane?'\n"
        "  • 'Give me interview questions for <name>'\n"
        "  • 'Add TypeScript as a requirement' (re-ranks)\n"
        "  • 'Run test scenarios' (auto-run 5 test flows)\n"
        "  • 'quit' to exit"
    )
    
    while True:
        try:
            user_query = user_input()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_query:
            continue
        
        if user_query.lower() in ("quit", "exit", "q"):
            print(Fore.YELLOW + "\n  Goodbye! Report saved at: " + (report_path or "reports/") + Style.RESET_ALL)
            break
        
        # Run test scenarios
        if "test scenarios" in user_query.lower() or "run tests" in user_query.lower():
            section("RUNNING 5 TEST SCENARIOS")
            for scenario in TEST_SCENARIOS:
                print(Fore.MAGENTA + f"\n  📌 {scenario['name']}" + Style.RESET_ALL)
                print(Fore.BLUE + f"  Query: {scenario['query']}" + Style.RESET_ALL)
                response = agent.query(scenario["query"])
                agent_say(response, prefix="  🤖")
                print()
            continue
        
        # Feedback/re-ranking commands
        feedback_keywords = ["add ", "require ", "must have", "need ", "more experience", "senior"]
        if any(kw in user_query.lower() for kw in feedback_keywords):
            thinking("Re-ranking candidates with new criteria")
            try:
                state = agent.apply_feedback(user_query)
                show_candidates_table(state)
                agent_say("Rankings updated based on your feedback!")
            except Exception as e:
                agent_say(f"Could not apply feedback: {e}")
        else:
            # Natural language query
            response = agent.query(user_query)
            agent_say(response)


if __name__ == "__main__":
    main()
