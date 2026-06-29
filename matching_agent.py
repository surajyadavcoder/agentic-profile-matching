"""
Agentic Profile Matching System
================================
LangGraph-based multi-round screening agent for candidate matching.

Architecture:
START → Parse JD → Extract Requirements → Search Resumes → 
Rank Candidates → Generate Report → Human Feedback Loop → END
"""

import os
import re
import json
import math
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ── LangGraph imports ────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END, START

# ── Type definitions ─────────────────────────────────────────────────────────

@dataclass
class Candidate:
    id: str
    name: str
    email: str
    file_path: str
    raw_text: str = ""
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: str = ""
    summary: str = ""
    score: float = 0.0
    match_reasons: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    rank: int = 0

@dataclass
class JobRequirements:
    title: str = ""
    must_have: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    min_experience: float = 0.0
    max_experience: float = 99.0
    raw_jd: str = ""

class AgentState(TypedDict):
    """Central state tracked across the entire agent graph."""
    # Input
    jd_text: str
    resume_dir: str
    
    # Parsed data
    job_requirements: Optional[Dict]
    all_candidates: List[Dict]
    
    # Screening rounds
    round1_shortlist: List[str]   # top 10 candidate IDs
    round2_shortlist: List[str]   # top 5
    final_recommendations: List[Dict]
    
    # Conversation
    conversation_history: List[Dict]
    current_query: str
    agent_response: str
    
    # Control
    current_round: int
    screening_complete: bool
    human_feedback: Optional[str]
    report_path: Optional[str]
    
    # Errors / debug
    errors: List[str]


# ── File System Tools (Milestone 1) ──────────────────────────────────────────

def list_resumes(directory: str) -> List[str]:
    """List all resume files in a directory."""
    if not os.path.exists(directory):
        return []
    extensions = {'.txt', '.pdf', '.docx', '.md'}
    files = []
    for f in os.listdir(directory):
        if any(f.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(directory, f))
    return sorted(files)


def read_resume(file_path: str) -> str:
    """Read resume content from file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"[Error reading file: {e}]"


def save_report(report_content: str, report_dir: str = "reports") -> str:
    """Save matching report to file system."""
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(report_dir, f"matching_report_{timestamp}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    return path


# ── RAG Search Tool (Milestone 2 — keyword-based fallback) ───────────────────

class SimpleVectorStore:
    """
    Lightweight keyword-based search (RAG substitute when embeddings unavailable).
    In production, swap with FAISS + sentence-transformers.
    """
    def __init__(self):
        self.documents: List[Dict] = []

    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        self.documents.append({
            "id": doc_id,
            "text": text,
            "tokens": tokens,
            "metadata": metadata or {}
        })

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = set(re.findall(r'\b\w+\b', query.lower()))
        scored = []
        for doc in self.documents:
            overlap = query_tokens & doc["tokens"]
            # TF-IDF inspired: reward rare query terms
            score = sum(
                1 + math.log(1 + len(overlap)) 
                for _ in overlap
            ) / (len(query_tokens) + 1)
            scored.append({"score": score, **doc})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


# ── Agent Tools ───────────────────────────────────────────────────────────────

def extract_requirements(jd: str) -> JobRequirements:
    """
    Parse job description into structured must-have vs nice-to-have requirements.
    Uses rule-based NLP (no LLM needed for deterministic parsing).
    """
    req = JobRequirements(raw_jd=jd)

    # Extract title
    title_match = re.search(r'Position:\s*(.+)', jd, re.IGNORECASE)
    if title_match:
        req.title = title_match.group(1).strip()

    # Extract experience range
    exp_patterns = [
        r'(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?',
        r'(\d+)\+\s*years?',
        r'minimum\s+(\d+)\s*years?',
    ]
    for pattern in exp_patterns:
        m = re.search(pattern, jd, re.IGNORECASE)
        if m:
            groups = m.groups()
            req.min_experience = float(groups[0])
            req.max_experience = float(groups[1]) if len(groups) > 1 and groups[1] else 99.0
            break

    # Extract must-have skills
    must_have_section = re.search(
        r'MUST.HAVE.*?(?=NICE.TO.HAVE|RESPONSIBILITIES|$)',
        jd, re.DOTALL | re.IGNORECASE
    )
    if must_have_section:
        text = must_have_section.group(0)
        # Extract bullet items
        items = re.findall(r'[-•*]\s*(.+)', text)
        req.must_have = [item.strip() for item in items]

    # Extract nice-to-have skills
    nice_section = re.search(
        r'NICE.TO.HAVE.*?(?=RESPONSIBILITIES|COMPENSATION|$)',
        jd, re.DOTALL | re.IGNORECASE
    )
    if nice_section:
        text = nice_section.group(0)
        items = re.findall(r'[-•*]\s*(.+)', text)
        req.nice_to_have = [item.strip() for item in items]

    # Fallback: extract common tech keywords
    if not req.must_have:
        tech_keywords = re.findall(
            r'\b(React|TypeScript|JavaScript|Node\.js|Python|AWS|Docker|'
            r'Kubernetes|GraphQL|MongoDB|PostgreSQL|Redis|Go|Java|'
            r'Next\.js|Vue\.js|Angular|Spring|FastAPI|Django)\b',
            jd, re.IGNORECASE
        )
        req.must_have = list(dict.fromkeys(tech_keywords))  # deduplicate

    return req


def parse_candidate(file_path: str) -> Candidate:
    """Parse a resume file into a structured Candidate object."""
    text = read_resume(file_path)
    candidate_id = os.path.splitext(os.path.basename(file_path))[0]

    # Extract name
    name = "Unknown"
    name_match = re.search(r'Name:\s*(.+)', text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
    elif text.split('\n'):
        # First non-empty line often is the name
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line.split()) <= 4:
                name = line
                break

    # Extract email
    email = ""
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
    if email_match:
        email = email_match.group(0)

    # Extract skills
    skills_section = re.search(
        r'SKILLS.*?(?=EXPERIENCE|EDUCATION|PROJECTS|$)',
        text, re.DOTALL | re.IGNORECASE
    )
    skills_text = skills_section.group(0) if skills_section else text
    
    tech_pattern = re.compile(
        r'\b(React|TypeScript|JavaScript|Node\.js|Python|AWS|Docker|'
        r'Kubernetes|GraphQL|MongoDB|PostgreSQL|Redis|Go|Java|Golang|'
        r'Next\.js|Vue\.js|Angular|Spring|FastAPI|Django|Express|'
        r'Kafka|Redis|Cassandra|MySQL|Git|CI/CD|Jest|Webpack|Vite|'
        r'Socket\.io|WebSocket|OAuth|JWT|REST|gRPC|Terraform|'
        r'Prometheus|Grafana|Jaeger|Redux|Zustand|SASS|Tailwind|'
        r'Firebase|Heroku|Vercel|GCP|Azure)\b',
        re.IGNORECASE
    )
    skills = list(dict.fromkeys(
        m.group(0) for m in tech_pattern.finditer(skills_text)
    ))

    # Extract years of experience
    total_exp = 0.0
    exp_patterns = [
        r'(\d+(?:\.\d+)?)\s*years?\b',
        r'\((\d+)\s*years?\)',
    ]
    for pattern in exp_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            val = float(m.group(1))
            if val < 30:  # sanity check
                total_exp = max(total_exp, val)

    # Extract education
    edu_match = re.search(
        r'(B\.Tech|B\.E|M\.Tech|MBA|MCA|BCA|PhD|B\.Sc)[^\n]*',
        text, re.IGNORECASE
    )
    education = edu_match.group(0).strip() if edu_match else ""

    # Extract summary
    summary_match = re.search(
        r'SUMMARY\n(.+?)(?=\n[A-Z]{3,}|\Z)',
        text, re.DOTALL | re.IGNORECASE
    )
    summary = summary_match.group(1).strip()[:300] if summary_match else ""

    return Candidate(
        id=candidate_id,
        name=name,
        email=email,
        file_path=file_path,
        raw_text=text,
        skills=skills,
        experience_years=total_exp,
        education=education,
        summary=summary,
    )


def score_candidate(candidate: Candidate, requirements: JobRequirements) -> Candidate:
    """
    Score a candidate against job requirements.
    Returns candidate with score (0-100), match_reasons, and gaps populated.
    """
    score = 0.0
    match_reasons = []
    gaps = []

    # 1. Must-have skills (50 points max)
    must_have_text = ' '.join(requirements.must_have).lower()
    candidate_text = (candidate.raw_text + ' ' + ' '.join(candidate.skills)).lower()
    
    must_have_keywords = re.findall(r'\b\w+\b', must_have_text)
    must_have_keywords = [w for w in must_have_keywords if len(w) > 2]
    
    matched_must = []
    missed_must = []
    for kw in set(must_have_keywords):
        if re.search(r'\b' + re.escape(kw) + r'\b', candidate_text):
            matched_must.append(kw)
        else:
            missed_must.append(kw)
    
    if must_have_keywords:
        must_score = (len(matched_must) / len(set(must_have_keywords))) * 50
        score += must_score
        if matched_must:
            match_reasons.append(f"Matches must-have: {', '.join(matched_must[:5])}")
        if missed_must:
            gaps.append(f"Missing must-have: {', '.join(missed_must[:5])}")

    # 2. Nice-to-have skills (20 points max)
    nice_text = ' '.join(requirements.nice_to_have).lower()
    nice_keywords = re.findall(r'\b\w+\b', nice_text)
    nice_keywords = [w for w in nice_keywords if len(w) > 2]
    
    matched_nice = []
    for kw in set(nice_keywords):
        if re.search(r'\b' + re.escape(kw) + r'\b', candidate_text):
            matched_nice.append(kw)
    
    if nice_keywords:
        nice_score = (len(matched_nice) / len(set(nice_keywords))) * 20
        score += nice_score
        if matched_nice:
            match_reasons.append(f"Bonus: {', '.join(matched_nice[:3])}")

    # 3. Experience (20 points max)
    exp = candidate.experience_years
    if requirements.min_experience <= exp <= requirements.max_experience:
        score += 20
        match_reasons.append(f"{exp:.0f} years experience (required: {requirements.min_experience:.0f}+)")
    elif exp > 0:
        partial = min(exp / max(requirements.min_experience, 1), 1.0) * 15
        score += partial
        if exp < requirements.min_experience:
            gaps.append(f"Experience {exp:.0f}yr < required {requirements.min_experience:.0f}yr")

    # 4. Education (10 points max)
    elite_colleges = ['iit', 'nit', 'bits', 'iiit']
    if any(c in candidate.education.lower() for c in elite_colleges):
        score += 10
        match_reasons.append(f"Top-tier education: {candidate.education}")
    elif candidate.education:
        score += 5

    candidate.score = round(min(score, 100), 1)
    candidate.match_reasons = match_reasons
    candidate.gaps = gaps
    return candidate


def compare_candidates(candidates: List[Candidate], criteria: List[str] = None) -> str:
    """Side-by-side comparison of candidates."""
    if not candidates:
        return "No candidates to compare."
    
    criteria = criteria or ["score", "experience_years", "skills", "education"]
    lines = [f"{'Candidate':<20} " + " | ".join(f"{c:<15}" for c in criteria)]
    lines.append("-" * (20 + len(criteria) * 18))
    
    for cand in sorted(candidates, key=lambda x: x.score, reverse=True):
        row = [f"{cand.name:<20}"]
        for criterion in criteria:
            if criterion == "score":
                row.append(f"{cand.score:<15}")
            elif criterion == "experience_years":
                row.append(f"{cand.experience_years:.0f}yr{'':<10}")
            elif criterion == "skills":
                row.append(f"{', '.join(cand.skills[:3]):<15}")
            elif criterion == "education":
                row.append(f"{cand.education[:15]:<15}")
            else:
                row.append(f"{'N/A':<15}")
        lines.append(" | ".join(row))
    
    return '\n'.join(lines)


def generate_interview_questions(candidate: Candidate, requirements: JobRequirements) -> List[str]:
    """Generate tailored screening questions for a candidate."""
    questions = []
    
    # Based on experience level
    if candidate.experience_years < 2:
        questions.append("Walk me through a React project you built from scratch. What challenges did you face?")
        questions.append("How do you approach learning a new technology or framework?")
    else:
        questions.append(f"With {candidate.experience_years:.0f} years of experience, describe the most complex system you've architected.")
        questions.append("How have you handled performance bottlenecks in production?")
    
    # Based on matched skills
    if 'TypeScript' in candidate.skills:
        questions.append("How do you leverage TypeScript's type system to prevent runtime errors? Give a specific example.")
    if 'React' in candidate.skills:
        questions.append("Explain your approach to state management in large React applications.")
    if 'Node.js' in candidate.skills or 'node' in ' '.join(candidate.skills).lower():
        questions.append("How do you handle async error propagation in Node.js microservices?")
    
    # Based on gaps
    for gap in candidate.gaps[:2]:
        gap_skill = gap.replace("Missing must-have:", "").strip().split(',')[0]
        questions.append(f"The role requires {gap_skill}. How quickly could you get up to speed, and what's your learning approach?")
    
    # Behavioral questions
    questions.append("Describe a time you had to mentor a junior developer. What was your approach?")
    questions.append("Tell me about a situation where you had to push back on a product requirement for technical reasons.")
    
    return questions[:6]  # Return top 6 questions


def generate_match_report(
    candidates: List[Candidate],
    requirements: JobRequirements,
    screening_round: int = 3
) -> str:
    """Generate a detailed Markdown match report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines = [
        f"# Candidate Matching Report",
        f"**Generated:** {timestamp}  ",
        f"**Position:** {requirements.title}  ",
        f"**Total Candidates Evaluated:** {len(candidates)}  ",
        f"**Screening Round:** {screening_round}",
        "",
        "---",
        "",
        "## Job Requirements Summary",
        "",
        "### Must-Have",
    ]
    for req in requirements.must_have:
        lines.append(f"- {req}")
    
    lines += ["", "### Nice-to-Have"]
    for req in requirements.nice_to_have:
        lines.append(f"- {req}")
    
    lines += [
        "",
        "---",
        "",
        "## Candidate Rankings",
        "",
    ]
    
    sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
    
    for i, cand in enumerate(sorted_candidates, 1):
        recommendation = "✅ STRONG HIRE" if cand.score >= 75 else \
                         "👍 HIRE" if cand.score >= 60 else \
                         "⚠️ BORDERLINE" if cand.score >= 45 else \
                         "❌ NO HIRE"
        
        lines += [
            f"### #{i} — {cand.name}  ({recommendation})",
            f"**Match Score:** {cand.score}/100  ",
            f"**Experience:** {cand.experience_years:.0f} years  ",
            f"**Email:** {cand.email}  ",
            "",
            "**Strengths:**",
        ]
        for reason in cand.match_reasons:
            lines.append(f"- {reason}")
        
        if cand.gaps:
            lines.append("")
            lines.append("**Gaps:**")
            for gap in cand.gaps:
                lines.append(f"- {gap}")
        
        # Improvement suggestions for borderline
        if 45 <= cand.score < 60:
            lines += [
                "",
                "**Improvement Suggestions (Borderline Candidate):**",
                f"- Strengthen TypeScript skills with hands-on projects",
                f"- Gain cloud platform experience (AWS/GCP certifications)",
                f"- Contribute to open source React projects",
            ]
        
        # Interview questions
        qs = generate_interview_questions(cand, requirements)
        lines += [
            "",
            "**Recommended Screening Questions:**",
        ]
        for j, q in enumerate(qs[:3], 1):
            lines.append(f"{j}. {q}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return '\n'.join(lines)


# ── LangGraph Node Functions ──────────────────────────────────────────────────

def node_parse_jd(state: AgentState) -> AgentState:
    """Node 1: Parse the job description."""
    jd_text = state.get("jd_text", "")
    if not jd_text:
        state["errors"] = state.get("errors", []) + ["No JD provided"]
        state["job_requirements"] = asdict(JobRequirements())
        return state
    
    req = extract_requirements(jd_text)
    state["job_requirements"] = asdict(req)
    state["conversation_history"] = state.get("conversation_history", []) + [{
        "role": "agent",
        "node": "parse_jd",
        "message": f"Parsed JD for: '{req.title}'. Found {len(req.must_have)} must-have and {len(req.nice_to_have)} nice-to-have requirements."
    }]
    return state


def node_search_resumes(state: AgentState) -> AgentState:
    """Node 2: Load and index all resumes."""
    resume_dir = state.get("resume_dir", "resumes")
    files = list_resumes(resume_dir)
    
    vector_store = SimpleVectorStore()
    candidates = []
    
    for file_path in files:
        cand = parse_candidate(file_path)
        vector_store.add_document(cand.id, cand.raw_text, {"name": cand.name})
        candidates.append(asdict(cand))
    
    state["all_candidates"] = candidates
    state["conversation_history"] = state.get("conversation_history", []) + [{
        "role": "agent",
        "node": "search_resumes",
        "message": f"Indexed {len(candidates)} resumes: {[c['name'] for c in candidates]}"
    }]
    return state


def node_rank_candidates(state: AgentState) -> AgentState:
    """Node 3: Score and rank all candidates."""
    req_dict = state.get("job_requirements", {})
    requirements = JobRequirements(**{
        k: v for k, v in req_dict.items() 
        if k in JobRequirements.__dataclass_fields__
    })
    
    candidates = []
    for cand_dict in state.get("all_candidates", []):
        cand = Candidate(**{k: v for k, v in cand_dict.items() if k in Candidate.__dataclass_fields__})
        cand = score_candidate(cand, requirements)
        candidates.append(cand)
    
    # Sort by score
    candidates.sort(key=lambda x: x.score, reverse=True)
    for i, c in enumerate(candidates, 1):
        c.rank = i
    
    # Round 1: top 10
    round1 = [c.id for c in candidates[:10]]
    # Round 2: top 5 from round 1
    round2 = [c.id for c in candidates[:5]]
    
    state["all_candidates"] = [asdict(c) for c in candidates]
    state["round1_shortlist"] = round1
    state["round2_shortlist"] = round2
    state["current_round"] = 2
    state["conversation_history"] = state.get("conversation_history", []) + [{
        "role": "agent",
        "node": "rank_candidates",
        "message": f"Round 1 shortlist (top {len(round1)}): {[c.name for c in candidates[:len(round1)]]}"
    }]
    return state


def node_generate_report(state: AgentState) -> AgentState:
    """Node 4: Generate detailed match report."""
    req_dict = state.get("job_requirements", {})
    requirements = JobRequirements(**{
        k: v for k, v in req_dict.items() 
        if k in JobRequirements.__dataclass_fields__
    })
    
    # Get final candidates (round 2 shortlist)
    round2_ids = set(state.get("round2_shortlist", []))
    final_candidates = [
        Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
        for c in state.get("all_candidates", [])
        if c["id"] in round2_ids
    ]
    
    # Build final recommendations
    final_recs = []
    for cand in sorted(final_candidates, key=lambda x: x.score, reverse=True):
        hire_decision = "STRONG HIRE" if cand.score >= 75 else \
                        "HIRE" if cand.score >= 60 else \
                        "BORDERLINE" if cand.score >= 45 else "NO HIRE"
        final_recs.append({
            "id": cand.id,
            "name": cand.name,
            "score": cand.score,
            "hire_decision": hire_decision,
            "interview_questions": generate_interview_questions(cand, requirements)
        })
    
    state["final_recommendations"] = final_recs
    
    # Generate and save report
    all_candidates = [
        Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
        for c in state.get("all_candidates", [])
    ]
    report_md = generate_match_report(all_candidates, requirements)
    report_path = save_report(report_md, "reports")
    state["report_path"] = report_path
    
    state["screening_complete"] = True
    state["agent_response"] = f"Screening complete! Report saved to {report_path}"
    state["conversation_history"] = state.get("conversation_history", []) + [{
        "role": "agent",
        "node": "generate_report",
        "message": f"Final recommendations: {[(r['name'], r['hire_decision'], r['score']) for r in final_recs]}"
    }]
    return state


def node_human_feedback(state: AgentState) -> AgentState:
    """Node 5: Process human feedback and re-rank if needed."""
    feedback = state.get("human_feedback", "")
    if not feedback:
        return state
    
    feedback_lower = feedback.lower()
    
    # Detect re-ranking requests
    if any(word in feedback_lower for word in ["more experience", "senior", "years"]):
        exp_match = re.search(r'(\d+)\+?\s*years?', feedback, re.IGNORECASE)
        if exp_match:
            min_exp = float(exp_match.group(1))
            req_dict = state.get("job_requirements", {})
            req_dict["min_experience"] = min_exp
            state["job_requirements"] = req_dict
            state["conversation_history"] = state.get("conversation_history", []) + [{
                "role": "agent",
                "node": "human_feedback",
                "message": f"Adjusted min experience to {min_exp} years. Re-ranking..."
            }]
            # Trigger re-ranking
            state = node_rank_candidates(state)
            state = node_generate_report(state)
    
    # Add new required skill
    elif any(word in feedback_lower for word in ["add", "require", "must have", "need"]):
        new_skill_match = re.search(r'(?:add|require|need)\s+(\w+)', feedback, re.IGNORECASE)
        if new_skill_match:
            new_skill = new_skill_match.group(1)
            req_dict = state.get("job_requirements", {})
            req_dict["must_have"] = req_dict.get("must_have", []) + [new_skill]
            state["job_requirements"] = req_dict
            state["conversation_history"] = state.get("conversation_history", []) + [{
                "role": "agent",
                "node": "human_feedback",
                "message": f"Added '{new_skill}' to must-have requirements. Re-ranking..."
            }]
            state = node_rank_candidates(state)
            state = node_generate_report(state)
    
    return state


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_matching_graph() -> StateGraph:
    """Construct the LangGraph state machine."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("parse_jd", node_parse_jd)
    graph.add_node("search_resumes", node_search_resumes)
    graph.add_node("rank_candidates", node_rank_candidates)
    graph.add_node("generate_report", node_generate_report)
    graph.add_node("human_feedback", node_human_feedback)
    
    # Add edges
    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "human_feedback")
    graph.add_edge("human_feedback", END)
    
    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

class MatchingAgent:
    """High-level wrapper around the LangGraph matching agent."""
    
    def __init__(self, resume_dir: str = "resumes", report_dir: str = "reports"):
        self.resume_dir = resume_dir
        self.report_dir = report_dir
        self.graph = build_matching_graph()
        self.state: Optional[AgentState] = None
    
    def run(self, jd_text: str) -> AgentState:
        """Run the full screening pipeline."""
        initial_state: AgentState = {
            "jd_text": jd_text,
            "resume_dir": self.resume_dir,
            "job_requirements": None,
            "all_candidates": [],
            "round1_shortlist": [],
            "round2_shortlist": [],
            "final_recommendations": [],
            "conversation_history": [],
            "current_query": "",
            "agent_response": "",
            "current_round": 1,
            "screening_complete": False,
            "human_feedback": None,
            "report_path": None,
            "errors": [],
        }
        self.state = self.graph.invoke(initial_state)
        return self.state
    
    def apply_feedback(self, feedback: str) -> AgentState:
        """Apply human feedback and re-run from feedback node."""
        if not self.state:
            raise RuntimeError("Run the agent first with .run(jd_text)")
        self.state["human_feedback"] = feedback
        self.state = self.graph.invoke(self.state)
        return self.state
    
    def query(self, question: str) -> str:
        """Answer natural language questions about the results."""
        if not self.state:
            return "Please run the agent first."
        
        q = question.lower()
        candidates = [
            Candidate(**{k: v for k, v in c.items() if k in Candidate.__dataclass_fields__})
            for c in self.state.get("all_candidates", [])
        ]
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # "Find candidates with X and Y years experience"
        exp_match = re.search(r'(\d+)\+?\s*years?', question, re.IGNORECASE)
        skill_match = re.search(
            r'(?:with|having|skilled in)\s+([\w.,\s]+?)(?:\s+and|\s+experience|\s+years|$)',
            question, re.IGNORECASE
        )
        
        if "compare" in q or "side by side" in q:
            num_match = re.search(r'top\s*(\d+)', q)
            n = int(num_match.group(1)) if num_match else 3
            to_compare = candidates[:n]
            return compare_candidates(to_compare)
        
        elif "why" in q and ("rank" in q or "higher" in q or "better" in q):
            # "Why did X rank higher than Y?"
            names = [c.name.lower() for c in candidates]
            for i, cand in enumerate(candidates):
                if cand.name.lower() in q:
                    reasons = '\n'.join(f"  - {r}" for r in cand.match_reasons)
                    gaps = '\n'.join(f"  - {g}" for g in cand.gaps) if cand.gaps else "  None"
                    return (
                        f"{cand.name} (Rank #{cand.rank}, Score: {cand.score}/100)\n"
                        f"Strengths:\n{reasons}\n"
                        f"Gaps:\n{gaps}"
                    )
            return "Could not identify the candidate. Please mention their name."
        
        elif "interview questions" in q or "screening questions" in q:
            for cand in candidates:
                if cand.name.lower() in q:
                    req = JobRequirements(**{
                        k: v for k, v in self.state.get("job_requirements", {}).items()
                        if k in JobRequirements.__dataclass_fields__
                    })
                    qs = generate_interview_questions(cand, req)
                    return f"Interview questions for {cand.name}:\n" + '\n'.join(f"{i}. {q}" for i, q in enumerate(qs, 1))
        
        elif "top" in q:
            num_match = re.search(r'top\s*(\d+)', q)
            n = int(num_match.group(1)) if num_match else 3
            results = []
            for i, c in enumerate(candidates[:n], 1):
                results.append(f"{i}. {c.name} — Score: {c.score}/100, {c.experience_years:.0f}yr exp")
            return '\n'.join(results)
        
        elif exp_match or skill_match:
            # Filter by criteria
            filtered = candidates
            if exp_match:
                min_exp = float(exp_match.group(1))
                filtered = [c for c in filtered if c.experience_years >= min_exp]
            if skill_match:
                skills_query = skill_match.group(1).strip().lower()
                filtered = [
                    c for c in filtered
                    if any(s.lower() in skills_query or skills_query in s.lower() 
                           for s in c.skills)
                ]
            if filtered:
                return "Matching candidates:\n" + '\n'.join(
                    f"- {c.name} ({c.experience_years:.0f}yr, score: {c.score})"
                    for c in filtered
                )
            else:
                return "No candidates match those criteria."
        
        else:
            # Default: show summary
            top3 = candidates[:3]
            return "Top candidates:\n" + '\n'.join(
                f"{i}. {c.name} — {c.score}/100" for i, c in enumerate(top3, 1)
            )
