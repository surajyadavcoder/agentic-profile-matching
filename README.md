#  Agentic Profile Matching System

> LangGraph-based multi-round AI recruiter agent — Airtribe Backend AI Assignment

---

##  Overview

This project implements a **fully autonomous candidate screening agent** using **LangGraph** that:

- Parses job descriptions into structured must-have / nice-to-have requirements
- Indexes resumes from the file system (Milestone 1 tools)
- Performs keyword-based semantic search (Milestone 2 RAG tool)
- Runs **multi-round screening** (100 → 10 → 5 → final recommendations)
- Generates detailed match reports with explainability
- Supports **natural language conversational queries**
- Allows **iterative re-ranking** via human feedback

---

##  Architecture

### LangGraph State Machine

```
START
  │
  ▼
┌─────────────┐
│  Parse JD   │  ← extract_requirements() — must-have vs nice-to-have
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Search Resumes  │  ← list_resumes() + read_resume() (Milestone 1)
│  (Index all CVs) │    SimpleVectorStore (Milestone 2 RAG)
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│   Rank Candidates    │  ← score_candidate() — 4-factor scoring
│  Round 1: top 10     │
│  Round 2: top 5      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│    Generate Report       │  ← generate_match_report() + save_report()
│  Final: hire/no-hire     │    generate_interview_questions()
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────┐
│   Human Feedback     │  ← adjust requirements, trigger re-rank
│        Loop          │
└───────────┬──────────┘
            │
            ▼
           END
```

### Agent State (`AgentState` TypedDict)

| Field | Type | Purpose |
|-------|------|---------|
| `jd_text` | str | Raw job description input |
| `job_requirements` | Dict | Parsed must-have / nice-to-have |
| `all_candidates` | List[Dict] | All parsed + scored candidates |
| `round1_shortlist` | List[str] | Top 10 candidate IDs |
| `round2_shortlist` | List[str] | Top 5 candidate IDs |
| `final_recommendations` | List[Dict] | Hire decisions |
| `conversation_history` | List[Dict] | Agent reasoning log |
| `human_feedback` | Optional[str] | Re-ranking trigger |
| `report_path` | Optional[str] | Saved report file path |

---

##  Tools

### Milestone 1 — File System Tools
| Tool | Description |
|------|-------------|
| `list_resumes(dir)` | Scans directory for resume files (.txt, .pdf, .docx) |
| `read_resume(path)` | Reads resume file content |
| `save_report(content, dir)` | Saves Markdown report to disk |

### Milestone 2 — RAG Search Tool
| Tool | Description |
|------|-------------|
| `SimpleVectorStore` | Keyword-based TF-IDF search over indexed resumes |
| `.add_document(id, text)` | Index a resume |
| `.search(query, top_k)` | Retrieve most relevant resumes |

>  In production, replace `SimpleVectorStore` with **FAISS + sentence-transformers** for dense vector similarity search.

### Agent-Specific Tools
| Tool | Description |
|------|-------------|
| `extract_requirements(jd)` | Parse must-have vs nice-to-have from JD text |
| `compare_candidates(candidates)` | Side-by-side comparison table |
| `generate_interview_questions(candidate)` | Tailored screening questions |
| `score_candidate(candidate, requirements)` | 4-factor scoring (0-100) |
| `generate_match_report(candidates, req)` | Full Markdown report with explainability |

---

##  Scoring Algorithm

Candidates are scored on a **100-point scale**:

| Factor | Max Points | Description |
|--------|-----------|-------------|
| Must-Have Skills | 50 | % of must-have keywords present in resume |
| Nice-to-Have Skills | 20 | % of nice-to-have keywords matched |
| Experience | 20 | Years within required range |
| Education | 10 | Elite college bonus (IIT/NIT/BITS) |

### Hire Decision Thresholds
-  **Strong Hire** — Score ≥ 75
-  **Hire** — Score 60–74
-  **Borderline** — Score 45–59 (with improvement suggestions)
-  **No Hire** — Score < 45

---

##  Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add resumes
Place `.txt` (or `.pdf`) resume files in the `resumes/` folder:
```
resumes/
  john_doe.txt
  jane_smith.txt
  rahul_sharma.txt
  ...
```

### 3. Add job description
Place a `.txt` JD file in the `jd/` folder:
```
jd/
  senior_frontend_dev.txt
```

### 4. Run CLI chat
```bash
python cli_chat.py
```

---

##  Conversational Interface — Example Queries

```
 Agent > Screening complete! You can now ask me anything.

You > Find me candidates with React and 3+ years experience
 → Matching candidates: John Doe (5yr, score: 82.5), Priya Patel (3yr, score: 71.0)

You > Compare the top 3 matches side by side
 → [Side-by-side comparison table]

You > Why did John rank higher than Jane?
 → John Doe (Rank #1, Score: 82.5/100)
      Strengths: Matches must-have: react, typescript, node.js...
      Gaps: None significant

You > Give me interview questions for Rahul Sharma
 → 1. Describe the most complex system you've architected...
      2. How do you handle distributed system failures?
      ...

You > Add Kubernetes as a requirement
→ Re-ranking with Kubernetes as must-have... [updated table]

You > Run test scenarios
 → [Runs all 5 predefined test flows]
```

---

## Project Structure

```
agentic_profile_matching/
├── matching_agent.py      ← Core LangGraph agent (Part A)
├── cli_chat.py            ← Interactive CLI interface (Part B)
├── requirements.txt
├── README.md
├── resumes/               ← Input: candidate resume files
│   ├── john_doe.txt
│   ├── jane_smith.txt
│   ├── rahul_sharma.txt
│   ├── priya_patel.txt
│   └── amit_kumar.txt
├── jd/                    ← Input: job description files
│   └── senior_frontend_dev.txt
└── reports/               ← Output: generated match reports
    └── matching_report_YYYYMMDD_HHMMSS.md
```

---

##  Test Scenarios (Part B)

| # | Query | Tests |
|---|-------|-------|
| 1 | `Find candidates with React and 3+ years` | Skill + experience filter |
| 2 | `Compare the top 3 matches side by side` | Comparison tool |
| 3 | `Why did John rank higher than Jane?` | Explainability |
| 4 | `Give me interview questions for John Doe` | Question generation tool |
| 5 | `Show me the top 2 candidates` | Ranking query |
| 6 | `Add Kubernetes as a requirement` | Re-ranking via feedback |

---

##  Multi-Round Screening (Part C)

| Round | Pool | Method |
|-------|------|--------|
| Round 1 | All candidates | Score-based filter → Top 10 |
| Round 2 | Top 10 | Deep analysis → Top 5 |
| Round 3 | Top 5 | Final hire/no-hire + interview questions |

---

##  Production Upgrades

To upgrade this demo to production-grade:

1. **Real LLM** — Replace rule-based parsing with `ChatOpenAI` or `Claude` via LangChain
2. **Dense RAG** — Replace `SimpleVectorStore` with FAISS + `sentence-transformers`
3. **PDF parsing** — Add `pdfplumber` or `pymupdf` to `read_resume()`
4. **Web UI** — Replace CLI with Streamlit (`streamlit run app.py`)
5. **Async** — Use `graph.ainvoke()` for concurrent resume processing

---

##  Author

**Suraj Yadav** | GitHub: https://github.com/surajyadavcoder Email: Surajyadavx.in@gmail.com