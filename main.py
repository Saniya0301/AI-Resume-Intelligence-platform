import streamlit as st
import math
import re
import os
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# GEMINI AI
# =========================

gemini_client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


st.set_page_config(
    page_title="AI Resume Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# EDITORIAL AI — PREMIUM UI
# =========================
st.markdown("""
<style>
    /* Main canvas */
    .stApp {
        background: #F7F5F2;
        color: #202124;
    }

    /* Remove excess top spacing */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #202124 !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown {
        color: #5F6068;
    }

    /* Brand header */
    .brand-wrap {
        padding: 0.25rem 0 1.8rem 0;
    }

    .brand-kicker {
        color: #8B7BB5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .brand-title {
        color: #202124;
        font-size: 2.7rem;
        font-weight: 750;
        line-height: 1.05;
        margin: 0;
    }

    .brand-title .star {
        color: #8B7BB5;
        margin-right: 0.35rem;
    }

    .brand-tagline {
        color: #77747B;
        font-size: 1.05rem;
        margin-top: 0.7rem;
    }

    /* Section headings */
    .section-label {
        color: #8B7BB5;
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 1.3rem 0 0.55rem;
    }

    /* Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E7E3DE;
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 8px 24px rgba(50, 45, 60, 0.055);
        min-height: 120px;
    }

    .metric-label {
        color: #77747B;
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: #29272D;
        font-size: 2rem;
        font-weight: 760;
        line-height: 1;
    }

    .metric-accent {
        color: #8B7BB5;
        font-size: 0.75rem;
        margin-top: 0.55rem;
    }

    .content-card {
        background: #FFFFFF;
        border: 1px solid #E7E3DE;
        border-radius: 18px;
        padding: 1.2rem 1.35rem;
        box-shadow: 0 8px 24px rgba(50, 45, 60, 0.045);
        margin-bottom: 1rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px solid #E7E3DE;
        border-radius: 16px;
        padding: 0.4rem;
        box-shadow: 0 6px 20px rgba(50, 45, 60, 0.035);
    }

    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
    }

    /* Text area */
    .stTextArea textarea {
        background: #FFFFFF !important;
        color: #29272D !important;
        border: 1px solid #DDD8D2 !important;
        border-radius: 14px !important;
    }

    .stTextArea textarea:focus {
        border-color: #A69BC2 !important;
        box-shadow: 0 0 0 1px #A69BC2 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #8B7BB5;
        color: #FFFFFF;
        border: 0;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #75639F;
        color: #FFFFFF;
        border: 0;
    }

    /* Skill badges */
    .badge {
        display: inline-block;
        background: #EEEAF5;
        color: #65558A;
        border: 1px solid #DDD5EB;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.78rem;
        font-weight: 650;
    }

    .badge-missing {
        display: inline-block;
        background: #F4F1EF;
        color: #756E68;
        border: 1px solid #E4DED8;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.78rem;
        font-weight: 650;
    }

    /* Recommendation */
    .recommendation {
        background: #F1EDF7;
        border: 1px solid #DDD5EB;
        border-left: 4px solid #8B7BB5;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        color: #4C4657;
        line-height: 1.55;
    }

    /* Streamlit metric cleanup */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E7E3DE;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(50, 45, 60, 0.045);
    }

    [data-testid="stMetricLabel"] {
        color: #77747B !important;
    }

    [data-testid="stMetricValue"] {
        color: #29272D !important;
    }


    /* Visual score intelligence */
    .visual-card {
        background: #FFFFFF;
        border: 1px solid #E7E3DE;
        border-radius: 20px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 8px 24px rgba(50, 45, 60, 0.045);
        margin: 0.8rem 0 1.1rem;
    }

    .visual-title {
        color: #29272D;
        font-size: 1.05rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
    }

    .visual-subtitle {
        color: #89858B;
        font-size: 0.82rem;
        margin-bottom: 1.1rem;
    }

    .ring-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        gap: 1rem;
        padding: 0.4rem 0.2rem 0.8rem;
    }

    .score-ring {
        --size: 132px;
        --track: #ECE8E4;
        --progress: #8B7BB5;
        width: var(--size);
        height: var(--size);
        border-radius: 50%;
        background: conic-gradient(
            var(--progress) calc(var(--score) * 1%),
            var(--track) 0
        );
        display: grid;
        place-items: center;
        position: relative;
    }

    .score-ring::before {
        content: "";
        width: 103px;
        height: 103px;
        border-radius: 50%;
        background: #FFFFFF;
        position: absolute;
    }

    .ring-content {
        position: relative;
        z-index: 2;
        text-align: center;
    }

    .ring-value {
        color: #29272D;
        font-size: 1.45rem;
        font-weight: 780;
        line-height: 1;
    }

    .ring-label {
        color: #77747B;
        font-size: 0.72rem;
        font-weight: 650;
        margin-top: 0.35rem;
    }

    .radar-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 330px;
    }

    .radar-title {
        color: #29272D;
        font-size: 0.92rem;
        font-weight: 720;
        text-align: center;
        margin-bottom: 0.35rem;
    }

    .radar-legend {
        color: #8B7BB5;
        font-size: 0.72rem;
        text-align: center;
        margin-top: -0.1rem;
    }

    .score-bar {
        margin: 0.7rem 0;
    }

    .score-bar-head {
        display: flex;
        justify-content: space-between;
        color: #5F6068;
        font-size: 0.78rem;
        font-weight: 650;
        margin-bottom: 0.32rem;
    }

    .score-track {
        width: 100%;
        height: 8px;
        background: #ECE8E4;
        border-radius: 999px;
        overflow: hidden;
    }

    .score-fill {
        height: 100%;
        background: #8B7BB5;
        border-radius: 999px;
    }

    /* Divider */
    hr {
        border-color: #E5E1DC;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E7E3DE;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="brand-wrap">
    <div class="brand-kicker">Career Intelligence Platform</div>
    <div class="brand-title"><span class="star">✦</span>AI Resume Intelligence</div>
    <div class="brand-tagline">Match smarter. Build your career.</div>
    <div style="color:#8B7BB5;font-size:0.72rem;font-weight:650;margin-top:0.45rem;">
        Phase 2 · Resume + Skill Intelligence
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="section-label">01 · Add your application materials</div>', unsafe_allow_html=True)
input_col1, input_col2 = st.columns([0.85, 1.15], gap="large")

with input_col1:
    resume_file = st.file_uploader("Upload Resume PDF", type="pdf")

with input_col2:
    job_description = st.text_area(
        "Paste Job Description",
        height=220,
        placeholder="Paste the job description here..."
    )



@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_pdf_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def calculate_semantic_score(resume_text, job_text, model):
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_text])
    score = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(score * 100, 2)


def get_keywords():
    return [
        "python", "sql", "machine learning", "deep learning", "nlp", "llm",
        "generative ai", "rag", "retrieval augmented generation", "langchain",
        "faiss", "chromadb", "vector database", "embeddings", "semantic search",
        "prompt engineering", "aws", "azure", "gcp", "docker", "kubernetes",
        "fastapi", "streamlit", "api", "git", "github", "pandas", "numpy",
        "scikit-learn", "power bi", "tableau", "excel", "jira", "confluence",
        "agile", "scrum", "requirements gathering", "stakeholder management",
        "user stories", "uat", "business analysis", "data analysis", "etl",
        "data visualization"
    ]


def analyze_keywords(resume_text, job_text):
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    required = []
    matching = []
    missing = []

    for keyword in get_keywords():
        if keyword in job_lower:
            required.append(keyword)
            if keyword in resume_lower:
                matching.append(keyword)
            else:
                missing.append(keyword)

    ats_score = round((len(matching) / len(required)) * 100, 2) if required else 0
    return ats_score, required, matching, missing


# =========================
# PHASE 2 — SKILL GAP ANALYZER
# =========================

SKILL_CATEGORIES = {
    "Programming": {"python", "sql"},
    "AI / ML": {"machine learning", "deep learning", "nlp", "llm", "generative ai"},
    "GenAI / NLP": {
        "rag", "retrieval augmented generation", "langchain",
        "embeddings", "semantic search", "prompt engineering"
    },
    "Data & Analytics": {
        "pandas", "numpy", "scikit-learn", "power bi", "tableau",
        "excel", "data analysis", "data visualization", "etl"
    },
    "Cloud & DevOps": {
        "aws", "azure", "gcp", "docker", "kubernetes", "fastapi"
    },
    "Data Infrastructure": {
        "faiss", "chromadb", "vector database", "api", "streamlit",
        "git", "github"
    },
    "Business / Delivery": {
        "jira", "confluence", "agile", "scrum",
        "requirements gathering", "stakeholder management",
        "user stories", "uat", "business analysis"
    },
}


def analyze_skill_gaps(resume_text, job_text):
    """Return category-aware skill alignment and prioritized skill gaps."""
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()

    required = []
    matching = []
    missing = []

    for skill in get_keywords():
        if skill in job_lower:
            required.append(skill)
            if skill in resume_lower:
                matching.append(skill)
            else:
                missing.append(skill)

    # Calculate a genuine category-level skill alignment score.
    category_scores = {}
    category_required = {}
    for category, skills in SKILL_CATEGORIES.items():
        present_in_jd = [s for s in skills if s in required]
        if not present_in_jd:
            continue
        matched_in_resume = [s for s in present_in_jd if s in matching]
        category_required[category] = present_in_jd
        category_scores[category] = round(
            len(matched_in_resume) / len(present_in_jd) * 100, 2
        )

    skill_alignment = (
        round(sum(category_scores.values()) / len(category_scores), 2)
        if category_scores else 0
    )

    # Prioritize missing skills using how often they occur in the JD.
    priority = []
    for skill in missing:
        frequency = len(re.findall(r"\b" + re.escape(skill) + r"\b", job_lower))
        if frequency >= 2:
            level = "High"
        else:
            level = "Medium"
        priority.append((skill, level, frequency))

    priority.sort(key=lambda item: (-item[2], item[0]))

    return {
        "required": required,
        "matching": matching,
        "missing": missing,
        "category_scores": category_scores,
        "category_required": category_required,
        "skill_alignment": skill_alignment,
        "priority": priority,
    }


def skill_gap_message(skill_data):
    missing_count = len(skill_data["missing"])
    alignment = skill_data["skill_alignment"]

    if missing_count == 0:
        return "Excellent skill coverage. Your resume contains the identified role-relevant skills."
    if alignment >= 75:
        return f"Strong technical alignment, with {missing_count} skill gap{'s' if missing_count != 1 else ''} worth addressing."
    if alignment >= 50:
        return f"Moderate skill alignment. Prioritize the highest-impact missing skills before applying."
    return f"Several role-relevant skills are missing. Focus on the priority gaps before applying."

def generate_recommendation(semantic_score, ats_score):
    avg = (semantic_score + ats_score) / 2

    if avg >= 75:
        return "Strong match. This candidate is well aligned with the role."
    elif avg >= 55:
        return "Moderate match. The resume has relevant experience but needs keyword and positioning improvements."
    else:
        return "Low match. The resume needs stronger alignment with the job description."


def generate_resume_suggestions(missing_keywords):
    if not missing_keywords:
        return ["Resume is well aligned. Add measurable project outcomes to improve impact."]

    return [
        f"Add '{keyword.title()}' naturally in your skills, projects, or experience section if you have relevant experience."
        for keyword in missing_keywords[:8]
    ]


def generate_resume_bullets(matching_keywords, missing_keywords):
    bullets = []

    if "rag" in missing_keywords or "retrieval augmented generation" in missing_keywords or "rag" in matching_keywords:
        bullets.append("Built a Retrieval-Augmented Generation application using document chunking, embeddings, semantic search, and vector retrieval.")

    if "faiss" in missing_keywords or "faiss" in matching_keywords or "vector database" in missing_keywords:
        bullets.append("Implemented FAISS-based vector search to retrieve relevant document sections using sentence embeddings and similarity search.")

    if "streamlit" in missing_keywords or "streamlit" in matching_keywords:
        bullets.append("Developed an interactive Streamlit web application for resume analysis, ATS optimization, and interview preparation.")

    if "python" in missing_keywords or "python" in matching_keywords:
        bullets.append("Built Python-based workflows for PDF parsing, skill extraction, semantic matching, and candidate-job analysis.")

    if not bullets:
        bullets.append("Improved resume-job alignment by identifying skill gaps, matching keywords, and generating targeted improvement suggestions.")

    return bullets


def generate_interview_questions(matching_keywords, missing_keywords):
    technical = []

    for skill in matching_keywords[:5]:
        technical.append(f"Can you explain your experience with {skill.title()}?")

    for skill in missing_keywords[:5]:
        technical.append(f"How would you approach learning or applying {skill.title()} for this role?")

    project = [
        "Explain how your AI Resume Intelligence Platform works end to end.",
        "Why did you use Sentence Transformers?",
        "What is FAISS and why is it useful for semantic search?",
        "How does semantic search differ from keyword search?",
        "How would you scale this application for multiple users?"
    ]

    behavioral = [
        "Tell me about yourself.",
        "Why are you interested in this role?",
        "Tell me about a challenging project you worked on.",
        "Describe a time you worked with stakeholders.",
        "Why should we hire you?"
    ]

    return technical[:8], project, behavioral



# =========================
# PHASE 2.2 — RESUME HEALTH & RED FLAG DETECTOR
# =========================

RESUME_SECTION_PATTERNS = {
    "Contact Information": [
        r"\b(?:email|e-mail)\b",
        r"(?:\+?\d[\d\s().-]{8,}\d)",
    ],
    "Education": [
        r"\b(?:education|academic background|qualifications)\b",
        r"\b(?:b\.?tech|bachelor|master|bsc|msc|mba|bca|mca|phd)\b",
    ],
    "Experience": [
        r"\b(?:experience|work experience|professional experience|internship|internships)\b",
    ],
    "Projects": [
        r"\b(?:projects|project experience|academic projects)\b",
    ],
    "Skills": [
        r"\b(?:skills|technical skills|core competencies|technologies)\b",
    ],
    "Certifications": [
        r"\b(?:certifications|certificates|licenses)\b",
    ],
}

COMMON_WEAK_PHRASES = [
    "worked on", "responsible for", "helped with", "helped to",
    "was responsible", "involved in", "participated in",
    "worked with", "assisted with", "did", "made", "handled",
    "learned about", "used", "knowledge of", "familiar with",
]

ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "created", "engineered",
    "analyzed", "automated", "optimized", "deployed", "integrated", "led",
    "managed", "improved", "reduced", "increased", "evaluated", "trained",
    "fine-tuned", "developed", "architected", "configured", "delivered",
]

QUANTIFICATION_PATTERN = re.compile(
    r"(?:\b\d+(?:\.\d+)?\b|"
    r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:k|m|million|thousand)\b|"
    r"\b(?:increased|reduced|improved|achieved|saved)\b.{0,35}\b\d+)",
    re.I,
)

DATE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"(?:[a-z]*)?\s+(?:19|20)\d{2}\b",
    re.I,
)

def _has_any_pattern(text, patterns):
    return any(re.search(pattern, text, re.I) for pattern in patterns)

def _split_resume_bullets(text):
    bullets = []
    for line in text.splitlines():
        clean = re.sub(r"^\s*(?:[-•▪●◦*]|\d+[.)])\s*", "", line).strip()
        if clean and len(clean.split()) >= 4:
            if line.strip().startswith(("-", "•", "▪", "●", "◦", "*")) or re.match(r"^\s*\d+[.)]\s+", line):
                bullets.append(clean)
    return bullets

def analyze_resume_health(resume_text, page_count=None):
    """Perform deterministic resume-health checks without requiring an LLM."""
    text = resume_text.strip()
    lower = text.lower()
    words = re.findall(r"\b[\w+#.-]+\b", text)
    word_count = len(words)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullets = _split_resume_bullets(text)

    strengths = []
    warnings = []
    critical = []
    checks = []

    # Contact details
    email_ok = bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I))
    phone_ok = bool(re.search(r"(?:\+?\d[\d\s().-]{8,}\d)", text))
    linkedin_ok = "linkedin.com" in lower or "linkedin" in lower
    github_ok = "github.com" in lower or re.search(r"\bgithub\b", lower) is not None

    if email_ok:
        strengths.append("Professional email address detected.")
    else:
        critical.append(("Missing email address", "Add a professional email address so recruiters can contact you."))
    if phone_ok:
        strengths.append("Phone number detected.")
    else:
        warnings.append(("Missing phone number", "Add a reachable phone number if appropriate for the application."))
    if linkedin_ok:
        strengths.append("LinkedIn profile detected.")
    else:
        warnings.append(("LinkedIn profile not detected", "Add your LinkedIn URL if you use LinkedIn professionally."))
    if github_ok:
        strengths.append("GitHub profile detected.")
    else:
        warnings.append(("GitHub profile not detected", "Add GitHub when your role is technical and you have relevant work to showcase."))

    # Sections
    section_presence = {}
    for section, patterns in RESUME_SECTION_PATTERNS.items():
        present = _has_any_pattern(text, patterns)
        section_presence[section] = present
        if present:
            checks.append((section, "Pass"))
        else:
            checks.append((section, "Review"))

    required_sections = ["Education", "Experience", "Projects", "Skills"]
    missing_sections = [s for s in required_sections if not section_presence[s]]
    if not missing_sections:
        strengths.append("Core resume sections are present.")
    else:
        for section in missing_sections:
            critical.append((f"Missing {section} section", f"Add a clear {section} section if it is relevant to your profile."))

    # Resume length
    if page_count is not None:
        if page_count > 2:
            critical.append(("Resume exceeds 2 pages", f"This PDF contains {page_count} pages. Consider tightening it unless your experience level justifies a longer resume."))
        elif page_count == 2:
            warnings.append(("Two-page resume", "Keep the second page only if the content is relevant and substantial."))
        elif page_count == 1:
            strengths.append("Resume is contained on one page.")
    if word_count < 180:
        warnings.append(("Very short resume", f"Only about {word_count} words were extracted. Check whether important experience or projects are missing."))
    elif word_count > 900:
        warnings.append(("Dense resume", f"About {word_count} words were extracted. Remove lower-value content and keep the strongest evidence."))
    else:
        strengths.append("Resume length is within a reasonable range.")

    # Bullets and action language
    if bullets:
        long_bullets = [b for b in bullets if len(b.split()) > 35]
        weak_bullets = [
            b for b in bullets
            if any(phrase in b.lower() for phrase in COMMON_WEAK_PHRASES)
        ]
        action_bullets = [
            b for b in bullets
            if any(re.match(rf"^\s*{re.escape(v)}\b", b, re.I) for v in ACTION_VERBS)
        ]
        quantified_bullets = [b for b in bullets if QUANTIFICATION_PATTERN.search(b)]

        if long_bullets:
            warnings.append(("Long bullet points", f"{len(long_bullets)} bullet(s) exceed roughly 35 words. Split or tighten them for faster scanning."))
        else:
            strengths.append("Bullet points are generally concise.")

        if weak_bullets:
            warnings.append(("Generic resume language", f"{len(weak_bullets)} bullet(s) use weak phrases such as 'worked on' or 'responsible for'. Replace them with specific actions and outcomes."))
        else:
            strengths.append("Bullets avoid common generic phrasing.")

        if action_bullets:
            strengths.append(f"Strong action verbs lead {len(action_bullets)} bullet(s).")
        else:
            warnings.append(("Weak action verbs", "Start bullets with precise action verbs such as developed, implemented, analyzed, optimized, or deployed."))

        if quantified_bullets:
            strengths.append(f"Measurable evidence detected in {len(quantified_bullets)} bullet(s).")
        else:
            warnings.append(("Low quantification", "Add metrics where truthful: accuracy, latency, users, dataset size, time saved, percentage improvement, or scale."))
    else:
        critical.append(("No bullet-style achievements detected", "Convert experience and project descriptions into concise, achievement-oriented bullets."))

    # Repetition
    normalized_words = [
        w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", text)
        if w.lower() not in {"the", "and", "for", "with", "from", "that", "this", "using", "your"}
    ]
    freq = {}
    for word in normalized_words:
        freq[word] = freq.get(word, 0) + 1
    repeated = sorted(
        [(word, count) for word, count in freq.items() if count >= 6],
        key=lambda x: (-x[1], x[0])
    )
    repeated = [item for item in repeated if item[0] not in {
        "python", "data", "project", "experience", "skills", "resume", "machine"
    }]
    if repeated:
        top_repeated = ", ".join(f"{w} ({c}×)" for w, c in repeated[:4])
        warnings.append(("Repeated wording", f"Frequent terms detected: {top_repeated}. Check whether the wording can be made more varied and specific."))
    else:
        strengths.append("No obvious excessive word repetition detected.")

    # Dates
    date_count = len(DATE_PATTERN.findall(text))
    if date_count >= 2:
        strengths.append("Career/education dates detected.")
    else:
        warnings.append(("Limited date information", "Add clear dates for education, experience, internships, or projects where relevant."))

    # Section-specific quality signals
    project_words = 0
    if section_presence["Projects"]:
        project_words = len(text.lower().split("projects", 1)[-1].split()) if "projects" in lower else 0
    if section_presence["Projects"] and project_words >= 30:
        strengths.append("Projects section contains substantive detail.")

    # Score: weighted deterministic health model
    score = 100
    score -= min(15, 7 * len(critical))
    score -= min(24, 4 * len(warnings))
    score += min(6, len(strengths))
    score = max(0, min(100, score))

    if score >= 85:
        health_label = "Excellent"
    elif score >= 70:
        health_label = "Healthy"
    elif score >= 50:
        health_label = "Needs Attention"
    else:
        health_label = "High Risk"

    priority_fixes = []
    for title, detail in critical[:4]:
        priority_fixes.append(("Critical", title, detail))
    for title, detail in warnings[:5]:
        priority_fixes.append(("Warning", title, detail))

    return {
        "score": score,
        "label": health_label,
        "strengths": strengths,
        "warnings": warnings,
        "critical": critical,
        "checks": checks,
        "priority_fixes": priority_fixes,
        "page_count": page_count,
        "word_count": word_count,
        "bullet_count": len(bullets),
        "quantified_bullets": sum(1 for b in bullets if QUANTIFICATION_PATTERN.search(b)),
        "weak_bullets": sum(1 for b in bullets if any(p in b.lower() for p in COMMON_WEAK_PHRASES)),
        "email_ok": email_ok,
        "phone_ok": phone_ok,
        "linkedin_ok": linkedin_ok,
        "github_ok": github_ok,
    }

def resume_health_summary(health):
    score = health["score"]
    if score >= 85:
        return "Your resume has a strong structural foundation. Focus next on polishing impact and tailoring it to each role."
    if score >= 70:
        return "Your resume is in good shape, but a few targeted fixes could make it more recruiter-friendly and ATS-ready."
    if score >= 50:
        return "Your resume has a workable foundation, but several structural and content issues should be fixed before applying broadly."
    return "Your resume needs a focused cleanup before applying. Resolve the critical red flags first, then improve bullet quality and evidence."


# =========================
# PHASE 3.1 — GEMINI AI RESUME INTELLIGENCE
# =========================

def generate_gemini_resume_analysis(resume_text, job_description):
    """Generate personalized AI analysis using Gemini."""
    prompt = f"""
You are an expert technical recruiter, ATS specialist, and career coach.
Analyze the candidate's resume against the target job description.

RESUME:
{resume_text[:18000]}

JOB DESCRIPTION:
{job_description[:12000]}

Return a concise, practical analysis in Markdown with EXACTLY these headings:

## Executive Assessment
Give a 3-5 sentence assessment of the candidate's fit.

## Top Strengths
Give 5 specific strengths based only on evidence in the resume and job description.

## Critical Gaps
Give the 5 most important gaps. Distinguish between missing skills and missing resume signals.

## Resume Improvements
Give 6 concrete edits the candidate should make. Never invent experience, skills, metrics, employers, projects, or achievements.

## ATS Optimization
Give 5 high-value ATS/keyword improvements and where each can naturally appear if genuinely applicable.

## Best Positioning
Suggest the strongest professional positioning for this role in 2-3 sentences.

## Recruiter Verdict
Give one of: Strong Fit, Potential Fit, or Weak Fit, followed by a short explanation.

Rules:
- Be specific to this resume and this job description.
- Do not fabricate facts.
- Do not claim the candidate has a skill merely because the job requires it.
- Treat deterministic scores as supporting signals, not absolute truth.
- Prioritize actionable advice over generic career advice.
"""
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text


if resume_file and job_description:
    with st.spinner("Analyzing resume and job description..."):
        model = load_embedding_model()
        resume_text = extract_pdf_text(resume_file)
        try:
            resume_page_count = len(PdfReader(resume_file).pages)
        except Exception:
            resume_page_count = None

        resume_health = analyze_resume_health(resume_text, resume_page_count)

        semantic_score = calculate_semantic_score(resume_text, job_description, model)
        ats_score, required_keywords, matching_keywords, missing_keywords = analyze_keywords(
            resume_text,
            job_description
        )

        skill_data = analyze_skill_gaps(resume_text, job_description)

        recommendation = generate_recommendation(semantic_score, ats_score)
        suggestions = generate_resume_suggestions(missing_keywords)
        bullets = generate_resume_bullets(matching_keywords, missing_keywords)
        technical_qs, project_qs, behavioral_qs = generate_interview_questions(
            matching_keywords,
            missing_keywords
        )

    st.markdown('<div class="section-label">02 · Your match overview</div>', unsafe_allow_html=True)
    st.markdown("### Analysis Dashboard")

    overall_score = round((semantic_score + ats_score) / 2, 2)

    score1, score2, score3 = st.columns(3)
    with score1:
        st.metric("Overall Match", f"{overall_score}%")
    with score2:
        st.metric("Semantic Match", f"{semantic_score}%")
    with score3:
        st.metric("ATS Keywords", f"{ats_score}%")

    # ---------------------------------
    # Editorial AI visual score section
    # ---------------------------------
    # Phase 2 dimensions: ATS remains keyword coverage,
    # while Skill Alignment now comes from category-level skill analysis.
    keyword_coverage = ats_score
    skill_alignment = skill_data["skill_alignment"]

    radar_values = [
        max(0, min(100, overall_score)),
        max(0, min(100, semantic_score)),
        max(0, min(100, ats_score)),
        max(0, min(100, keyword_coverage)),
        max(0, min(100, skill_alignment)),
    ]

    # Lightweight SVG radar chart — no additional Python package required.
    cx, cy, radius = 160, 155, 105
    angles = [-90 + i * 72 for i in range(5)]

    def polar_point(r, angle):
        rad = math.radians(angle)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    grid_levels = []
    for level in (25, 50, 75, 100):
        pts = []
        for angle in angles:
            x, y = polar_point(radius * level / 100, angle)
            pts.append(f"{x:.1f},{y:.1f}")
        grid_levels.append(
            '<polygon points="' + " ".join(pts) +
            '" fill="none" stroke="#E6E1DC" stroke-width="1"/>'
        )

    axes = []
    labels = []
    label_names = ["Overall", "Semantic", "ATS", "Keywords", "Skills"]

    for angle, name in zip(angles, label_names):
        x1, y1 = polar_point(0, angle)
        x2, y2 = polar_point(radius, angle)
        axes.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#E6E1DC" stroke-width="1"/>'
        )

        lx, ly = polar_point(radius + 23, angle)
        anchor = "middle"
        if lx < cx - 15:
            anchor = "end"
        elif lx > cx + 15:
            anchor = "start"

        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'fill="#77747B" font-size="11" font-family="Arial, sans-serif">{name}</text>'
        )

    data_pts = []
    data_circles = []
    for value, angle in zip(radar_values, angles):
        x, y = polar_point(radius * value / 100, angle)
        data_pts.append(f"{x:.1f},{y:.1f}")
        data_circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#8B7BB5"/>'
        )

    radar_svg = (
        '<svg viewBox="0 0 320 320" width="100%" height="320" '
        'xmlns="http://www.w3.org/2000/svg" '
        'aria-label="Resume match intelligence radar chart">'
        + "".join(grid_levels)
        + "".join(axes)
        + '<polygon points="' + " ".join(data_pts) +
        '" fill="#8B7BB5" fill-opacity="0.18" stroke="#8B7BB5" '
        'stroke-width="2.5" stroke-linejoin="round"/>'
        + "".join(data_circles)
        + "".join(labels)
        + '</svg>'
    )

    st.markdown('<div class="section-label">03 · Match intelligence</div>', unsafe_allow_html=True)

    visual_left, visual_right = st.columns([1.05, 0.95], gap="large")

    with visual_left:
        st.markdown(
            f"""
            <div class="visual-card">
                <div class="visual-title">✦ Score profile</div>
                <div class="visual-subtitle">A quick visual read of your resume-to-role alignment.</div>
                <div class="ring-row">
                    <div>
                        <div class="score-ring" style="--score:{overall_score};">
                            <div class="ring-content">
                                <div class="ring-value">{overall_score:.0f}%</div>
                                <div class="ring-label">Overall</div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="score-ring" style="--score:{semantic_score}; --progress:#9B8BC2;">
                            <div class="ring-content">
                                <div class="ring-value">{semantic_score:.0f}%</div>
                                <div class="ring-label">Semantic</div>
                            </div>
                        </div>
                    </div>
                    <div>
                        <div class="score-ring" style="--score:{ats_score}; --progress:#A99CC5;">
                            <div class="ring-content">
                                <div class="ring-value">{ats_score:.0f}%</div>
                                <div class="ring-label">ATS</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with visual_right:
        st.markdown(
            f"""
            <div class="visual-card">
                <div class="radar-title">Resume Match Profile</div>
                <div class="radar-legend">Lavender area = current alignment</div>
                <div class="radar-wrap">{radar_svg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="visual-card">
            <div class="visual-title">Alignment breakdown</div>
            <div class="visual-subtitle">Exact scores behind the visual profile.</div>
        """,
        unsafe_allow_html=True
    )

    for label, value in [
        ("Overall match", overall_score),
        ("Semantic similarity", semantic_score),
        ("ATS compatibility", ats_score),
        ("Keyword coverage", keyword_coverage),
        ("Skill alignment", skill_alignment),
    ]:
        safe_value = max(0, min(100, float(value)))
        st.markdown(
            f"""
            <div class="score-bar">
                <div class="score-bar-head">
                    <span>{label}</span><span>{safe_value:.1f}%</span>
                </div>
                <div class="score-track">
                    <div class="score-fill" style="width:{safe_value}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------
    # Phase 2 — Skill Gap Intelligence
    # ---------------------------------
    st.markdown('<div class="section-label">04 · Skill gap intelligence</div>', unsafe_allow_html=True)

    gap_col1, gap_col2 = st.columns([1.05, 0.95], gap="large")

    with gap_col1:
        st.markdown(
            f"""
            <div class="visual-card">
                <div class="visual-title">✦ Skill coverage</div>
                <div class="visual-subtitle">
                    Category-level alignment based on skills explicitly detected in the job description.
                </div>
                <div style="display:flex;align-items:center;gap:1rem;margin:0.4rem 0 1rem;">
                    <div style="font-size:2.5rem;font-weight:800;color:#29272D;">
                        {skill_alignment:.0f}%
                    </div>
                    <div style="color:#77747B;font-size:0.82rem;line-height:1.45;">
                        {len(skill_data["matching"])} of {len(skill_data["required"])}
                        identified role skills are present in the resume.
                    </div>
                </div>
                <div class="score-track">
                    <div class="score-fill" style="width:{max(0,min(100,skill_alignment))}%;"></div>
                </div>
                <div style="margin-top:1rem;color:#5F6068;font-size:0.85rem;line-height:1.55;">
                    {skill_gap_message(skill_data)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="visual-card">
                <div class="visual-title">Skill categories</div>
                <div class="visual-subtitle">Where your resume is strongest and where it needs work.</div>
            """,
            unsafe_allow_html=True
        )

        if skill_data["category_scores"]:
            for category, value in skill_data["category_scores"].items():
                safe_value = max(0, min(100, float(value)))
                st.markdown(
                    f"""
                    <div class="score-bar">
                        <div class="score-bar-head">
                            <span>{category}</span><span>{safe_value:.0f}%</span>
                        </div>
                        <div class="score-track">
                            <div class="score-fill" style="width:{safe_value}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No categorized skills were detected in the job description.")

        st.markdown('</div>', unsafe_allow_html=True)

    with gap_col2:
        st.markdown(
            """
            <div class="visual-card">
                <div class="visual-title">⚠ Priority skill gaps</div>
                <div class="visual-subtitle">Start with the skills that appear most important in the target role.</div>
            """,
            unsafe_allow_html=True
        )

        if skill_data["priority"]:
            for skill, level, frequency in skill_data["priority"][:8]:
                level_class = "#8B7BB5" if level == "High" else "#A69BC2"
                frequency_text = (
                    f"mentioned {frequency}× in the JD"
                    if frequency > 1
                    else "relevant to the target role"
                )
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:0.72rem 0;border-bottom:1px solid #EEEAE6;">
                        <div>
                            <div style="color:#29272D;font-weight:720;font-size:0.88rem;">
                                {skill.title()}
                            </div>
                            <div style="color:#89858B;font-size:0.74rem;margin-top:0.2rem;">
                                {frequency_text}
                            </div>
                        </div>
                        <span style="background:#F1EDF7;color:{level_class};border:1px solid #DDD5EB;
                                     border-radius:999px;padding:0.25rem 0.55rem;font-size:0.68rem;
                                     font-weight:750;">
                            {level}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("No priority skill gaps detected.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="visual-card">
            <div class="visual-title">✓ Matched skills</div>
            <div class="visual-subtitle">Skills detected in both your resume and the target job description.</div>
        """,
        unsafe_allow_html=True
    )

    if skill_data["matching"]:
        badges = "".join(
            f'<span class="badge">{skill.title()}</span>'
            for skill in skill_data["matching"]
        )
        st.markdown(badges, unsafe_allow_html=True)
    else:
        st.info("No role-relevant skills were matched yet.")

    st.markdown('</div>', unsafe_allow_html=True)


    # ---------------------------------
    # Phase 2.2 — Resume Health & Red Flags
    # ---------------------------------
    st.markdown('<div class="section-label">05 · Resume health intelligence</div>', unsafe_allow_html=True)

    health = resume_health
    health_score = health["score"]
    health_color = "#8B7BB5" if health_score >= 70 else "#B38B5D" if health_score >= 50 else "#A45D6A"

    health_left, health_right = st.columns([0.95, 1.05], gap="large")

    with health_left:
        st.markdown(
            f"""
            <div class="visual-card">
                <div class="visual-title">✦ Resume Health</div>
                <div class="visual-subtitle">A structural and content-quality check of the uploaded resume.</div>
                <div style="display:flex;align-items:center;gap:1.25rem;margin:0.5rem 0 1rem;">
                    <div class="score-ring" style="--score:{health_score};--progress:{health_color};">
                        <div class="ring-content">
                            <div class="ring-value">{health_score:.0f}</div>
                            <div class="ring-label">/ 100</div>
                        </div>
                    </div>
                    <div>
                        <div style="font-size:1.35rem;font-weight:780;color:#29272D;">{health["label"]}</div>
                        <div style="color:#77747B;font-size:0.82rem;line-height:1.5;margin-top:0.35rem;">
                            {resume_health_summary(health)}
                        </div>
                    </div>
                </div>
                <div class="score-track">
                    <div class="score-fill" style="width:{health_score}%;background:{health_color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Words", health["word_count"])
        with m2:
            st.metric("Bullets", health["bullet_count"])
        with m3:
            st.metric("Pages", health["page_count"] if health["page_count"] is not None else "—")

    with health_right:
        st.markdown(
            """
            <div class="visual-card">
                <div class="visual-title">✦ Resume signals</div>
                <div class="visual-subtitle">Fast checks for recruiter-readiness and professional completeness.</div>
            """,
            unsafe_allow_html=True
        )

        signal_items = [
            ("Email", health["email_ok"]),
            ("Phone", health["phone_ok"]),
            ("LinkedIn", health["linkedin_ok"]),
            ("GitHub", health["github_ok"]),
        ]
        signal_html = ""
        for label, passed in signal_items:
            icon = "✓" if passed else "!"
            bg = "#F1EDF7" if passed else "#F6F0EA"
            fg = "#65558A" if passed else "#8A6847"
            signal_html += f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            padding:0.65rem 0;border-bottom:1px solid #EEEAE6;">
                    <span style="color:#4C4657;font-size:0.84rem;font-weight:650;">{label}</span>
                    <span style="background:{bg};color:{fg};border:1px solid #DDD5EB;
                                 border-radius:999px;padding:0.22rem 0.55rem;font-size:0.68rem;font-weight:750;">
                        {icon} {"Detected" if passed else "Review"}
                    </span>
                </div>
            """
        st.markdown(signal_html, unsafe_allow_html=True)
        st.markdown(
            f"""
                <div style="margin-top:0.9rem;color:#77747B;font-size:0.78rem;">
                    Quantified bullets: <strong>{health["quantified_bullets"]}</strong>
                    &nbsp;·&nbsp;
                    Generic/weak bullets: <strong>{health["weak_bullets"]}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    red_col, strength_col = st.columns(2, gap="large")

    with red_col:
        st.markdown(
            """
            <div class="visual-card">
                <div class="visual-title">🚨 Red flags</div>
                <div class="visual-subtitle">Issues that can reduce clarity, credibility, or recruiter scanability.</div>
            """,
            unsafe_allow_html=True
        )
        if health["critical"]:
            for title, detail in health["critical"]:
                st.markdown(
                    f"""
                    <div style="padding:0.75rem 0;border-bottom:1px solid #EEEAE6;">
                        <div style="color:#7F4350;font-weight:750;font-size:0.86rem;">✕ {title}</div>
                        <div style="color:#77747B;font-size:0.76rem;line-height:1.45;margin-top:0.2rem;">{detail}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.success("No critical red flags detected.")
        if health["warnings"]:
            st.markdown("**⚠ Warnings to review**")
            for title, detail in health["warnings"][:6]:
                st.markdown(f"**{title}** — {detail}")
        st.markdown("</div>", unsafe_allow_html=True)

    with strength_col:
        st.markdown(
            """
            <div class="visual-card">
                <div class="visual-title">✓ What's working</div>
                <div class="visual-subtitle">Positive signals detected in the current resume.</div>
            """,
            unsafe_allow_html=True
        )
        if health["strengths"]:
            for strength in health["strengths"][:10]:
                st.markdown(f"✓ {strength}")
        else:
            st.info("No strong positive signals were detected yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Resume health checklist", expanded=False):
        for item, status in health["checks"]:
            icon = "✓" if status == "Pass" else "⚠"
            st.write(f"{icon} **{item}** — {status}")

    with st.expander("Priority fixes", expanded=False):
        if health["priority_fixes"]:
            for level, title, detail in health["priority_fixes"]:
                st.write(f"**{level} · {title}**")
                st.caption(detail)
        else:
            st.success("No priority fixes detected.")

    st.markdown('<div class="section-label">06 · Hiring insight</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="recommendation"><strong>✦ Recommendation</strong><br>{recommendation}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">07 · Skill alignment</div>', unsafe_allow_html=True)
    skill_col1, skill_col2 = st.columns(2, gap="large")

    with skill_col1:
        st.markdown("#### ✓ Matching skills")
        if matching_keywords:
            badges = "".join(
                f'<span class="badge">{keyword.title()}</span>'
                for keyword in matching_keywords
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("No matching keywords found.")

    with skill_col2:
        st.markdown("#### + Skills to strengthen")
        if missing_keywords:
            badges = "".join(
                f'<span class="badge-missing">{keyword.title()}</span>'
                for keyword in missing_keywords
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.success("No major missing keywords found.")

    st.markdown('<div class="section-label">08 · Resume improvement</div>', unsafe_allow_html=True)
    suggestion_col, bullet_col = st.columns(2, gap="large")

    with suggestion_col:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### ✦ Improvement suggestions")
        for suggestion in suggestions:
            st.write("•", suggestion)
        st.markdown('</div>', unsafe_allow_html=True)

    with bullet_col:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("#### ✦ Suggested bullet points")
        for bullet in bullets:
            st.write("•", bullet)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">09 · Interview preparation</div>', unsafe_allow_html=True)

    with st.expander("Technical interview questions", expanded=False):
        for i, q in enumerate(technical_qs, start=1):
            st.write(f"**{i}.** {q}")

    with st.expander("Project-based interview questions", expanded=False):
        for i, q in enumerate(project_qs, start=1):
            st.write(f"**{i}.** {q}")

    with st.expander("Behavioral interview questions", expanded=False):
        for i, q in enumerate(behavioral_qs, start=1):
            st.write(f"**{i}.** {q}")

    with st.expander("Resume text preview", expanded=False):
        st.write(resume_text[:2000])

    # ---------------------------------
    # Phase 3.1 — Gemini AI Career Intelligence
    # ---------------------------------
    st.markdown('<div class="section-label">10 · Gemini AI career intelligence</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="visual-card">
            <div class="visual-title">✦ AI-powered resume review</div>
            <div class="visual-subtitle">
                Gemini reviews your resume and target role to turn the existing Phase 2 signals into personalized career advice.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("✦ Generate Gemini AI Analysis", key="generate_gemini_analysis"):
        with st.spinner("Gemini is analyzing your resume and target role..."):
            try:
                gemini_analysis = generate_gemini_resume_analysis(
                    resume_text,
                    job_description
                )
                st.session_state["gemini_analysis"] = gemini_analysis
                st.session_state["gemini_analysis_error"] = None
            except Exception as exc:
                st.session_state["gemini_analysis"] = None
                st.session_state["gemini_analysis_error"] = str(exc)

    if st.session_state.get("gemini_analysis"):
        st.markdown(
            """
            <div class="visual-card">
                <div class="visual-title">✦ Gemini's personalized assessment</div>
                <div class="visual-subtitle">AI-generated insights based on your uploaded resume and target job description.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(st.session_state["gemini_analysis"])

    if st.session_state.get("gemini_analysis_error"):
        st.error(
            "Gemini analysis could not be generated. Check that your Gemini API key is configured correctly and try again."
        )
        with st.expander("Technical error details", expanded=False):
            st.code(st.session_state["gemini_analysis_error"])

        # ==========================================
# PHASE 3.2 — AI SECTION IMPROVER
# ==========================================

def improve_resume_section(section_name, section_text):
    """
    Use Gemini to improve one resume section without inventing
    experience, skills, metrics, projects, employers, or achievements.
    """

    prompt = f"""
You are an expert resume writer, ATS specialist, and technical recruiter.

Your task is to improve ONE section of a candidate's resume.

SECTION NAME:
{section_name}

ORIGINAL SECTION:
{section_text[:10000]}

Improve the section while preserving the candidate's factual information.

STRICT RULES:
1. Do NOT invent experience.
2. Do NOT invent skills.
3. Do NOT invent employers.
4. Do NOT invent projects.
5. Do NOT invent achievements.
6. Do NOT invent numbers, percentages, metrics, users, datasets, or results.
7. Do NOT add technologies that are not present in the original text.
8. Do NOT change the meaning of the candidate's experience.
9. Make the writing concise, professional, impactful, and ATS-friendly.
10. Use strong action verbs where appropriate.
11. Remove unnecessary filler and repetitive wording.
12. Preserve truthful information even if the original writing is weak.

Return Markdown with EXACTLY these headings:

## Improved Version

Provide the improved resume section.

## What Was Improved

Give 3-5 concise bullet points explaining the changes.

## ATS Keywords

List important keywords that are already supported by the original section.

## Recruiter Impact

Give 2-3 sentences explaining why the improved version is stronger.

Remember:
The goal is to improve the candidate's writing, NOT to create new qualifications.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


# ==========================================
# PHASE 3.2 — UI
# ==========================================

st.markdown(
    '<div class="section-label">11 · AI section improver</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="visual-card">
        <div class="visual-title">✦ Improve any resume section</div>
        <div class="visual-subtitle">
            Select a section, paste its current content, and let Gemini
            rewrite it into a clearer, stronger, ATS-friendly version.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

section_col1, section_col2 = st.columns([0.75, 1.25], gap="large")

with section_col1:

    section_name = st.selectbox(
        "Select resume section",
        [
            "Professional Summary",
            "Skills",
            "Experience",
            "Projects",
            "Education",
            "Certifications",
        ],
        key="phase_32_section_name"
    )

with section_col2:

    st.markdown(
        """
        <div style="
            background:#F1EDF7;
            border:1px solid #DDD5EB;
            border-radius:12px;
            padding:0.85rem 1rem;
            color:#5F5868;
            font-size:0.82rem;
            line-height:1.5;
            margin-top:1.8rem;
        ">
            <strong style="color:#65558A;">✦ Tip</strong><br>
            Paste the section exactly as it currently appears on your resume.
            Gemini will improve the wording without inventing information.
        </div>
        """,
        unsafe_allow_html=True
    )


section_text = st.text_area(
    "Paste the section you want to improve",
    height=230,
    placeholder=(
        "Example:\n\n"
        "Built a machine learning model using Python and scikit-learn "
        "to predict customer churn."
    ),
    key="phase_32_section_text"
)


improve_col1, improve_col2 = st.columns([0.75, 0.25])

with improve_col1:

    if st.button(
        "✦ Improve This Section",
        key="phase_32_improve_button"
    ):

        if not section_text.strip():

            st.warning(
                "Please paste the resume section you want Gemini to improve."
            )

        elif len(section_text.strip()) < 15:

            st.warning(
                "Please provide a little more content so Gemini can make a meaningful improvement."
            )

        else:

            with st.spinner(
                f"Gemini is improving your {section_name} section..."
            ):

                try:

                    improved_section = improve_resume_section(
                        section_name,
                        section_text
                    )

                    st.session_state["phase_32_result"] = improved_section
                    st.session_state["phase_32_error"] = None

                except Exception as exc:

                    st.session_state["phase_32_result"] = None
                    st.session_state["phase_32_error"] = str(exc)


# ==========================================
# PHASE 3.2 — RESULTS
# ==========================================

if st.session_state.get("phase_32_result"):

    st.markdown(
        '<div class="section-label">12 · AI improvement results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="visual-card">
            <div class="visual-title">
                ✦ Gemini's improved version
            </div>

            <div class="visual-subtitle">
                Your original information has been retained while the
                writing has been improved for clarity, impact, and ATS readability.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state["phase_32_result"]
    )


# ==========================================
# PHASE 3.2 — ERROR HANDLING
# ==========================================

if st.session_state.get("phase_32_error"):

    st.error(
        "The section could not be improved. "
        "Please check your Gemini configuration and try again."
    )

    with st.expander(
        "Technical error details",
        expanded=False
    ):

        st.code(
            st.session_state["phase_32_error"]
        )
    # ==========================================
# PHASE 3.3 — JD-AWARE OPTIMIZATION
# ==========================================

def optimize_section_for_jd(
    section_name,
    section_text,
    job_description,
    matching_keywords,
    missing_keywords,
    skill_data
):
    """
    Optimize one resume section specifically for the target job description.

    IMPORTANT:
    Gemini must only use facts that already exist in the candidate's
    resume section. It must never invent qualifications.
    """

    matched_skills_text = ", ".join(matching_keywords) if matching_keywords else "None detected"
    missing_skills_text = ", ".join(missing_keywords) if missing_keywords else "None detected"

    category_scores_text = ", ".join(
        f"{category}: {score}%"
        for category, score in skill_data.get("category_scores", {}).items()
    )

    prompt = f"""
You are an expert technical recruiter, ATS specialist, resume strategist,
and career coach.

Your task is to optimize ONE section of a candidate's resume specifically
for a TARGET JOB DESCRIPTION.

========================
SECTION
========================

Section name:
{section_name}

Original section:
{section_text[:10000]}

========================
TARGET JOB DESCRIPTION
========================

{job_description[:14000]}

========================
PHASE 2 INTELLIGENCE
========================

Matching skills already detected:
{matched_skills_text}

Missing skills detected:
{missing_skills_text}

Category-level skill alignment:
{category_scores_text if category_scores_text else "No category-level data available"}

========================
YOUR OBJECTIVE
========================

Rewrite the selected resume section so that it is:

- More relevant to the target role
- ATS-friendly
- Clear and concise
- Recruiter-friendly
- Focused on the most relevant existing experience
- Naturally aligned with terminology used in the job description

========================
STRICT FACTUAL RULES
========================

NEVER invent or assume:

- Skills
- Technologies
- Programming languages
- Tools
- Employers
- Job titles
- Projects
- Responsibilities
- Achievements
- Metrics
- Percentages
- Users
- Dataset sizes
- Business results
- Certifications
- Education
- Years of experience

You may ONLY use information supported by the ORIGINAL SECTION.

If the job description asks for a skill that is missing from the
original section, DO NOT add that skill to the optimized version.

Do not pretend the candidate has experience they do not demonstrate.

You may:

- Reorder information
- Improve sentence structure
- Replace weak wording with stronger wording
- Remove irrelevant wording
- Emphasize relevant information that is already present
- Use terminology from the job description ONLY when it accurately
  describes information already present in the original section
- Improve ATS readability

========================
IMPORTANT ATS RULE
========================

Do not keyword-stuff.

Use relevant keywords naturally and only when supported by the
candidate's original content.

========================
OUTPUT FORMAT
========================

Return Markdown with EXACTLY these headings:

## JD-Optimized Version

Provide the complete improved section.

## Why This Version Is Better

Give 4-6 concise bullet points explaining how the section was aligned
with the target job.

## Relevant JD Keywords

List the important job-description keywords that are genuinely supported
by the original resume section.

## Missing JD Keywords

List important JD keywords that are NOT supported by the original section.

Do NOT recommend pretending to have these skills.

## ATS Strategy

Give 3-5 practical suggestions for improving ATS alignment while
remaining truthful.

## Recruiter Impact

Give 2-3 sentences explaining how the optimized section would be
perceived by a recruiter.

Remember:

The goal is NOT to make the candidate look artificially qualified.

The goal is to present their EXISTING qualifications in the strongest,
most relevant, and most truthful way possible.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


# ==========================================
# PHASE 3.3 — UI
# ==========================================

st.markdown(
    '<div class="section-label">13 · JD-aware optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="visual-card">
        <div class="visual-title">✦ Optimize a section for this job</div>

        <div class="visual-subtitle">
            Connect your resume section with the target job description.
            Gemini will identify relevant JD requirements and strengthen
            your section without inventing qualifications.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------
# Section selection
# ------------------------------------------

jd_section_col1, jd_section_col2 = st.columns(
    [0.75, 1.25],
    gap="large"
)

with jd_section_col1:

    jd_section_name = st.selectbox(
        "Select resume section",
        [
            "Professional Summary",
            "Skills",
            "Experience",
            "Projects",
            "Education",
            "Certifications",
        ],
        key="phase_33_section_name"
    )


with jd_section_col2:

    st.markdown(
        """
        <div style="
            background:#F1EDF7;
            border:1px solid #DDD5EB;
            border-radius:12px;
            padding:0.85rem 1rem;
            color:#5F5868;
            font-size:0.82rem;
            line-height:1.5;
            margin-top:1.8rem;
        ">
            <strong style="color:#65558A;">✦ JD-aware mode</strong><br>
            Your target job description is automatically combined with
            Phase 2 skill intelligence to make the optimization more targeted.
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------
# Resume section input
# ------------------------------------------

jd_section_text = st.text_area(
    "Paste the resume section you want to optimize",
    height=230,
    placeholder=(
        "Paste the section exactly as it currently appears "
        "in your resume..."
    ),
    key="phase_33_section_text"
)


# ------------------------------------------
# Show current JD
# ------------------------------------------

with st.expander(
    "View target job description",
    expanded=False
):

    st.write(job_description if job_description else "No job description available.")


# ------------------------------------------
# Phase 2 intelligence preview
# ------------------------------------------

if job_description:

    preview_col1, preview_col2, preview_col3 = st.columns(3)

    with preview_col1:

        st.metric(
            "Matching Skills",
            len(matching_keywords)
        )

    with preview_col2:

        st.metric(
            "Missing Skills",
            len(missing_keywords)
        )

    with preview_col3:

        st.metric(
            "Skill Alignment",
            f"{skill_data['skill_alignment']:.0f}%"
        )


# ==========================================
# OPTIMIZE BUTTON
# ==========================================

if st.button(
    "✦ Optimize Section For This Job",
    key="phase_33_optimize_button"
):

    if not jd_section_text.strip():

        st.warning(
            "Please paste the resume section you want to optimize."
        )

    elif len(jd_section_text.strip()) < 15:

        st.warning(
            "Please provide a little more content so Gemini can "
            "make a meaningful JD-aware improvement."
        )

    elif not job_description.strip():

        st.warning(
            "Please add a job description before using JD-aware optimization."
        )

    else:

        with st.spinner(
            f"Gemini is optimizing your {jd_section_name} for this role..."
        ):

            try:

                jd_optimized_result = optimize_section_for_jd(
                    section_name=jd_section_name,
                    section_text=jd_section_text,
                    job_description=job_description,
                    matching_keywords=matching_keywords,
                    missing_keywords=missing_keywords,
                    skill_data=skill_data
                )

                st.session_state["phase_33_result"] = jd_optimized_result
                st.session_state["phase_33_error"] = None

            except Exception as exc:

                st.session_state["phase_33_result"] = None
                st.session_state["phase_33_error"] = str(exc)


# ==========================================
# PHASE 3.3 — RESULTS
# ==========================================

if st.session_state.get("phase_33_result"):

    st.markdown(
        '<div class="section-label">14 · JD optimization results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="visual-card">
            <div class="visual-title">
                ✦ JD-optimized section
            </div>

            <div class="visual-subtitle">
                Your section has been optimized around the target role
                while preserving the factual information you provided.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state["phase_33_result"]
    )


# ==========================================
# PHASE 3.3 — ERROR HANDLING
# ==========================================

if st.session_state.get("phase_33_error"):

    st.error(
        "JD-aware optimization could not be generated. "
        "Please check your Gemini configuration and try again."
    )

    with st.expander(
        "Technical error details",
        expanded=False
    ):

        st.code(
            st.session_state["phase_33_error"]
        )

    # ==========================================
# PHASE 3.4 — FULL RESUME OPTIMIZER
# ==========================================

def generate_full_resume_optimization(
    resume_text,
    job_description,
    gemini_analysis,
    matching_keywords,
    missing_keywords,
    skill_data,
):
    """
    Generate a complete JD-tailored resume using only factual
    information supported by the original resume.

    Phase 3.4 combines:
    - Full resume
    - Job description
    - Phase 2 skill intelligence
    - Phase 3.1 Gemini analysis
    """

    matched_skills_text = (
        ", ".join(matching_keywords)
        if matching_keywords
        else "None detected"
    )

    missing_skills_text = (
        ", ".join(missing_keywords)
        if missing_keywords
        else "None detected"
    )

    category_scores_text = ", ".join(
        f"{category}: {score}%"
        for category, score in skill_data.get(
            "category_scores", {}
        ).items()
    )

    gemini_analysis_text = (
        gemini_analysis
        if gemini_analysis
        else "Phase 3.1 Gemini analysis has not been generated."
    )

    prompt = f"""
You are an expert technical recruiter, ATS specialist,
professional resume writer, and career strategist.

Your task is to create a COMPLETE, JD-TAILORED version
of the candidate's resume.

==================================================
ORIGINAL RESUME
==================================================

{resume_text[:20000]}

==================================================
TARGET JOB DESCRIPTION
==================================================

{job_description[:14000]}

==================================================
PHASE 2 — MATCHING SKILLS
==================================================

{matched_skills_text}

==================================================
PHASE 2 — MISSING SKILLS
==================================================

{missing_skills_text}

==================================================
PHASE 2 — CATEGORY ALIGNMENT
==================================================

{category_scores_text if category_scores_text else "No category-level data available"}

==================================================
PHASE 3.1 — GEMINI RESUME ANALYSIS
==================================================

{gemini_analysis_text[:12000]}

==================================================
PRIMARY OBJECTIVE
==================================================

Create a stronger version of the candidate's EXISTING resume
that is specifically tailored to the target job description.

The optimized resume should:

- Be ATS-friendly
- Be recruiter-friendly
- Prioritize relevant experience
- Improve clarity
- Improve action language
- Improve keyword alignment
- Remove unnecessary wording
- Improve professional positioning
- Preserve the candidate's actual qualifications

==================================================
STRICT FACTUAL INTEGRITY RULES
==================================================

THIS IS EXTREMELY IMPORTANT.

NEVER invent or assume:

- Skills
- Technologies
- Programming languages
- Frameworks
- Tools
- Employers
- Job titles
- Job responsibilities
- Projects
- Certifications
- Education
- Degrees
- Dates
- Years of experience
- Achievements
- Metrics
- Percentages
- User counts
- Dataset sizes
- Business results
- Awards
- Publications

If something appears in the job description but is NOT
supported by the original resume, DO NOT add it.

For example:

If the JD says:

AWS, Docker, Kubernetes

but the resume does not demonstrate AWS, Docker, or Kubernetes,

DO NOT add those skills to the optimized resume.

Instead, mention them under:

"Keywords Not Added"

because they are not supported by the candidate's current resume.

==================================================
TRUTHFUL REWRITING RULE
==================================================

You MAY:

- Rewrite sentences
- Improve grammar
- Strengthen action verbs
- Reorder information
- Remove irrelevant information
- Improve formatting
- Emphasize relevant existing experience
- Combine repetitive information
- Use JD terminology when it accurately describes existing experience
- Make existing skills more visible
- Improve ATS readability

You MUST NOT:

- Create new qualifications
- Turn a missing skill into an existing skill
- Create fake achievements
- Add fake numbers
- Add fake metrics
- Add technologies merely because they appear in the JD

==================================================
ATS KEYWORD RULE
==================================================

Use job-description keywords naturally.

Do not keyword stuff.

Only incorporate a keyword into the optimized resume if
the original resume provides evidence supporting that keyword.

==================================================
RESUME STRUCTURE
==================================================

Create the optimized resume using these sections
where the information exists in the original resume:

## PROFESSIONAL SUMMARY

Write a concise 3-4 sentence summary tailored to the target role.

Only mention skills, technologies, education, projects,
and experience supported by the original resume.

## TECHNICAL SKILLS

Organize the candidate's existing skills into logical categories.

Do not add skills that are not supported by the resume.

Prioritize skills relevant to the target role.

## EXPERIENCE

Rewrite existing experience bullets.

Use strong action verbs.

Improve clarity and relevance.

Preserve all factual information.

Do not invent metrics.

## PROJECTS

Rewrite existing projects to highlight aspects relevant
to the target role.

Do not invent project functionality.

Do not invent technologies.

Do not invent results.

## EDUCATION

Clean and organize the existing education information.

Do not change degrees, institutions, dates, or grades.

## CERTIFICATIONS

Include only certifications actually present
in the original resume.

Do not create certifications.

==================================================
ADDITIONAL ANALYSIS
==================================================

After the optimized resume, provide:

## ATS KEYWORDS INCORPORATED

List the important JD keywords that were naturally
incorporated because they were supported by the resume.

## KEYWORDS NOT ADDED

List important JD keywords that were NOT added because
the original resume did not provide enough evidence.

Do NOT suggest pretending to have them.

## CHANGES MADE

Provide a clear Before → After explanation.

For example:

- Before: Weak wording
- After: Stronger action-oriented wording

Only describe changes that were actually made.

Give approximately 5-10 important changes.

## RECRUITER IMPACT

Explain in 3-5 sentences how the optimized resume
improves recruiter readability and JD alignment.

==================================================
FINAL QUALITY CHECK
==================================================

Before returning your answer, internally verify:

1. Every skill in the optimized resume exists in or is
   clearly supported by the original resume.

2. Every project exists in the original resume.

3. Every employer exists in the original resume.

4. No fake metrics were introduced.

5. No fake achievements were introduced.

6. No unsupported JD skills were added.

7. The resume is tailored to the JD.

8. The writing is professional and concise.

Return ONLY the structured Markdown result.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return response.text


# ==========================================
# PHASE 3.4 — UI
# ==========================================

st.markdown(
    '<div class="section-label">15 · Full resume optimization</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="visual-card">
        <div class="visual-title">
            ✦ Optimize My Entire Resume
        </div>

        <div class="visual-subtitle">
            Combine your resume, target job description, skill intelligence,
            and Gemini insights into one complete ATS-friendly resume.
            The optimizer preserves your existing qualifications and
            never invents experience.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# OPTIMIZATION READINESS
# ==========================================

readiness_col1, readiness_col2, readiness_col3 = st.columns(3)

with readiness_col1:
    st.metric(
        "Resume Loaded",
        "✓ Yes" if resume_text.strip() else "✕ No"
    )

with readiness_col2:
    st.metric(
        "Job Description",
        "✓ Added" if job_description.strip() else "✕ Missing"
    )

with readiness_col3:
    st.metric(
        "Gemini Analysis",
        "✓ Ready"
        if st.session_state.get("gemini_analysis")
        else "Optional"
    )


st.markdown(
    """
    <div style="
        background:#F1EDF7;
        border:1px solid #DDD5EB;
        border-radius:14px;
        padding:1rem 1.15rem;
        color:#5F5868;
        font-size:0.82rem;
        line-height:1.55;
        margin:1rem 0;
    ">
        <strong style="color:#65558A;">
            ✦ Truth-preserving optimization
        </strong><br>
        The AI can rewrite and reorganize your existing qualifications,
        but it will not add unsupported skills, experience, achievements,
        metrics, certifications, or technologies.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# MAIN OPTIMIZE BUTTON
# ==========================================

if st.button(
    "✦ Optimize My Resume",
    key="phase_34_optimize_button",
    use_container_width=True
):

    if not resume_text.strip():

        st.warning(
            "Please upload your resume before optimizing it."
        )

    elif not job_description.strip():

        st.warning(
            "Please add a job description before optimizing your resume."
        )

    else:

        with st.spinner(
            "Gemini is building your complete JD-tailored resume..."
        ):

            try:

                gemini_analysis_for_optimizer = (
                    st.session_state.get(
                        "gemini_analysis",
                        ""
                    )
                )

                full_resume_result = (
                    generate_full_resume_optimization(
                        resume_text=resume_text,
                        job_description=job_description,
                        gemini_analysis=gemini_analysis_for_optimizer,
                        matching_keywords=matching_keywords,
                        missing_keywords=missing_keywords,
                        skill_data=skill_data,
                    )
                )

                st.session_state[
                    "phase_34_result"
                ] = full_resume_result

                st.session_state[
                    "phase_34_error"
                ] = None

            except Exception as exc:

                st.session_state[
                    "phase_34_result"
                ] = None

                st.session_state[
                    "phase_34_error"
                ] = str(exc)


# ==========================================
# PHASE 3.4 — RESULTS
# ==========================================

if st.session_state.get("phase_34_result"):

    st.markdown(
        '<div class="section-label">16 · Optimized resume</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="visual-card">
            <div class="visual-title">
                ✦ Your optimized resume
            </div>

            <div class="visual-subtitle">
                A complete JD-tailored resume generated from your
                existing qualifications.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state["phase_34_result"]
    )


# ==========================================
# PHASE 3.4 — DOWNLOAD
# ==========================================

if st.session_state.get("phase_34_result"):

    st.markdown(
        '<div class="section-label">17 · Export</div>',
        unsafe_allow_html=True
    )

    optimized_resume_text = st.session_state[
        "phase_34_result"
    ]

    st.download_button(
        label="↓ Download Optimized Resume",
        data=optimized_resume_text,
        file_name="optimized_resume.md",
        mime="text/markdown",
        key="phase_34_download"
    )


# ==========================================
# PHASE 3.4 — ERROR HANDLING
# ==========================================

if st.session_state.get("phase_34_error"):

    st.error(
        "Full resume optimization could not be generated. "
        "Please check your Gemini configuration and try again."
    )

    with st.expander(
        "Technical error details",
        expanded=False
    ):

        st.code(
            st.session_state["phase_34_error"]
        )

    