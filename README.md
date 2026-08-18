# 🤖 AI Resume Intelligence Platform

An AI-powered Resume Intelligence Platform that analyzes resumes against target job descriptions and provides intelligent insights to improve job-readiness.



🌐 **Live Demo:**  
https://ai-resume-intelligence-platform-55.streamlit.app/

---

## 📌 Overview

The **AI Resume Intelligence Platform** is a Generative AI-powered application designed to help job seekers analyze and optimize their resumes for specific job opportunities.

Users can upload their resume, provide a target job description, and receive an intelligent analysis including:

- Resume-job match analysis
- ATS keyword analysis
- Missing skills and keywords
- Resume strengths and weaknesses
- AI-powered improvement suggestions
- Resume optimization recommendations
- Interview preparation questions
- Candidate readiness insights

The project demonstrates the practical application of **Generative AI, Natural Language Processing, resume intelligence, semantic analysis, and AI-driven recommendation systems**.

---

## 🚀 Live Demo

### 🌐 Try the Application

**https://ai-resume-intelligence-platform-55.streamlit.app/**

The application is deployed using **Streamlit Community Cloud** and can be accessed directly from a browser.

---

## ✨ Features

### 📄 Resume Upload & Analysis

- Upload resume PDF files
- Automatically extract resume text
- Process candidate information
- Analyze resume content using AI
- Identify important skills, technologies, and experience

---

### 🎯 Resume & Job Description Analysis

Users can provide a target job description and compare it with their resume.

The system analyzes:

- Relevant skills
- Technologies
- Experience
- Keywords
- Role requirements
- Overall alignment

This helps candidates understand whether their resume is suitable for a particular job.

---

### 📊 ATS Keyword Analysis

The platform performs ATS-style keyword analysis to identify how well the resume matches the target job description.

It identifies:

- Matching keywords
- Missing keywords
- Important technical skills
- Relevant tools and technologies
- Job-specific terminology

The analysis helps candidates improve their resumes for Applicant Tracking Systems (ATS).

---

### 🧠 AI-Powered Resume Intelligence

Google Gemini is used to generate intelligent insights from the resume and job description.

The AI can provide:

- Resume strengths
- Resume weaknesses
- Missing skills
- Improvement suggestions
- Optimization recommendations
- Candidate readiness insights

---

### ✍️ Resume Improvement Suggestions

The platform provides actionable recommendations to improve a resume.

Examples include:

- Improving resume bullet points
- Adding relevant keywords
- Highlighting important skills
- Improving descriptions of projects
- Making experience more relevant to the target role
- Removing unnecessary information

---

### 💼 Candidate Evaluation

The system evaluates the overall alignment between a candidate's resume and a target job.

It can provide insights into:

- Resume-job fit
- Skill alignment
- Missing requirements
- Candidate strengths
- Areas requiring improvement

---

### 🎤 Interview Preparation

The platform can generate interview preparation questions based on the candidate's resume and target role.

Question categories include:

- Technical questions
- Project-based questions
- Resume-based questions
- Behavioral questions
- Role-specific questions

This allows candidates to prepare for interviews using their own resume and target job description.

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Frontend / Application Framework

- Streamlit

### Generative AI

- Google Gemini
- Google GenAI SDK

### Natural Language Processing

- NLP
- Text Processing
- Keyword Matching
- Semantic Analysis

### Resume Processing

- PDF Text Extraction
- PyPDF

### Data Processing

- Python
- Pandas
- Scikit-learn

### Development Tools

- Git
- GitHub
- VS Code

### Deployment

- Streamlit Community Cloud

---

## 🏗️ System Architecture

The application follows the following workflow:

```text
                ┌─────────────────────┐
                │       User          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Upload Resume     │
                │       (PDF)         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Extract Resume Text │
                └──────────┬──────────┘
                           │
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌─────────────────────┐   ┌─────────────────────┐
   │   Resume Analysis   │   │  Job Description    │
   │                     │   │      Analysis       │
   └──────────┬──────────┘   └──────────┬──────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
                ┌─────────────────────┐
                │ ATS Keyword         │
                │ Analysis            │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ AI-Powered Gemini   │
                │ Analysis            │
                └──────────┬──────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ Match    │ │ Missing  │ │ Improvement  │
        │ Analysis │ │ Skills   │ │ Suggestions  │
        └──────────┘ └──────────┘ └──────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Interview           │
                │ Preparation         │
                └─────────────────────┘



#🔄 Application Workflow
User opens the Streamlit application.
User uploads a resume in PDF format.
Resume text is extracted automatically.
User provides a target job description.
Resume and job description are processed.
Relevant keywords and skills are identified.
ATS-style keyword matching is performed.
Resume-job alignment is analyzed.
Google Gemini generates AI-powered insights.
Missing skills and improvement areas are identified.
Resume optimization recommendations are displayed.
Interview preparation questions are generated.
The final analysis is presented through the Streamlit interface.


/
#📂 Project Structure
ai-resume-intelligence-platform/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation & Setup
1. Clone the Repository

git clone https://github.com/harikrupa-ai/ai-resume-intelligence-platform.git

2. Navigate to the Project
cd ai-resume-intelligence-platform


3. Create a Virtual Environment
python -m venv venv

4. Install Dependencies
pip install -r requirements.txt

5. Configure Gemini API Key

The application requires a Google Gemini API key.

For local development, configure the Streamlit secret:

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"

6. Run the Application
streamlit run main.py

🔐 Environment & API Security

The Gemini API key is stored securely using Streamlit Secrets.

The application accesses the key using:
st.secrets["GEMINI_API_KEY"]

🎓 Skills Demonstrated

This project demonstrates practical experience in:

Python Development
Generative AI
Google Gemini API
Prompt Engineering
Natural Language Processing
Resume Intelligence
Text Processing
ATS Keyword Analysis
Semantic Analysis
AI Recommendation Systems
Streamlit Application Development
API Integration
Git & GitHub
Cloud Deployment
AI Product Development



👩‍💻 Author
Saniya Chhabra

B.Tech — Artificial Intelligence & Data Science

Interested in:

Artificial Intelligence
Data Science
Machine Learning
Generative AI
NLP
AI Product Development



💻 GitHub Repository

https://github.com/Saniya0301/AI-Resume-Intelligence-platform


⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub!

📜 License

This project is intended for educational, portfolio, and demonstration purposes.
