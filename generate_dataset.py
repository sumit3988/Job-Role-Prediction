"""
Dataset Generator for Job Role Prediction Project
Generates a realistic synthetic dataset with 1200 rows.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
NUM_ROWS = 1200

DEGREES = ["B.Tech", "BCA", "MCA", "B.Sc", "M.Tech", "MBA", "B.E"]
MAJORS = ["Computer Science", "Information Technology", "Electronics", "Data Science"]

SKILLS_POOL = [
    "Python", "Java", "SQL", "AWS", "Machine Learning", "JavaScript",
    "React", "Node.js", "Docker", "Kubernetes", "TensorFlow", "Deep Learning",
    "C++", "PHP", "MongoDB", "Azure", "Linux", "Git"
]

CERTIFICATIONS = ["AWS", "Azure", "Google Cloud", "None", "Cisco", "PMP"]

JOB_ROLES = [
    "Data Scientist", "Backend Developer", "Frontend Developer",
    "DevOps Engineer", "Cloud Engineer", "AI Engineer", "Software Developer"
]

# Weighted skill sets per role to simulate realistic patterns
ROLE_SKILL_WEIGHTS = {
    "Data Scientist":       {"Python": 0.95, "SQL": 0.85, "Machine Learning": 0.90, "TensorFlow": 0.70, "Deep Learning": 0.65, "AWS": 0.40},
    "Backend Developer":    {"Java": 0.80, "Python": 0.60, "SQL": 0.80, "Node.js": 0.60, "MongoDB": 0.55, "Git": 0.75},
    "Frontend Developer":   {"JavaScript": 0.90, "React": 0.85, "PHP": 0.45, "Git": 0.80, "Node.js": 0.50},
    "DevOps Engineer":      {"Docker": 0.90, "Kubernetes": 0.85, "Linux": 0.90, "AWS": 0.70, "Azure": 0.65, "Git": 0.85, "Python": 0.55},
    "Cloud Engineer":       {"AWS": 0.90, "Azure": 0.85, "Kubernetes": 0.75, "Docker": 0.70, "Linux": 0.75},
    "AI Engineer":          {"Python": 0.95, "TensorFlow": 0.90, "Deep Learning": 0.90, "Machine Learning": 0.85, "C++": 0.50},
    "Software Developer":   {"Java": 0.70, "Python": 0.65, "C++": 0.60, "SQL": 0.65, "Git": 0.80, "JavaScript": 0.55},
}

ROLE_CERT_WEIGHTS = {
    "Data Scientist":       {"None": 0.30, "AWS": 0.25, "Azure": 0.20, "Google Cloud": 0.15, "Cisco": 0.05, "PMP": 0.05},
    "Backend Developer":    {"None": 0.40, "AWS": 0.20, "Azure": 0.15, "Google Cloud": 0.10, "Cisco": 0.10, "PMP": 0.05},
    "Frontend Developer":   {"None": 0.50, "AWS": 0.15, "Azure": 0.10, "Google Cloud": 0.10, "Cisco": 0.05, "PMP": 0.10},
    "DevOps Engineer":      {"AWS": 0.35, "Azure": 0.25, "Google Cloud": 0.15, "Cisco": 0.15, "None": 0.05, "PMP": 0.05},
    "Cloud Engineer":       {"AWS": 0.40, "Azure": 0.30, "Google Cloud": 0.20, "Cisco": 0.05, "None": 0.03, "PMP": 0.02},
    "AI Engineer":          {"None": 0.25, "AWS": 0.25, "Google Cloud": 0.25, "Azure": 0.15, "Cisco": 0.05, "PMP": 0.05},
    "Software Developer":   {"None": 0.45, "AWS": 0.20, "Azure": 0.15, "Google Cloud": 0.10, "Cisco": 0.05, "PMP": 0.05},
}

ROLE_DEGREE_WEIGHTS = {
    "Data Scientist":       {"M.Tech": 0.30, "B.Tech": 0.30, "MCA": 0.15, "B.Sc": 0.15, "BCA": 0.05, "MBA": 0.03, "B.E": 0.02},
    "Backend Developer":    {"B.Tech": 0.35, "MCA": 0.20, "BCA": 0.20, "B.Sc": 0.10, "M.Tech": 0.10, "B.E": 0.04, "MBA": 0.01},
    "Frontend Developer":   {"B.Tech": 0.30, "BCA": 0.30, "B.Sc": 0.20, "MCA": 0.10, "M.Tech": 0.05, "B.E": 0.04, "MBA": 0.01},
    "DevOps Engineer":      {"B.Tech": 0.40, "M.Tech": 0.20, "BCA": 0.15, "MCA": 0.15, "B.E": 0.07, "B.Sc": 0.02, "MBA": 0.01},
    "Cloud Engineer":       {"B.Tech": 0.40, "M.Tech": 0.25, "MCA": 0.15, "BCA": 0.10, "B.E": 0.07, "B.Sc": 0.02, "MBA": 0.01},
    "AI Engineer":          {"M.Tech": 0.40, "B.Tech": 0.35, "B.Sc": 0.10, "MCA": 0.10, "BCA": 0.03, "B.E": 0.01, "MBA": 0.01},
    "Software Developer":   {"B.Tech": 0.35, "BCA": 0.25, "MCA": 0.15, "B.Sc": 0.10, "M.Tech": 0.08, "B.E": 0.06, "MBA": 0.01},
}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def weighted_choice(options_weights: dict):
    choices = list(options_weights.keys())
    weights = list(options_weights.values())
    total = sum(weights)
    weights = [w / total for w in weights]
    return np.random.choice(choices, p=weights)


def generate_skills(role: str) -> str:
    skill_weights = ROLE_SKILL_WEIGHTS[role]
    selected = [s for s, p in skill_weights.items() if random.random() < p]
    # Add 0-2 random skills
    extra = random.sample([s for s in SKILLS_POOL if s not in selected], k=random.randint(0, 2))
    selected += extra
    if not selected:
        selected = [random.choice(SKILLS_POOL)]
    return ", ".join(sorted(set(selected)))


def generate_gpa(role: str) -> float:
    if role in ["Data Scientist", "AI Engineer"]:
        return round(np.random.normal(8.2, 0.6), 2)
    elif role in ["Cloud Engineer", "DevOps Engineer"]:
        return round(np.random.normal(7.9, 0.7), 2)
    else:
        return round(np.random.normal(7.5, 0.8), 2)


def generate_internships(role: str) -> int:
    if role in ["Data Scientist", "AI Engineer"]:
        return np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
    elif role in ["Frontend Developer", "Backend Developer"]:
        return np.random.choice([0, 1, 2, 3], p=[0.15, 0.4, 0.3, 0.15])
    else:
        return np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])


def generate_projects(role: str) -> int:
    base = {
        "Data Scientist": (4, 2),
        "AI Engineer": (5, 2),
        "Backend Developer": (3, 1),
        "Frontend Developer": (4, 2),
        "DevOps Engineer": (3, 1),
        "Cloud Engineer": (3, 1),
        "Software Developer": (3, 2),
    }
    mean, std = base[role]
    val = int(np.random.normal(mean, std))
    return max(1, min(10, val))


# ------------------------------------------------------------------
# Generate dataset
# ------------------------------------------------------------------

rows = []
for _ in range(NUM_ROWS):
    role = random.choice(JOB_ROLES)
    degree = weighted_choice(ROLE_DEGREE_WEIGHTS[role])
    major = weighted_choice({
        "Computer Science": 0.40,
        "Information Technology": 0.30,
        "Data Science": 0.20,
        "Electronics": 0.10,
    })
    skills = generate_skills(role)
    certification = weighted_choice(ROLE_CERT_WEIGHTS[role])
    gpa = generate_gpa(role)
    gpa = max(4.0, min(10.0, gpa))
    internships = generate_internships(role)
    projects = generate_projects(role)

    rows.append({
        "Degree": degree,
        "Major": major,
        "Skills": skills,
        "Certifications": certification,
        "GPA": gpa,
        "Internship_Experience": internships,
        "Number_of_Projects": projects,
        "Job_Role": role,
    })

df = pd.DataFrame(rows)

# Introduce ~3% missing values in non-target columns to simulate real data
for col in ["Certifications", "GPA", "Internship_Experience"]:
    mask = np.random.rand(len(df)) < 0.03
    df.loc[mask, col] = np.nan

df.to_csv("dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df.shape[1]} columns")
print(df["Job_Role"].value_counts())
