import streamlit as st

# ---------- CONFIG ----------
NAME = "Rohit Bangalore Vijaya"
TAGLINE = "Graduate CS Student @ UMass Amherst | Ex - Cloud developer II @ HPE | NITK'22"

st.set_page_config(page_title=f"{NAME} - Portfolio", page_icon="🌟", layout="wide")

# ---------- CSS ----------
st.markdown("""
    <style>
    /* General */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9f9f9;
    }

    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        text-align: center;
        padding: 80px 20px;
        position: relative;
        margin-bottom: 60px;
        border-radius: 0 0 50px 50px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.2);
    }
    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: 15px;
    }
    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Section Titles */
    .section-title {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 50px;
        margin-bottom: 25px;
        color: #333;
        text-align: center;
        position: relative;
    }
    .section-title:after {
        content: '';
        display: block;
        width: 50px;
        height: 4px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        margin: 8px auto 0;
        border-radius: 2px;
    }

    /* About Me text */
    .about-text {
        max-width: 700px;
        margin: 0 auto 40px auto;
        text-align: center;
        font-size: 1.1rem;
        color: #444;
        line-height: 1.6;
    }

    /* Skills */
    .skills-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
    }
    .badge {
        padding: 10px 15px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 25px;
        font-size: 0.9rem;
        color: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        transition: transform 0.2s ease-in-out;
        white-space: nowrap;
    }
    .badge:hover {
        transform: translateY(-3px);
    }

    /* Project Tiles */
    .project-tile {
        background-color: white;
        padding: 25px 20px 20px 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        min-height: 180px;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .project-tile:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 30px rgba(0,0,0,0.1);
    }
    .project-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
    .project-desc {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 20px;
    }
    .project-link a {
        display: inline-block;
        padding: 8px 14px;
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-size: 0.9rem;
        transition: background 0.2s ease-in-out;
    }
    .project-link a:hover {
        opacity: 0.9;
    }

    /* Contact Section */
    .contact-links {
        text-align: center;
        font-size: 1.1rem;
        color: #333;
    }
    .contact-links a {
        color: #6a11cb;
        text-decoration: none;
        margin: 0 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        border-bottom: 2px solid transparent;
        padding-bottom: 2px;
    }
    .contact-links a:hover {
        color: #2575fc;
        border-bottom: 2px solid #2575fc;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- HERO SECTION ----------
st.markdown(f"""
    <div class="hero">
        <h1>{NAME}</h1>
        <p>{TAGLINE}</p>
    </div>
""", unsafe_allow_html=True)

# ---------- ABOUT ME ----------
st.markdown('<div class="section-title">About Me</div>', unsafe_allow_html=True)
about_me_text = """
I'm Rohit, a graduate student at University of Massachusetts Amherst. I am currently pursuing my masters in Computer Science. Prior to this I was working as a cloud developer at Hewlett Packard Enterprise. 
"""
st.markdown(f'<div class="about-text">{about_me_text}</div>', unsafe_allow_html=True)

# ---------- SKILLS ----------
st.markdown('<div class="section-title">Skills</div>', unsafe_allow_html=True)
skills = ["Python (Programming Language)", "Javascript", "Back-End Development", "Structured Query Language (SQL)", "Linux", "Large Language Models (LLMs)", "Tensorflow", "Keras", "Python Numpy", "Cloud Computing", "PostgreSQL", "Docker (Software)", "Kubernetes", "MongoDB", "Go Programming Language", "Git", "Java", "Neural Networks", "Deep Neural Networks (DNNS)", "Recurrent Neural Networks", "Reinforcement Learning", "GitHub", "RESTful APIs", "PyTorch", "Streamlit", "REACT", "Node.js"
]
badges_html = '<div class="skills-container">' + "".join(f'<span class="badge">{skill}</span>' for skill in skills) + '</div>'
st.markdown(badges_html, unsafe_allow_html=True)

# ---------- WORK EXPERIENCE ----------
st.markdown('<div class="section-title">Work Experience</div>', unsafe_allow_html=True)

work_experiences = [
    {
        "role": "Cloud Developer",
        "company": "Hewlett Packard Enterprise",
        "duration": "July 2022 - July 2025",
        "desc": "In my role as a backend developer on the HPE GreenLake platform, I was responsible for building and maintaining REST APIs to manage firmware, server, and OS upgrades at scale. A key project involved developing a cloud native solution which uses Natural Language Processing (NLP) techniques to infer firmware versions from installed component data, improving automation and accuracy. I also developed an automated profiling system to detect memory leaks early, helping shift performance issue identification to earlier stages in the development lifecycle. To enhance system resilience, I optimized the PostgreSQL database client by incorporating a circuit breaker pattern, enabling automatic retries during transient failures. Additionally, I implemented Horizontal Pod Autoscaling (HPA) for a critical microservice, which significantly improved scalability and ensured high availability under fluctuating workloads."
    },
    {
        "role": "Research Intern",
        "company": "Hewlett Packard Enterprise",
        "duration": "June 2021 - July 2021",
        "desc": "I worked on researching ways to optimize script to retrieve component versions from cloud rather than from on premises code. This ensured faster retrieval of versions improving user wait time and experience"
    },
]

for exp in work_experiences:
    st.markdown(f"""
        <div style="max-width:700px; margin: 0 auto 40px auto; padding: 15px 20px; background: white; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; font-size: 1.1rem; color: #6a11cb;">{exp['role']} <span style="font-weight: 500; color: #666;">@ {exp['company']}</span></div>
            <div style="font-size: 0.9rem; color: #999; margin-bottom: 10px;">{exp['duration']}</div>
            <div style="font-size: 1rem; color: #444;">{exp['desc']}</div>
        </div>
    """, unsafe_allow_html=True)


# ---------- PROJECTS ----------
st.markdown('<div class="section-title">Projects</div>', unsafe_allow_html=True)
projects = [
    {
        "name": "Retinal Vessel Segmentation",
        "desc": "A deep learning model to segment retinal vessels",
        "url": "https://github.com/rohit-b-v/Retinal-Vessel-Segmentation"
    },
    {
        "name": "Pansharpening of satellite images",
        "desc": "Development of automatic land cover change detection and analysis system from high-resolution remote sensing images",
        "url": "https://github.com/rohit-b-v/Pansharpening-of-satellite-images"
    },
    {
        "name": "Alivean Website",
        "desc": "Centralized healthcare website",
        "url": "https://github.com/rohit-b-v/Alivean-patient-website"
    },
    {
        "name": "Speech Enhancement",
        "desc": "Neural network approach to speech enhancement",
        "url": "https://github.com/rohit-b-v/speech-enhancement"
    },
    {
        "name": "Fast Fourier Transform using OpenMP and MPI",
        "desc": "Optimized FFT algorithm using parallel computing tools",
        "url": "https://github.com/rohit-b-v/Parallel-FFT"
    },
    {
        "name": "Ethernals",
        "desc": "decentralized Pinterest-like platform that allows users to discover, share, and support creators using blockchain technology",
        "url": "https://github.com/rohit-b-v/Ethernals"
    },
]

cols = st.columns(3)
for idx, project in enumerate(projects):
    with cols[idx % 3]:
        st.markdown(f"""
            <div class="project-tile">
                <div class="project-title">{project['name']}</div>
                <div class="project-desc">{project['desc']}</div>
                <div class="project-link"><a href="{project['url']}" target="_blank" rel="noopener noreferrer">View on GitHub</a></div>
            </div>
        """, unsafe_allow_html=True)

# ---------- CONTACT ----------
st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
st.markdown("""
<div class="contact-links">
📧 <a href="mailto:youremail@example.com">rohitbv.vips@gmail.com</a>  
💼 <a href="https://www.linkedin.com/in/rohitbv2012/" target="_blank" rel="noopener noreferrer">LinkedIn</a>  
🐙 <a href="https://github.com/rohit-b-v" target="_blank" rel="noopener noreferrer">GitHub</a>
</div>
""", unsafe_allow_html=True)
