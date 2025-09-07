from collections import defaultdict

from langchain_core.documents import Document


documents=defaultdict(list)

def create_documents():

    document_1 = Document(
            page_content="""
            You the LLM agent were built using streamlit and langchain. The LLM used is google gemini-2.5-flash. You use Chroma DB to persist and store information to provide answers to users questions. 
            """,
            metadata={"source": "About you"},
            id=1,
        )

    document_2 = Document(
            page_content="""
            EDUCATION 
    University of Massachusetts Amherst, Amherst, MA 
    Master of Science in Computer Science (Expected 2027) 
    Pursuing M.S. in Computer Science with focus on AI, systems, backend engineering, and distributed computing. 
    courses taken for this fall - Advanced Natural Language Processing (CS-689), Machine Learning (CS-685) and systems for data science (CS-532)
    
    National Institute of Technology Karnataka (NITK), Surathkal, India 
    Bachelor of Technology, Major in Electronics and Communication Engineering; Minor in Information Technology | 
    2018 - 2022 
    CGPA: Major CGPA 8.41/10 | Minor CGPA 9/10
            """,
            metadata={"source": "Education"},
            id=2,
        )

    document_3 = Document(
            page_content="""
            Project
    RETINAL VESSEL SEGMENTATION


    This paper presents a deep learning model for retinal vessel segmentation using a task-driven Generative Adversarial Network (GAN). Retinal vessels are crucial for diagnosing various diseases, but manual segmentation is labor-intensive and affected by image noise and low contrast. To address this, the model automates segmentation using a U-Net-based generator and three discriminators—one global and two multi-scale—trained with perceptual loss from a pre-trained LeNet. This ensures that the generated segmentations preserve high-level structural similarity to expert annotations, effectively highlighting even faint vessels. Input images are preprocessed with grayscale conversion, histogram equalization, and gamma adjustment to enhance clarity and accelerate training, and the model is evaluated on the DRIVE and STARE datasets to ensure generalization across diverse retinal images.

Results show that this approach outperforms traditional U-Net and GAN methods, particularly in scenarios with thin or low-contrast vessels. The paper notes limitations of pixel-wise accuracy metrics and advocates for measures that account for structural correctness. The study concludes that task-driven GANs can significantly reduce medical professionals’ workload and improve diagnostic workflows. Future work includes extending the model to classify images as healthy or defective and refining evaluation methods, demonstrating the potential of this approach as a precise, automated tool for retinal image analysis.
            
            """,
            metadata={"source": "Projects"},
            id=3,
        )

    document_4 = Document(
            page_content="""
            Project
    DEVELOPMENT OF AUTOMATIC LAND COVER CHANGE DETECTION AND ANALYSIS SYSTEM FROM HIGH-RESOLUTION REMOTE SENSING IMAGES


    This project focuses on developing an automatic land cover change detection system using high-resolution remote sensing images, emphasizing pansharpening to enhance the spatial resolution of multispectral imagery with high-resolution panchromatic images. A literature review shows that traditional methods like Brovey and ESRI transformations often lack accuracy, while modern deep learning approaches, including CNNs and autoencoders, improve performance but still face challenges in feature extraction and reconstruction. The dataset includes 512×512 multispectral and panchromatic images, preprocessed with scaling, Gaussian blurring, and downsampling, then divided into 64×64 patches for model training. The base model extracts features from multispectral and panchromatic images and fuses them through convolutional and pooling layers, trained using MSE loss and the Adam optimizer, with Swish activation to introduce non-linearity.

Advanced architectures were also explored, including ResNet with skip connections, LagConv with locally adaptive kernels and harmonic bias, Nested U-Net with CBAM attention modules, and W-Net for encoder-decoder image reconstruction. These models improve feature extraction, focus on relevant spatial and channel information, and reduce distortions, resulting in higher pansharpening accuracy and visual quality. The project demonstrates that deep learning-based approaches can significantly outperform traditional statistical methods in remote sensing image fusion, offering strong potential for scalable and accurate land cover change detection.
            """,
            metadata={"source": "Projects"},
            id=4,
        )

    document_5 = Document(
            page_content="""
            Project
    SONG RECOMMENDATION BASED ON MOOD

    This project develops a system that recognizes a user’s mood from speech and recommends a song aligned with that emotion. People naturally choose music based on how they feel, but manually finding a suitable track is challenging. The system captures a spoken command, uses speech emotion recognition (SER) to detect mood, and queries a MySQL database of songs categorized by emotion. The RAVDESS dataset trains an MLPClassifier model built with Scikit-learn, while Python libraries like Librosa, NumPy, PyAudio, and Soundfile handle audio processing, and FFMPEG ensures proper sample rates. Audio features such as pitch and power are extracted, normalized, and used to train the model for accurate emotion detection.

The workflow allows users to record speech, detect emotion, and receive song recommendations through an intuitive interface. Python scripts handle feature extraction, model training, audio conversion, and real-time input processing, while the notebook Interfaced.ipynb integrates all components. The trained MLPClassifier model is saved and reused for predictions, providing seamless recommendations without retraining. Overall, the project demonstrates the practical potential of combining machine learning, speech processing, and database management to create a functional, user-friendly mood-based music recommendation system.
            """,
            metadata={"source": "Projects"},
            id=5,
        )

    document_6 = Document(
            page_content="""
            Project
    A REGRESSION APPROACH TO SPEECH ENHANCEMENT BASED ON DEEP NEURAL NETWORKS



    This project focuses on enhancing the clarity and intelligibility of speech signals degraded by background noise, which is critical for applications like mobile communications, teleconferencing, speech recognition, and hearing aids. Traditional methods such as spectral subtraction, Wiener filtering, and MMSE often produce artifacts or struggle with non-stationary noises. To overcome these limitations, the project employs deep neural networks (DNNs) to model the complex relationships between noisy and clean speech. The DNN is trained on diverse speech and noise datasets, incorporating dropout for regularization, noise-aware training, and global variance equalization to improve robustness and reduce over-smoothing.

The methodology involves mixing clean speech with various noise types, extracting features, and inputting them into the DNN to generate enhanced speech signals. Tools such as Python libraries, MATLAB, and the TIMIT dataset are used, and Perceptual Evaluation of Speech Quality (PESQ) scores assess improvements. The enhanced audio is also visualized via spectrograms. Overall, this approach produces a reliable, adaptable speech enhancement system that significantly improves both objective and subjective measures of speech quality in challenging acoustic environments.
            
            """,
            metadata={"source": "Projects"},
            id=6,
        )

    document_7 = Document(
            page_content="""
            Project
    ALIVEAN: THE HEALTHCARE WEBSITE

    This project addresses inefficiencies in the fragmented healthcare system by creating a centralized platform that aggregates services from multiple hospitals. Patients often face difficulties accessing or transferring medical records, booking appointments, and retrieving past treatment documents. The proposed website allows patients to store all medical records digitally, access doctors, schedule appointments, and manage treatment histories in one place. This reduces reliance on physical documentation and enhances convenience and accessibility, enabling doctors to efficiently review patient histories for better diagnosis and treatment.

The platform features two user modules—Patient and Doctor—each with tailored functionalities. Patients can search for doctors, request appointments, and view treatment histories, while doctors can manage patient records and access past treatments. Built using HTML, CSS, JavaScript for the front end, Node.js and Express for back-end logic, and MongoDB for scalable storage, the system follows the MVC architectural pattern with client- and server-side validations and session handling. Overall, it streamlines patient-doctor interactions and improves healthcare service efficiency through digital data management.
            """,
            metadata={"source": "Projects"},
            id=7,
        )

    document_10 = Document(
            page_content="""
            Project
    Ethernals
    A Blockchain-based Pinterest for Supporting Creators

 Overview
Ethernals is a decentralized, Pinterest-like platform that enables users to discover, share, and support creators using blockchain technology. By leveraging blockchain, the platform ensures secure content ownership, transparent tipping, and immutable creator credits, empowering creators while allowing users to directly support their favorite content. Users can explore image boards, upload content, and engage with creators in a secure and fair environment, reducing the risk of plagiarism or content misuse.

The platform operates with a tech stack including React.js/Next.js for the frontend, Node.js and Express for the backend, and Ethereum smart contracts written in Solidity for handling tipping, ownership, and attribution. Creators upload images stored on IPFS for decentralization, and users sign in via crypto wallets like MetaMask. Smart contracts facilitate direct crypto payments to creators, enabling a transparent, blockchain-powered content-sharing ecosystem.
            """,
            metadata={"source": "Projects"},
            id=7,
        )

    document_8 = Document(
            page_content="""
            Work EXPERIENCE 
    Cloud Developer II, Hewlett Packard Enterprise, Bangalore                                                              
    2025 
    July 2022 - July 
    ● Pioneered a cloud-native application to detect firmware versions using an NLP classifier model, reducing 
    customer escalations by 55% during software updates. 
    ● Enhanced microservice reliability through Test-Driven Development (TDD) and automated memory leak 
    detection with runtime profiling metrics; reduced manual diagnostics by 10% weekly. 
    ● Developed and maintained backend services for hybrid cloud platforms, designing and implementing REST 
    APIs to orchestrate and manage upgrades for storage OS, ESXi hypervisors, and VM managers. 
    ● Coded an automated artifact extraction script for Linux-based virtual machines to run during runtime, 
    reducing manual extraction-related escalations by 80% and significantly enhancing user experience. 
    ● Designed and implemented Horizontal Pod Autoscaling (HPA) for Kubernetes-based services, reducing 
    resource costs by 40% and improving load scalability. 
    ● Improved PostgreSQL client performance by applying the circuit breaker pattern, reducing database 
    latency by 30% and increasing application stability. 
    ● Built a Kafka-based event processing system to track artifact downloads, reducing weekly support requests 
    by 15% through better user feedback. 
    ● Created Grafana dashboards using Prometheus metrics to visualize software upgrade trends across 1,000+ 
    systems and monitor Horizontal Pod Autoscaling (HPA) behavior using CPU and memory utilization 
    patterns. 
    ● Mentored new hires and led onboarding sessions, fostering team cohesion and accelerating ramp-up 
    times.
            """,
            metadata={"source": "Work Experience"},
            id=8,
        )
    
    document_9 = Document(
        page_content="""
        SKILLS
        Programming Languages: Python, GoLang, Java, JavaScript, C 
Operating Systems: Linux/Unix, Windows 
Databases: PostgreSQL, SQL, Firestore, MongoDB 
Tools & Frameworks: Git, GitHub, Jenkins, Docker, Kubernetes, Prometheus, Grafana, Keras, TensorFlow 
Software Architecture & Practices: Microservices, REST APIs, Object-Oriented Design, System Design, 
Agile, Scrum, Test-Driven Development (TDD), Full stack development. 
        """
    )
    
    document_11 = Document(
            page_content="""
            📌 Parallel-FFT
FFT is an improved algorithm for evalulating Fourier transform of signals, which otherwise had higher time complexity due to the number of heavy computations involved in it. FFT algorithm can be further sped up by integrating Parallel Computing tools into it. For the demonstration of the Parallel FFT, the well known Cooley and Tukey algorithm is used on the input data array and implemented as a parallel non-recursive version of FFT using OPENMP and MPI.

✨ Why This Project?
This project was built to demonstrate practical skills in:

🖥️ Programming: C++
🛠️ Tools: OpenMP, MPI
Screen shots
image image
🛠️ Tech Stack
Languages: C++
Other Tools: MPI, OpenMP
🚀 Features
On running Cooley-Tukey algorithm on OpenMP and MPI there is an improvement in the execution time by 4 times when compared to serial execution.
OpenMP was more efficient in parallelising the algorithm than MPI.
There was approximately 2 times decrease in execution time when using OpenMP in comparison with MPI.
            """,
            metadata={"source": "Projects"},
            id=7,
        )
    

    documents["projects"].append(document_3)
    documents["projects"].append(document_4)
    documents["projects"].append(document_5)
    documents["projects"].append(document_6)
    documents["projects"].append(document_7)
    documents["projects"].append(document_10)
    documents["projects"].append(document_11)
    documents["education"].append(document_2)
    documents["about"].append(document_1)
    documents["work"].append(document_8)
    documents["skills"].append(document_9)


def get_documents(s):
    s.lower()
    print("returning these docs" ,s, documents[s])
    return documents[s]

def get_all_document():
    ret = []
    for _,v in documents.items():
        for i in v:
            ret.append(i)
    return ret
        