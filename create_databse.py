import getpass
import os
from dotenv import load_dotenv

load_dotenv()

import asyncio
import threading

if threading.current_thread().name == "ScriptRunner.scriptThread":
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

from langchain_chroma import Chroma

import chromadb

client = chromadb.PersistentClient(path="./langchain/chroma")

vector_store_from_client = Chroma(
    client=client,
    collection_name="records",
    embedding_function=embeddings,
)

from uuid import uuid4

from langchain_core.documents import Document

from collections import defaultdict

def create_database():
    documents = defaultdict(list)
    documents["Projects"] = []
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
Retinal vessel segmentation

This paper presents a deep learning model for retinal vessel segmentation using a task-driven Generative Adversarial Network (GAN). Retinal vessels provide crucial diagnostic information for various diseases, but manual segmentation is labor-intensive and often affected by image noise and low contrast. To address these issues, the proposed model automates the segmentation process using a GAN framework trained with perceptual loss to closely mimic expert-annotated segmentations.

The model architecture consists of a U-Net-based generator and three discriminators with varying receptive fields: a global discriminator and two multi-scale discriminators. These discriminators evaluate the quality of the generated segmentations by comparing them to manual annotations. The use of perceptual loss, computed from features extracted by a pre-trained LeNet model, ensures that the generated images preserve high-level structural similarity to the reference images. This combination allows the model to effectively highlight even the faintest retinal vessels.

Before training, input images undergo preprocessing steps including grayscale conversion, histogram equalization, and gamma adjustment, which improve image clarity and accelerate training. The model is trained and evaluated on two widely-used datasets: DRIVE, containing images from diabetic retinopathy screening, and STARE, which includes images with various lesions. These datasets allow the model to generalize across different retinal image conditions.

Results demonstrate that the proposed method surpasses traditional U-Net and GAN approaches in terms of vessel visibility and segmentation accuracy. It performs particularly well in challenging scenarios where vessels are thin or low in contrast. Although pixel-to-pixel accuracy is commonly used to evaluate segmentation models, the paper highlights its limitations and calls for the development of better performance metrics that account for structural correctness.

The paper concludes by emphasizing the potential of such models to significantly reduce the workload of medical professionals and improve diagnostic workflows. Future directions include creating models that can classify retinal images as healthy or defective, and refining evaluation methods to better compare segmentation architectures. Overall, this task-driven GAN approach proves to be a powerful tool for retinal image analysis, combining precision with automation in a clinically relevant setting.
        
        """,
        metadata={"source": "Projects"},
        id=3,
    )

    document_4 = Document(
        page_content="""
        Project
Development of Automatic Land Cover Change Detection and Analysis System from High-Resolution Remote Sensing Images

This project report focuses on developing an automatic land cover change detection and analysis system using high-resolution remote sensing images, with a specific emphasis on the technique of pansharpening. Pansharpening is a widely used image fusion technique aimed at enhancing the spatial resolution of multispectral satellite imagery by integrating it with high-resolution panchromatic images. A literature survey of various traditional and deep learning-based pansharpening methods highlights the evolution of this technique. Non-deep learning methods such as Brovey and ESRI transformations have been used historically, but often suffer from inefficiencies and inaccuracies in varying datasets. In contrast, modern approaches, including convolutional neural networks (CNNs) and autoencoders, have improved the accuracy and adaptability of pansharpening models, although some still struggle with effective feature extraction or image reconstruction in all regions.

The data used in this project consists of multispectral images with dimensions 512x512x3 and panchromatic images with dimensions 512x512. These images undergo preprocessing involving calculation of scaling factors, Gaussian blurring for interpolation, and spatial downsampling. The preprocessed images are split into smaller 64x64 patches for efficient model training. The proposed base model comprises three blocks: the first for extracting features from multispectral images, the second for capturing spatial features from panchromatic images, and the third for fusing outputs from both sources to produce a high-resolution pansharpened image. Each block is composed of convolutional and pooling layers, and the model is trained using Mean Squared Error loss with the Adam optimizer. The Swish activation function is employed to introduce non-linearity in the model.

A basic model was initially developed to verify the correctness of the preprocessing pipeline and training functions. This simpler network passes the 64x64x4 input image through multiple convolutional layers with decreasing filter sizes and ReLU activation, maintaining the same output dimensions. A more complex ResNet model was introduced next, which uses skip connections to address the vanishing gradient issue in deep networks. This model incorporates identity and convolutional blocks with varying numbers of filters, allowing the gradient to flow more freely and preserving important features during training.

The LagConv model offers a further innovation by enhancing the design of convolutional kernels. It introduces locally adaptive convolution (LAGConv) that dynamically adjusts the convolution weights for each pixel using a dot product between the kernel and a learned pixel-specific weight. This allows the model to capture both local and global image features more effectively and reduces distortions by integrating a global harmonic bias mechanism. The network comprises several LCA-ResBlocks, which combine the output of the LAGConv layers with the input through concatenation.

Another advanced approach explored is the Nested U-Net model integrated with a Convolutional Block Attention Module (CBAM). This attention mechanism helps the model to prioritize important image features over background noise. The multispectral and panchromatic images are processed separately, and their outputs are fused using a nested structure where each convolutional block is enhanced by CBAM. This setup allows the model to learn contextual importance at both channel and spatial levels, leading to improved pansharpening results.

The W-Net architecture is also implemented, consisting of two U-Net networks stacked together. The first U-Net acts as an encoder that generates segmentations of the input images, while the second U-Net decodes these segmentations back into high-resolution images. The encoder learns the segmentation map from unlabelled images, and the decoder reconstructs the image using upsampling and reduction in feature maps, maintaining the spatial resolution and improving segmentation fidelity.

In conclusion, the project successfully implements a four-layer CNN to validate preprocessing and baseline model design. Multiple advanced architectures including ResNet, LagConv, Nested U-Net with CBAM, and W-Net were explored to enhance pansharpening performance. The introduction of novel techniques such as LAGConv and attention modules allows the models to focus more precisely on relevant features, improving accuracy and visual quality compared to traditional statistical methods. Overall, the project demonstrates that deep learning-based approaches can significantly outperform conventional techniques in remote sensing image fusion, with promising implications for large-scale land cover change detection.
        """,
        metadata={"source": "Projects"},
        id=4,
    )

    document_5 = Document(
        page_content="""
        Project
SONG RECOMMENDATION BASED ON MOOD

This project focuses on developing a system that can recognize a person’s mood through their speech and recommend a suitable song based on that emotion. The core idea stems from the observation that people often choose music that aligns with how they feel—listening to upbeat or calming songs when happy, or slower, more emotional tracks when sad. However, given the vast number of songs available, it’s often hard to manually find the right one. To solve this, the system identifies the user’s mood through a spoken command like “Play a song” and suggests a random song from a database categorized by emotion.

The system utilizes speech emotion recognition (SER) to determine mood from voice tone. This is achieved using machine learning techniques that analyze audio features extracted from a dataset. The recommended song is selected from a MySQL database that organizes songs by emotional genre. The RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset serves as the training base for the machine learning model, which is built using the MLPClassifier from the Scikit-learn library. The project is implemented using Python in JupyterLab, leveraging additional libraries like Librosa, NumPy, PyAudio, and Soundfile. FFMPEG is used to ensure audio samples meet the necessary specifications for analysis.

The workflow begins with extracting features such as pitch and power from audio signals using Librosa. These features are then used to train the MLPClassifier model. Once trained, the system can record speech input from a user, detect the emotion behind it, and query a MySQL database for a relevant song. The audio input is captured and processed to ensure clarity and compatibility, including normalization, silence trimming, and appropriate sampling rates. The entire process is wrapped in a user interface that allows users to record audio and view the identified mood and recommended song.

Several Python scripts and components are used to build this system. The Utils.py file extracts features and defines emotions from the RAVDESS dataset. test.py captures user input through the microphone and prepares it for analysis, using the Utils.py functions. ser.py is responsible for training the machine learning model using the RAVDESS dataset. The convert_wavs.py script ensures all audio samples have the correct sampling rate by using FFMPEG to generate 16,000 Hz versions of the recordings.

The interactive notebook Interfaced.ipynb ties everything together. It provides a graphical interface for recording audio and displaying the final output—both the detected emotion and the suggested song. The trained model, saved as mlp_classifier.model, is loaded to make predictions without the need for retraining every time.

In summary, the project effectively integrates machine learning, speech processing, and database management to create a functional mood-based music recommendation system. It provides a seamless user experience by converting emotional speech into song suggestions, illustrating the practical potential of speech emotion recognition in personalized media applications.
        """,
        metadata={"source": "Projects"},
        id=5,
    )

    document_6 = Document(
        page_content="""
        Project
A Regression Approach to Speech Enhancement Based on Deep Neural Networks

This project focuses on enhancing the quality and intelligibility of speech signals that have been degraded by background noise. Speech enhancement plays a vital role in many real-world applications such as mobile communications, teleconferencing systems, speech recognition, and hearing aids. The central goal is to improve the clarity of speech in adverse acoustic environments where noise often distorts or masks the speaker’s voice. Traditional enhancement methods have limitations when dealing with complex or dynamic noise conditions, prompting the need for more robust and adaptive techniques.

The motivation behind this work stems from the shortcomings of conventional methods such as spectral subtraction and iterative Wiener filtering. These methods frequently produce artifacts known as "musical noise"—unnatural tones caused by inconsistent sub-band processing of random noise variations. Although approaches like the Minimum Mean Square Error (MMSE) technique reduce musical noise, they introduce a trade-off between speech distortion and residual noise. Moreover, such unsupervised techniques often struggle with non-stationary or unexpected noises in real-world settings, where noise patterns are less predictable.

To overcome these issues, this project explores the use of deep neural networks (DNNs) as a supervised learning approach for speech enhancement. The non-linear and flexible nature of DNNs makes them suitable for modeling the complex relationships between noisy and clean speech signals. The model is trained on a large and diverse dataset combining various speech samples and noise types to ensure robustness and generalization. Enhancements to the DNN framework include dropout for regularization, noise-aware training to handle a wider range of noise profiles, and global variance equalization to combat over-smoothing during regression.

The tools and datasets used in this project include Python libraries such as speechmetrics and python-pesq, MATLAB, Perl scripts, and the TIMIT dataset, which provides phonetically rich speech data. The Perceptual Evaluation of Speech Quality (PESQ) tool is employed to quantitatively assess the quality of speech before and after enhancement.

The overall methodology begins with preparing and mixing clean speech data with various noise types to simulate real-world scenarios. All features of these mixed audio signals are packed for input into the DNN model. A more complex DNN architecture is designed to enhance the model’s efficiency and capability. The model is trained to not only identify the type of noise present but also to generate a cleaner version of the speech signal. After enhancement, PESQ scores are calculated to measure quality improvements, and the enhanced audio signals are converted into spectrograms to visually analyze the results.

Through this approach, the project aims to create a reliable speech enhancement system that is effective in diverse and challenging acoustic environments, improving both subjective and objective measures of speech quality.
        
        """,
        metadata={"source": "Projects"},
        id=6,
    )

    document_7 = Document(
        page_content="""
        Project
ALIVEAN: The Healthcare Website

This project addresses several inefficiencies in the current healthcare system, which is fragmented across numerous private and government hospitals, each operating independently. Patients often face challenges when transitioning between facilities, such as difficulty in accessing or transferring medical records, repetitive appointment booking processes, and manual retrieval of past treatment documents. To streamline this experience, the project proposes a unified healthcare platform—a centralized website that aggregates services from multiple hospitals, offering patients a single portal to access doctors, schedule appointments, and manage their treatment history digitally.

The idea is to reduce the dependency on physical documentation by allowing patients to store all their medical records in one place. This ensures that doctors can easily access a patient's past treatments, making diagnosis and treatment more efficient. Moreover, the platform eliminates the need for patients to physically visit hospitals for appointments or record submission, thereby enhancing accessibility and convenience.

For the front-end development of the website, HTML, CSS, and JavaScript were used to create an interactive and user-friendly interface. On the back-end, the application is powered by Node.js and Express for server-side logic and routing. MongoDB serves as the database, storing user information, medical records, and appointment data. The online hosting of the database facilitates real-time collaboration and ensures accessibility across different systems.

The website features two main user modules—Patient and Doctor—each offering tailored functionalities. Patients can search for doctors based on hospital affiliation and location, request appointments via email, manage their personal profiles, and view their treatment history. Doctors, on the other hand, can access a list of patients they’ve treated, search for specific patients, and review treatment histories. They also have the ability to manage patient records by adding, updating, or deleting treatment details. A common landing page serves as an entry point, offering information about the platform and allowing users to register or log in based on their role.

The project is built using the Model-View-Controller (MVC) architectural pattern, which separates concerns and improves code organization. Express handles routing and dynamic content rendering, while MongoDB ensures robust and scalable data storage. Both client-side and server-side validations are implemented to ensure secure user authentication. Session handling mechanisms are used to maintain a personalized and persistent user experience across different pages.

Overall, the project aims to create a centralized, digital healthcare ecosystem that simplifies patient-doctor interactions and enhances the efficiency of medical services through streamlined data access and appointment management.
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

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
    ]
    

    vector_store_from_client.add_documents(documents=documents)
    results = vector_store_from_client.similarity_search("", k=3, filter={"source":"Projects"})
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    print(context_text)

if __name__=="__main__":
    create_database()
    
