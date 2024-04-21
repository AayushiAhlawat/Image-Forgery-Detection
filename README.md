# Image Forgery Detection


  <img src="https://img.shields.io/badge/Python-3.7-blue.svg" alt="Python">


## Overview 🔍

In the project "Image Forgery Detection," I developed a robust system aimed at detecting Copy-Move forgery, a commonly employed technique in image manipulation. 

### Objective 🎯

The purpose of choosing this project is:

- **Digital Images Forensics (DIF):** Vanguard of security techniques aiming at restoration of lost trust in digital imagery by exposing digital forgery techniques.
- **Existing Techniques:** Explore active and passive (blind) approaches in image forgery detection.
- **Validation:** Validate the originality of digital images by recovering information about their history.
- **Trust Building:** Analyze images under specific conditions to build trust and genuineness.

### Methodology 🛠️

The proposed system utilizes SVM classifier for forgery detection, employing hashing techniques and RSA key encryption for security. The methodology involves two main phases: training and testing.

1. **Training Phase:**
   - **Database Creation:** A database of images is created for training purposes. Images are sourced from various online repositories or captured using digital cameras. Images can vary in size and format (jpg, jpeg).
   - **RSA Key:** An RSA key is generated after training images are ingested into the system. During testing, users are prompted to enter a consistent key to ensure authorized access.
   - **Pre-processing:** Images undergo pre-processing steps such as conversion to grayscale from RGB, noise removal using median filtering, and enhancement techniques like histogram equalization and sharpening.
   - **Feature Extraction:** Various image features are extracted including:
     - **Pixel Analysis:** Calculation of mean and standard deviation of pixel values.
     - **Texture Analysis:** GLCM (Gray-Level Co-occurrence Matrix) analysis for texture representation using Haralick functions.
   - **Hash Values:** Hash values are computed for the extracted features to facilitate efficient comparison and identification of duplicated or manipulated regions within images.
   - **SVM Classifier:** Support Vector Machine (SVM) classifier is trained using labeled datasets to establish decision boundaries and identify fraudulent image regions with high precision.

2. **Testing Phase:**
   - **Input Query Image:** Users provide a query image to be authenticated.
   - **RSA Key Authentication:** Users are prompted to enter the consistent RSA key generated during training for authentication.
   - **Pre-processing and Feature Extraction:** Similar pre-processing and feature extraction steps are performed on the query image.
   - **Hash Values Calculation:** Hash values are computed for the extracted features of the query image.
   - **SVM Classification:** SVM classifier is utilized to classify the query image based on the decision boundaries established during training.

### Results 📊

- **High Accuracy:** Achieved a remarkable 95% accuracy rate in identifying forged images, showcasing the robustness and reliability of the detection algorithms.
- **Effective Detection:** Successfully detected instances of Copy-Move forgery, a challenging form of image manipulation commonly employed to deceive viewers.


## Getting Started 🚀

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/AayushiAhlawat/Image-Forgery-Detection.git`
2. Run the main script: `python Implementation.py`

### Conclusion 🎉

The Image Forgery Detection project demonstrates the effectiveness of machine learning techniques, specifically SVM classifiers, in identifying instances of digital image forgery. With a focus on Copy-Move forgery, the system achieves high accuracy rates and provides a reliable solution for image authenticity verification.

