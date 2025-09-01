# PVSCAN - AI-Powered Solar Panel Inspection & Analytics System

<p align="center">
  <img src="path/to/your/logo.png" alt="PVSCAN Logo" width="200"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python Version"></a>
  <a href="#"><img src="https://img.shields.io/badge/Backend-Flask-green.svg" alt="Flask Backend"></a>
  <a href="#"><img src="https://img.shields.io/badge/Frontend-Vue.js-brightgreen.svg" alt="Vue.js Frontend"></a>
  <a href="#"><img src="https://img.shields.io/badge/AI%20Model-PyTorch-orange.svg" alt="PyTorch Model"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/PRs-welcome-pink.svg" alt="PRs Welcome"></a>
</p>

<p align="center">
  An enterprise-grade web application designed to automate the inspection and efficiency analysis of solar panel installations using cutting-edge computer vision technology.
</p>

---

## 📋 Table of Contents

-   [Executive Summary](#-executive-summary)
-   [✨ Key Features](#-key-features)
-   [📸 Visual Tour](#-visual-tour)
-   [🏗️ System Architecture & Tech Stack](#️-system-architecture--tech-stack)
-   [📡 API Overview](#-api-overview)
-   [🚀 Getting Started](#-getting-started)
-   [☁️ Deployment Guide](#️-deployment-guide)
-   [🔮 Future Enhancements](#-future-enhancements)
-   [🤝 Contributing](#-contributing)
-   [📄 License](#-license)

## 📖 Executive Summary

PVSCAN is an AI-powered platform that modernizes the process of solar panel maintenance. Originally a basic Streamlit application, it has been completely re-engineered into a professional-grade system with a modern Vue.js frontend and a robust Flask backend API. The system processes single or batch image uploads to detect a wide range of defects with **98.5% model accuracy**, calculates fleet-wide efficiency scores, and generates prioritized maintenance recommendations. This tool transforms raw visual data into actionable intelligence, enabling operators to optimize performance, reduce costs, and maximize the efficiency of their solar assets.

## ✨ Key Features

### 🤖 AI-Powered Analysis
-   **High-Accuracy Defect Detection**: Utilizes a PyTorch-based MobileNetV3 model to identify conditions with 98.5% accuracy.
-   **Single & Batch Processing**: Analyze a single panel image or a ZIP archive of up to 100 images for fleet-wide insights.
-   **Comprehensive Condition Assessment**: Detects a wide range of issues including:
    -   Physical Damage (cracks, chips)
    -   Electrical Issues
    -   Environmental Factors (snow, water)
    -   Contamination (dirt, debris)
    -   Bird Interference

### 📊 Advanced Analytics Dashboard
-   **Efficiency Meter**: Assigns an overall condition score (0-100%) and a categorical rating (Excellent, Good, Average, Poor, Critical) to each panel.
-   **Key Performance Indicators (KPIs)**: Get an at-a-glance view of your fleet's health, including average efficiency score and the number of panels needing attention.
-   **Maintenance Prioritization**: Automatically identifies common issues and generates prioritized maintenance tasks (High, Medium, Low priority) based on severity and fleet impact.
-   **Data Export**: Download comprehensive analysis reports for documentation and offline review.

### 💻 Modern User Experience
-   **Intuitive Interface**: A clean, mobile-responsive UI with a modern glassmorphism design, following a "less is more" philosophy to provide relevant information in under 5 seconds.
-   **Drag-and-Drop Uploads**: Easily upload single images or batch ZIP files.
-   **Real-time Progress**: Track upload and analysis progress in real-time with clear feedback and error handling.

## 📸 Visual Tour



| Landing Page | Single Panel Analysis | Batch Fleet Analysis | Efficiency Dashboard |
| :---: | :---: | :---: | :---: |
| ![PVSCAN Landing Page](path/to/your/image1.png) | ![Single Analysis](path/to/your/image2.png) | ![Batch Fleet Analysis](path/to/your/image3.png) | ![Efficiency Dashboard](path/to/your/image4.png) |

## 🏗️ System Architecture & Tech Stack

The system is built on a modern, decoupled stack for scalability and maintainability.

| Component | Technology / Library |
| :--- | :--- |
| **Frontend** | Vue.js 3 (Composition API), Pinia, Vue Router, Axios, Custom CSS |
| **Backend** | Flask, Python 3.11+ |
| **AI/ML** | PyTorch (MobileNetV3) |
| **Image Processing** | Pillow, OpenCV-Python |
| **API** | RESTful with JSON responses |

```
PVSCAN/
    src/
        models/         
        routes/          
        templates/         
        static/        
        route/
        database/
    main.py
    utils.py
    __init__.py

    requirements.txt
```

## 📡 API Overview

The backend exposes a clean RESTful API for all core functionalities. The base URL is `/api`. All endpoints return consistent JSON error responses in case of failure.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Checks the system health and model availability. |
| `POST` | `/analyze-single` | Analyzes a single solar panel image for condition assessment. |
| `POST` | `/analyze-batch` | Processes a ZIP file containing multiple images for fleet analysis. |
| `POST` | `/efficiency-metrics` | Calculates advanced analytics and recommendations from batch results. |

For complete details on request/response models and status codes, please see the `API_DOCUMENTATION.md` file.

## 🚀 Getting Started

Follow these instructions to set up the project for local development.

### Prerequisites
-   Python 3.11+
-   Node.js 20+
-   A modern web browser

### 1. Backend Setup (Flask)
```bash
# Navigate to the backend directory
cd pvscan_backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Run the Flask server
# NOTE: Replace with the absolute path to your project's root backend folder.
PYTHONPATH=/path/to/pvscan_backend/pvscan_backend/ python3 src/main.py
````

The backend server will run on `http://localhost:5000`.

### 2\. Frontend Setup (Vue.js)

```bash
# Open a new terminal and navigate to the frontend directory
cd pvscan-frontend

# Install dependencies using npm
npm install

# Start the development server
npm run serve
```

The frontend application will be accessible at the address provided by the development server (e.g., `http://localhost:8080`).

## ☁️ Deployment Guide

The application is designed for flexible deployment in various environments.

- **🐳 Deploying with Docker**

A `Dockerfile` is provided for easy containerization.

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pvscan_backend/ .

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "src/main.py"]
```

Build and run the container:

```bash
docker build -t pvscan .
docker run -p 5000:5000 pvscan
```

- **🖥️ Deploying on a Traditional Server**

1.  **Server Requirements**: Ubuntu 20.04+, Python 3.11+, 4GB RAM.
2.  **Setup**: Clone the repo, install dependencies into a virtual environment, and use a production-grade WSGI server like Gunicorn.
3.  **Reverse Proxy**: It is recommended to use Nginx as a reverse proxy to handle incoming traffic.

- **🌐 Deploying on Cloud Platforms (Render, Heroku)**

  - **Render**:
      - **Build Command**: `pip install -r requirements.txt`
      - **Start Command**: `PYTHONPATH=/opt/render/project/src python3 src/main.py`
  - **Heroku**:
      - Create a `Procfile` in the root with: `web: python src/main.py`.
      - Deploy using the Heroku CLI and Git.




## 🔮 Future Enhancements

  - **Real-time Processing**: WebSocket integration for live progress updates.
  - **Advanced Analytics**: Machine learning-based trend prediction and historical tracking.
  - **Mobile Application**: Native mobile app for in-field inspections.
  - **Cloud Integration**: Deeper integration with cloud services (e.g., AWS S3 for storage).
  - **Enterprise Features**: Multi-tenant support, subscription management, and third-party integrations.

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

Please report any bugs or suggest features by opening an issue on the GitHub repository.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

-----

*Built with ❤️ For A Sustainable Future by Anjitesh Shandilya*

```