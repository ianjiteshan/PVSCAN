# SPICE.AI
## Solar Panel Inspection & Classification Engine

![SPICE.AI Logo](https://github.com/kushalgupta1203/SPICE.AI/blob/main/deployment/logo_phone.png?raw=true)

## Live Demo

https://spice-ai.streamlit.app/

## Overview

SPICE.AI (Solar Panel Inspection & Classification Engine) is an AI-powered application designed to automate the inspection and classification of solar panel conditions. Using advanced computer vision and deep learning techniques, the system can detect various issues like physical damage, electrical damage, snow coverage, water obstruction, foreign particle contamination, and bird interference that may affect solar panel performance.

## Features

- **Automated Solar Panel Detection**: Verifies the presence of solar panels in uploaded images
- **Multi-Class Classification**: Identifies multiple types of defects and obstructions
- **Severity Scoring**: Provides quantitative assessment of panel condition
- **Cleaning & Maintenance Recommendations**: Offers actionable suggestions based on detected issues
- **Responsive Web Interface**: Works seamlessly across desktop and mobile devices

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI Models**: PyTorch (MobileNetV3)
- **Deployment**: GitHub

## Model Versions

The application uses three different trained models:
- **Panel Detection Model (v2.0)**: For identifying the presence of solar panels
- **Inspection Model (v1.1)**: For defect classification
- **Inspection Model (v2.0)**: Enhanced defect classification

## Installation

### Prerequisites
- Python 3.8+
- Pip

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/SPICE.AI.git
cd SPICE.AI
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run deployment/app.py
```

## Usage

1. Open the application in your web browser
2. Upload an image of a solar panel using the file uploader
3. Wait for panel detection confirmation
4. Review the inspection analysis and severity scores
5. Follow the provided maintenance and cleaning suggestions

## Classification Categories

- **Clean Panel**: Solar panel in optimal condition
- **Physical Damage**: Cracks, broken glass, or frame damage
- **Electrical Damage**: Hot spots, discoloration, or burn marks
- **Snow Covered**: Snow accumulation on panel surface
- **Water Obstruction**: Water spots or pooling
- **Foreign Particle Contamination**: Dust, dirt, leaves, or other debris
- **Bird Interference**: Bird droppings or nesting materials

## Performance

- **Train Accuracy:** 95.84%, **Train Loss:** 0.0946
- **Validation Accuracy:** 95.50%, **Validation Loss:** 0.1062
- **Test Accuracy:** 95.39%, **Test Loss:** 0.1081