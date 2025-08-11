# PVSCAN API Documentation

## Overview

The PVSCAN API provides endpoints for solar panel condition analysis using AI-powered computer vision. The API supports both single image analysis and batch processing of multiple images.

## Base URL

```
http://localhost:5000/api
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing API key authentication or OAuth 2.0.

## Content Types

- **Request**: `multipart/form-data` for file uploads, `application/json` for data
- **Response**: `application/json`

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error description",
  "status": "error",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Endpoints

### Health Check

Check the system status and model availability.

**Endpoint**: `GET /api/health`

**Response**:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "2.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes**:
- `200`: System is healthy
- `503`: System is unavailable

---

### Single Image Analysis

Analyze a single solar panel image for condition assessment.

**Endpoint**: `POST /api/analyze-single`

**Parameters**:
- `image` (file, required): Solar panel image file (JPG, PNG, WEBP)

**Request Example**:
```bash
curl -X POST \
  http://localhost:5000/api/analyze-single \
  -F "image=@panel_image.jpg"
```

**Response**:
```json
{
  "filename": "panel_image.jpg",
  "total_score": 85.2,
  "condition": "Good",
  "predictions": {
    "Panel Detected": 98.5,
    "Clean Panel": 85.2,
    "Physical Damage": 12.3,
    "Electrical Damage": 5.1,
    "Snow Covered": 2.0,
    "Water Obstruction": 1.5,
    "Contamination": 15.8,
    "Bird Interference": 3.2
  },
  "suggestions": [
    "Regular cleaning recommended to maintain efficiency",
    "Monitor for physical damage development",
    "Schedule routine electrical inspection"
  ],
  "analysis_time": 1.23,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes**:
- `200`: Analysis completed successfully
- `400`: Invalid file format or missing file
- `413`: File too large
- `500`: Analysis failed

---

### Batch Analysis

Process multiple solar panel images from a ZIP file.

**Endpoint**: `POST /api/analyze-batch`

**Parameters**:
- `zipfile` (file, required): ZIP file containing solar panel images

**Request Example**:
```bash
curl -X POST \
  http://localhost:5000/api/analyze-batch \
  -F "zipfile=@panel_images.zip"
```

**Response**:
```json
{
  "summary": {
    "total_images": 25,
    "successful_analyses": 24,
    "failed_analyses": 1,
    "average_score": 78.5,
    "processing_time": 45.2
  },
  "results": [
    {
      "filename": "panel_001.jpg",
      "total_score": 92.1,
      "condition": "Excellent",
      "predictions": {
        "Panel Detected": 99.2,
        "Clean Panel": 92.1,
        "Physical Damage": 3.2,
        "Electrical Damage": 1.1,
        "Snow Covered": 0.5,
        "Water Obstruction": 0.8,
        "Contamination": 8.9,
        "Bird Interference": 1.2
      },
      "suggestions": [
        "Panel in excellent condition",
        "Continue regular maintenance schedule"
      ]
    }
  ],
  "errors": [
    "panel_corrupt.jpg: Unable to process image file"
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes**:
- `200`: Batch analysis completed
- `400`: Invalid ZIP file or no valid images found
- `413`: File too large
- `500`: Processing failed

---

### Efficiency Metrics

Calculate comprehensive efficiency metrics from batch analysis results.

**Endpoint**: `POST /api/efficiency-metrics`

**Parameters**:
```json
{
  "results": [
    {
      "filename": "panel_001.jpg",
      "total_score": 92.1,
      "condition": "Excellent",
      "predictions": { ... }
    }
  ]
}
```

**Request Example**:
```bash
curl -X POST \
  http://localhost:5000/api/efficiency-metrics \
  -H "Content-Type: application/json" \
  -d '{"results": [...]}'
```

**Response**:
```json
{
  "total_panels": 25,
  "average_score": 78.5,
  "median_score": 82.0,
  "min_score": 45.2,
  "max_score": 96.8,
  "efficiency_rating": "Good",
  "score_distribution": {
    "excellent": {
      "count": 8,
      "percentage": 32.0
    },
    "good": {
      "count": 10,
      "percentage": 40.0
    },
    "average": {
      "count": 5,
      "percentage": 20.0
    },
    "poor": {
      "count": 2,
      "percentage": 8.0
    },
    "critical": {
      "count": 0,
      "percentage": 0.0
    }
  },
  "maintenance_priority": [
    {
      "issue": "Physical Damage",
      "affected_panels": 3,
      "percentage": 12.0,
      "priority": "High"
    },
    {
      "issue": "Contamination",
      "affected_panels": 8,
      "percentage": 32.0,
      "priority": "Medium"
    }
  ],
  "common_issues": {
    "contamination": 8,
    "physical_damage": 3,
    "electrical_damage": 1,
    "bird_interference": 2
  },
  "recommendations": [
    "Schedule cleaning for 8 contaminated panels",
    "Inspect 3 panels with physical damage",
    "Overall fleet performance is good"
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes**:
- `200`: Metrics calculated successfully
- `400`: Invalid input data
- `500`: Calculation failed

## Data Models

### Prediction Scores

All prediction scores are returned as percentages (0-100):

- **Panel Detected**: Confidence that a solar panel is present in the image
- **Clean Panel**: Cleanliness score (higher = cleaner)
- **Physical Damage**: Likelihood of physical damage (higher = more damage)
- **Electrical Damage**: Likelihood of electrical issues (higher = more issues)
- **Snow Covered**: Snow coverage percentage
- **Water Obstruction**: Water obstruction level
- **Contamination**: Contamination level (dirt, debris)
- **Bird Interference**: Bird-related interference level

### Condition Categories

- **Excellent**: 90-100% total score
- **Good**: 80-89% total score
- **Average**: 70-79% total score
- **Poor**: 60-69% total score
- **Critical**: 0-59% total score

### Priority Levels

- **High**: Immediate attention required
- **Medium**: Schedule maintenance within 30 days
- **Low**: Include in next routine maintenance

## Rate Limiting

Currently, no rate limiting is implemented. For production deployment, consider implementing:

- Request rate limits per IP address
- Concurrent processing limits
- File size and processing time limits

## File Constraints

### Single Image Analysis
- **Supported Formats**: JPG, JPEG, PNG, WEBP
- **Maximum File Size**: 10MB
- **Minimum Resolution**: 224x224 pixels
- **Maximum Resolution**: 4096x4096 pixels

### Batch Analysis
- **Supported Archive**: ZIP files only
- **Maximum Archive Size**: 100MB
- **Maximum Images per Archive**: 100
- **Individual Image Constraints**: Same as single image analysis

## Error Codes

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid input)
- **413**: Payload Too Large
- **415**: Unsupported Media Type
- **500**: Internal Server Error
- **503**: Service Unavailable

### Application Error Codes

```json
{
  "error_code": "INVALID_FILE_FORMAT",
  "error": "Unsupported file format. Please use JPG, PNG, or WEBP.",
  "status": "error"
}
```

Common error codes:
- `INVALID_FILE_FORMAT`: Unsupported file type
- `FILE_TOO_LARGE`: File exceeds size limit
- `NO_PANEL_DETECTED`: No solar panel found in image
- `PROCESSING_FAILED`: Analysis processing error
- `MODEL_UNAVAILABLE`: AI model not loaded

## SDK Examples

### Python Example

```python
import requests

# Single image analysis
with open('panel.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze-single',
        files={'image': f}
    )
    result = response.json()
    print(f"Score: {result['total_score']}%")

# Batch analysis
with open('panels.zip', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/analyze-batch',
        files={'zipfile': f}
    )
    result = response.json()
    print(f"Average Score: {result['summary']['average_score']}%")
```

### JavaScript Example

```javascript
// Single image analysis
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('/api/analyze-single', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Score:', data.total_score + '%');
});

// Batch analysis with progress tracking
const xhr = new XMLHttpRequest();
xhr.upload.addEventListener('progress', (e) => {
    const progress = (e.loaded / e.total) * 100;
    console.log('Upload progress:', progress + '%');
});

xhr.open('POST', '/api/analyze-batch');
xhr.send(formData);
```

## Changelog

### Version 2.0
- Added efficiency metrics endpoint
- Enhanced error handling
- Improved batch processing performance
- Added comprehensive analytics

### Version 1.1
- Initial API implementation
- Single and batch analysis endpoints
- Basic health check functionality

---

*This API documentation provides complete reference for integrating with the PVSCAN solar panel analysis system.*

