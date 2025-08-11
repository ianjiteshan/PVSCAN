# PVSCAN Deployment Guide

## Quick Start

### Local Development

1. **Start Backend Server**
   ```bash
   cd pvscan_backend
   source venv/bin/activate
   python src/main.py
   ```
   Server will run on `http://localhost:5000`

2. **Access Application**
   Open your browser and navigate to `http://localhost:5000`

### Production Deployment Options

#### Option 1: Traditional Server Deployment

1. **Server Requirements**
   - Ubuntu 20.04+ or similar Linux distribution
   - Python 3.11+
   - 4GB RAM minimum (8GB recommended)
   - 10GB storage space

2. **Installation Steps**
   ```bash
   # Clone or upload project files
   cd pvscan_backend
   
   # Install system dependencies
   sudo apt update
   sudo apt install python3-pip python3-venv nginx
   
   # Setup Python environment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   
   # Configure Nginx (optional)
   sudo nano /etc/nginx/sites-available/pvscan
   ```

3. **Nginx Configuration Example**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

#### Option 2: Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.11-slim
   
   WORKDIR /app
   COPY pvscan_backend/ .
   
   RUN pip install -r requirements.txt
   
   EXPOSE 5000
   CMD ["python", "src/main.py"]
   ```

2. **Build and Run**
   ```bash
   docker build -t pvscan .
   docker run -p 5000:5000 pvscan
   ```

#### Option 3: Cloud Platform Deployment

**Heroku Deployment**
```bash
# Install Heroku CLI
# Create Procfile in project root
echo "web: python src/main.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

**AWS/Azure/GCP**
- Use platform-specific deployment tools
- Configure environment variables
- Set up load balancing if needed

## Environment Configuration

### Required Environment Variables

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False

# File Upload Settings
MAX_CONTENT_LENGTH=50MB
UPLOAD_FOLDER=/tmp/uploads

# Model Configuration
MODEL_PATH_V1=models/pvscan_mobilenetv3_v1.1.pth
MODEL_PATH_V2=models/pvscan_mobilenetv3_v2.0.pth
```

### Security Settings

1. **CORS Configuration**
   - Update allowed origins in production
   - Restrict to specific domains

2. **File Upload Security**
   - Implement file type validation
   - Set appropriate size limits
   - Configure secure temporary directories

## Monitoring and Maintenance

### Health Checks

The application provides a health check endpoint:
```
GET /api/health
```

Response includes:
- System status
- Model availability
- Memory usage
- Uptime information

### Logging

Configure logging for production:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('pvscan.log'),
        logging.StreamHandler()
    ]
)
```

### Performance Optimization

1. **Model Loading**
   - Models are loaded once at startup
   - Consider model caching strategies

2. **File Processing**
   - Implement cleanup for temporary files
   - Monitor disk space usage

3. **Memory Management**
   - Monitor memory usage during batch processing
   - Implement request queuing if needed

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model files are present
   - Check file permissions
   - Ensure sufficient memory

2. **File Upload Issues**
   - Check file size limits
   - Verify upload directory permissions
   - Monitor disk space

3. **CORS Errors**
   - Update CORS configuration
   - Check allowed origins

### Debug Mode

Enable debug mode for development:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

**Warning**: Never enable debug mode in production!

## Scaling Considerations

### Horizontal Scaling

1. **Load Balancer Configuration**
   - Distribute requests across multiple instances
   - Implement session affinity if needed

2. **Database Integration**
   - Add database for result persistence
   - Implement user management

### Vertical Scaling

1. **Resource Allocation**
   - Increase memory for larger batch processing
   - Optimize CPU usage for model inference

2. **Caching Strategies**
   - Implement Redis for session management
   - Cache frequently accessed data

## Backup and Recovery

### Data Backup

1. **Model Files**
   - Backup trained model files
   - Version control for model updates

2. **Configuration**
   - Backup environment configurations
   - Document deployment procedures

### Recovery Procedures

1. **Service Recovery**
   - Automated restart procedures
   - Health check monitoring

2. **Data Recovery**
   - Restore from backups
   - Validate system functionality

## Support and Maintenance

### Regular Maintenance Tasks

1. **Log Rotation**
   - Configure log rotation policies
   - Monitor log file sizes

2. **Security Updates**
   - Regular dependency updates
   - Security patch management

3. **Performance Monitoring**
   - Monitor response times
   - Track resource usage

### Contact Information

For technical support and maintenance:
- System Administrator: [Contact Information]
- Development Team: [Contact Information]
- Emergency Contact: [Contact Information]

---

*This deployment guide provides comprehensive instructions for deploying the PVSCAN application in various environments, from local development to production cloud platforms.*

