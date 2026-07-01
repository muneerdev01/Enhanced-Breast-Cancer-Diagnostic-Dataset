# Enhanced Breast Cancer Diagnosis System - Deployment Guide

## Complete Setup & Deployment Instructions

### 📋 Table of Contents
1. [Local Setup](#local-setup)
2. [Streamlit Community Cloud](#streamlit-community-cloud)
3. [Render Deployment](#render-deployment)
4. [Hugging Face Spaces](#hugging-face-spaces)
5. [Troubleshooting](#troubleshooting)

---

## Local Setup

### Step 1: System Requirements
- Python 3.8 or higher
- 4GB RAM (8GB recommended)
- 500MB disk space
- Modern web browser

### Step 2: Installation

#### Windows
```bash
# Clone the repository
git clone <repository-url>
cd breast_cancer_diagnosis

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

#### macOS/Linux
```bash
# Clone the repository
git clone <repository-url>
cd breast_cancer_diagnosis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Step 3: Verify Installation
The app should automatically open at `http://localhost:8501`

If not, open your browser and navigate to that URL.

**Expected Output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## Streamlit Community Cloud

**Easiest cloud deployment option**

### Step 1: Prepare GitHub Repository
```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Enhanced Breast Cancer Diagnosis System - v1.0"

# Add remote repository
git remote add origin https://github.com/yourusername/breast_cancer_diagnosis.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Visit https://streamlit.io/cloud
2. Sign in with GitHub account (or create one)
3. Click "New app"
4. Select your repository and branch
5. Set main file path to: `app.py`
6. Click "Deploy"

### Step 3: Configuration (Optional)
Create `.streamlit/config.toml` in project root:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200

[client]
showErrorDetails = true
```

### Monitoring
- View deployment logs on Streamlit Cloud dashboard
- Monitor app usage and performance
- Manage secrets and environment variables

---

## Render Deployment

### Step 1: Connect GitHub Repository
1. Visit https://render.com
2. Sign up or log in
3. Click "Connect Repository"
4. Select your GitHub repository

### Step 2: Create Web Service
1. Click "New +"
2. Select "Web Service"
3. Choose the repository
4. Fill in configuration:

**Build Command**:
```bash
pip install -r requirements.txt
```

**Start Command**:
```bash
streamlit run app.py --server.port=10000 --server.address=0.0.0.0
```

### Step 3: Set Environment Variables
In Render dashboard, add:
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_MAXUPLOADSIZE=200
STREAMLIT_CLIENT_SHOWSTATETRACEBACK=false
```

### Step 4: Deploy
Click "Deploy" and wait for the build to complete (3-5 minutes).

**Your app will be at**: `https://your-app-name.onrender.com`

---

## Hugging Face Spaces

### Step 1: Create Hugging Face Account
Visit https://huggingface.co/join

### Step 2: Create Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose a name and select "Streamlit" as SDK
4. Select "Public" or "Private"
5. Create Space

### Step 3: Upload Files
Upload the following files to your Space:
- `app.py`
- `requirements.txt`
- `breast_cancer_ml_ready.csv`
- `README.md`
- `12_comprehensive_report.md`

### Step 4: Space Configuration
The app should auto-deploy when files are uploaded.

**Your app will be at**: `https://huggingface.co/spaces/yourusername/breast-cancer-diagnosis`

---

## Advanced Deployment Options

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY breast_cancer_ml_ready.csv .
COPY 12_comprehensive_report.md .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t breast_cancer_diagnosis .
docker run -p 8501:8501 breast_cancer_diagnosis
```

### AWS Deployment

Using AWS Elastic Beanstalk:
```bash
# Install EB CLI
pip install awsebcli

# Create EB app
eb init -p python-3.9 breast_cancer_diagnosis

# Create environment
eb create production

# Deploy
eb deploy
```

---

## Environment Variables

If deploying with authentication or custom settings:

### .env File (Local Development)
```
STREAMLIT_SERVER_HEADLESS=false
STREAMLIT_CLIENT_SHOWSTATETRACKBACK=true
STREAMLIT_LOGGER_LEVEL=info
```

### Production Variables
```
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_PORT=10000
STREAMLIT_LOGGER_LEVEL=warning
```

---

## Performance Optimization

### For Better Performance:

1. **Use Streamlit caching**:
```python
@st.cache_resource
def load_and_train_model():
    # Model loading code
```

2. **Optimize CSV files**:
- Use Parquet format instead of CSV for faster loading
- Compress data before deployment

3. **Limit data shown**:
- Show first 100 rows by default
- Allow users to expand if needed

4. **Lazy load resources**:
- Load models only when needed
- Defer heavy computations

---

## Monitoring & Maintenance

### Check Application Health
```bash
# Local
streamlit run app.py --logger.level=debug

# Cloud (check dashboard)
```

### Update Dependencies
```bash
pip list --outdated
pip install --upgrade <package-name>
```

### Common Issues & Solutions

#### Issue: "ModuleNotFoundError"
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

#### Issue: "Out of Memory"
```python
# Add to app.py start
import gc
gc.collect()
```

#### Issue: "SSL Certificate Error"
```bash
pip install --upgrade certifi
```

#### Issue: "CSV File Not Found"
- Verify file paths are relative
- Upload CSV to cloud deployment
- Check file naming matches exactly

### Logs & Debugging

**Local debugging**:
```bash
streamlit run app.py --logger.level=debug
```

**View Streamlit Cloud logs**:
1. Go to Streamlit Cloud dashboard
2. Click app name
3. View "Logs" tab

---

## Scaling Considerations

### For Small Scale (< 100 concurrent users):
- Streamlit Community Cloud or Render free tier
- Single instance deployment
- Basic monitoring

### For Medium Scale (100-1000 users):
- Render paid tier or AWS
- Load balancing
- Database for predictions logging
- Performance monitoring

### For Large Scale (> 1000 users):
- Kubernetes deployment
- Microservices architecture
- CDN for static assets
- API rate limiting
- Comprehensive monitoring

---

## Security Best Practices

1. **Data Privacy**:
   - No data stored on client side
   - No data transmission to third parties
   - HTTPS only for cloud deployment

2. **Model Security**:
   - Model files not exposed
   - Predictions logged locally
   - Input validation implemented

3. **Access Control**:
   - Use authentication if needed
   - Rate limiting for API
   - Regular security updates

4. **Compliance**:
   - HIPAA consideration (private deployment)
   - Data retention policies
   - Audit logging

---

## Testing Before Deployment

### Checklist:
- [ ] All dependencies installed
- [ ] App runs locally without errors
- [ ] All pages load correctly
- [ ] Predictions work with sample data
- [ ] CSV upload/download functions
- [ ] Visualizations render properly
- [ ] Model performance page displays metrics
- [ ] No console errors in browser
- [ ] Responsive design works on mobile
- [ ] Performance acceptable (< 2s load time)

### Test Predictions:
Use sample inputs:
```
radius_mean: 15.0
area_mean: 800.0
texture_mean: 20.0
concavity_mean: 0.1
... (other features)
```

Expected output: Prediction + confidence score

---

## Post-Deployment Monitoring

### Track Metrics:
- Number of predictions made
- Average prediction time
- Error rates
- User feedback

### Maintenance Tasks:
- Weekly log review
- Monthly performance audit
- Quarterly model retraining
- Annual security review

---

## Rollback Procedures

### If Issues After Deployment:

**Streamlit Cloud**:
```bash
# Roll back to previous commit
git revert <commit-hash>
git push
# Redeploy on Streamlit Cloud
```

**Render**:
- Go to Render dashboard
- Select deployment
- Click "Revert" to previous version

**Docker**:
```bash
# Use previous image version
docker run -p 8501:8501 breast_cancer_diagnosis:v1.0
```

---

## Support Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Render Docs**: https://render.com/docs/
- **Hugging Face**: https://huggingface.co/docs/
- **GitHub Issues**: Create issue in repository

---

## FAQ

**Q: Can I use custom domain?**  
A: Yes, on Render and AWS. Configure DNS settings in provider dashboard.

**Q: How much does deployment cost?**  
A: Streamlit Cloud is free. Render free tier has limitations. AWS pricing varies.

**Q: Can I add authentication?**  
A: Yes, use Streamlit-Authenticator or OAuth providers.

**Q: How to backup data?**  
A: Export predictions to CSV regularly, store in version control.

**Q: Can healthcare professionals use this?**  
A: Yes, as supplementary tool with proper disclaimers and oversight.

---

## Final Checklist

Before going live:
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Disclaimer visible
- [ ] Performance optimized
- [ ] Security verified
- [ ] Monitoring enabled
- [ ] Backup plan ready
- [ ] Support contact available

---

**Status**: Production Ready ✓  
**Version**: 1.0  
**Last Updated**: 2024
