# 🚀 Deployment Guide - EEGSpeech Healthcare

This guide provides comprehensive deployment options for sharing your EEG Healthcare VLM project.

## 📋 Quick Reference

| Method | Difficulty | Cost | Best For |
|--------|-----------|------|----------|
| **Streamlit Cloud** | Easy | Free | Demos, prototypes |
| **Docker** | Medium | Free/Paid | Production, scaling |
| **Heroku** | Easy | Free tier | Quick sharing |
| **Railway** | Easy | Free tier | Auto-deployment |
| **AWS/GCP** | Hard | Paid | Enterprise |

## 🌐 Online Deployment (Sharing)

### 1. **Streamlit Cloud** ⭐ (Recommended for sharing)

**Pros**: Free, easy, automatic updates from GitHub
**Cons**: Limited resources, public by default

**Steps**:
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `eegspeech/app/healthcare_vlm_app.py` as main file
5. Deploy!

**Your app will be available at**: `https://[your-app-name].streamlit.app`

### 2. **Heroku** 

**Pros**: Easy deployment, free tier available
**Cons**: Limited free tier, can be slow

```bash
# Install Heroku CLI first
heroku login
heroku create your-eeg-app-name
heroku stack:set container
heroku config:set STREAMLIT_SERVER_PORT=8501
git push heroku main
```

### 3. **Railway**

**Pros**: Modern deployment, GitHub integration
**Cons**: Limited free tier

```bash
# Connect your GitHub repo to Railway
# railway.toml config file already included
```

### 4. **Docker Hub** (For distribution)

```bash
# Build and push to Docker Hub
docker build -t yourusername/eegspeech-healthcare .
docker push yourusername/eegspeech-healthcare

# Others can run with:
docker run -p 8501:8501 yourusername/eegspeech-healthcare
```

## 💻 Local Deployment (For others to use)

### 1. **One-Click Setup Script**

Create `setup.sh` (already included):
```bash
#!/bin/bash
curl -sSL https://raw.githubusercontent.com/tirth8205/EEGSpeech-Healthcare/main/setup.sh | bash
```

### 2. **Docker Compose** (Easiest for users)

```bash
# User runs this:
git clone https://github.com/tirth8205/EEGSpeech-Healthcare.git
cd EEGSpeech-Healthcare
docker-compose up
```

### 3. **Python Package Installation**

```bash
# For advanced users
pip install git+https://github.com/tirth8205/EEGSpeech-Healthcare.git
eegspeech-healthcare-app  # If you add this command
```

## 🔧 Configuration Files

### Streamlit Cloud Configuration (`.streamlit/config.toml`)
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Docker Configuration (`Dockerfile`)
- Multi-stage build for smaller image
- Health checks included
- Proper dependency management

### Docker Compose (`docker-compose.yml`)
- Volume mounting for data persistence
- Environment variables
- Resource limits

## 📱 Mobile-Friendly Deployment

### Responsive Design
- Uses Streamlit's responsive components
- Mobile-optimized layouts
- Touch-friendly interactions

### PWA Support (Progressive Web App)
Add to your Streamlit app:
```python
# In your main app
st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#1f77b4">
""", unsafe_allow_html=True)
```

## 🔒 Security Considerations

### For Public Deployment
- Remove sensitive API keys
- Add authentication if needed
- Use HTTPS (automatically handled by platforms)
- Implement rate limiting

### For Healthcare Data
- Ensure HIPAA compliance
- Use encrypted connections
- Implement audit logging
- Add data anonymization

## 📊 Monitoring & Analytics

### Basic Monitoring
```python
# Add to your Streamlit app
import streamlit as st

# Usage tracking
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
st.session_state.usage_count += 1
```

### Advanced Monitoring
- Use Streamlit's built-in analytics
- Add custom logging
- Monitor resource usage
- Track user interactions

## 🚀 Scaling Options

### Horizontal Scaling
- Multiple Streamlit instances
- Load balancer (nginx)
- Database for session storage

### Vertical Scaling
- Increase Docker resources
- Use GPU instances for VLM
- Optimize memory usage

## 🎯 Platform-Specific Tips

### **Streamlit Cloud**
- Keep dependencies minimal
- Use `.streamlit/config.toml` for configuration
- Monitor resource usage
- Add secrets management

### **Heroku**
- Use `heroku.yml` for container deployment
- Add health checks
- Configure environment variables
- Use add-ons for databases

### **AWS/GCP**
- Use managed container services
- Implement auto-scaling
- Add CloudWatch/Stackdriver monitoring
- Use CDN for static assets

## 🔍 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size, optimize models
2. **Slow loading**: Use model caching, compress images
3. **Port conflicts**: Change port in configuration
4. **Dependencies**: Use exact versions in requirements.txt

### Debug Commands
```bash
# Check logs
docker logs <container-id>

# Test locally
streamlit run eegspeech/app/healthcare_vlm_app.py --server.port=8502

# Memory usage
docker stats

# Health check
curl -f http://localhost:8501/healthz
```

## 📚 Resources

### Official Documentation
- [Streamlit Deployment](https://docs.streamlit.io/streamlit-cloud)
- [Docker Documentation](https://docs.docker.com/)
- [Heroku Container Registry](https://devcenter.heroku.com/articles/container-registry-and-runtime)

### Community Resources
- [Streamlit Community](https://discuss.streamlit.io/)
- [Docker Hub](https://hub.docker.com/)
- [GitHub Actions for CI/CD](https://github.com/features/actions)

## 🎉 Quick Start Templates

### For Sharing (Streamlit Cloud)
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Share the URL: `https://your-app.streamlit.app`

### For Distribution (Docker)
1. Build image: `docker build -t your-app .`
2. Push to registry: `docker push your-app`
3. Share run command: `docker run -p 8501:8501 your-app`

### For Collaboration (Setup Script)
1. Create `setup.sh` script
2. Host on GitHub
3. Share: `curl -sSL https://your-repo/setup.sh | bash`

---

**Ready to deploy? Choose your platform and follow the guide above! 🚀**