# 🚀 Heroku Deployment Guide

This guide will help you deploy your Stroke Prediction Dashboard to Heroku.

## 📋 Prerequisites

1. Heroku account (free tier available)
2. Git repository connected to GitHub
3. Heroku app created at: <https://dashboard.heroku.com/apps/stroke-app>

## 🛠️ Deployment Files Ready

Your project is now configured with all necessary Heroku deployment files:

### ✅ Files Configured

- `Procfile` - Tells Heroku how to run your app
- `requirements.txt` - Python dependencies (optimized for Heroku size limits)
- `requirements-full.txt` - Complete dependencies for local development
- `.python-version` - Specifies Python version (3.11)
- `.slugignore` - Excludes unnecessary files from deployment
- `streamlit_dashboard/setup.sh` - Streamlit configuration

## 🚀 Deployment Steps

### Option 1: GitHub Integration (Recommended)

1. **Connect GitHub to Heroku:**
   - Go to <https://dashboard.heroku.com/apps/stroke-app/deploy/github>
   - Connect your GitHub account if not already connected
   - Search for your repository: `Stroke-prediction`
   - Click "Connect"

2. **Enable Automatic Deploys:**
   - Choose `main` branch
   - Enable "Automatic deploys from main"
   - Click "Deploy Branch"

3. **Manual Deploy (if needed):**
   - Scroll to "Manual deploy" section
   - Select `main` branch
   - Click "Deploy Branch"

### Option 2: Heroku CLI

```bash
# Install Heroku CLI first
# Then login and add remote
heroku login
heroku git:remote -a stroke-app

# Deploy
git push heroku main
```

## ⚡ Optimization for Heroku

### Size Optimization

- **requirements.txt** is streamlined to essential packages only
- Heavy ML packages (XGBoost, LightGBM) removed to stay under 500MB limit
- **requirements-full.txt** contains all packages for local development

### File Exclusions

- Jupyter notebooks excluded from deployment
- Documentation and images excluded
- Development files (.venv, cache) excluded

## 🔧 Configuration

### Environment Variables (if needed)

Go to <https://dashboard.heroku.com/apps/stroke-app/settings>

- Click "Reveal Config Vars"
- Add any environment variables your app needs

### Custom Domain (Optional)

- Go to Settings > Domains
- Add your custom domain

## 📊 App Structure

```text
streamlit_dashboard/app.py    # Main Streamlit application
datasets/                     # Data files (included in deployment)
Procfile                     # Heroku process definition
requirements.txt             # Lightweight Python dependencies
.python-version              # Python version
```

## 🐛 Troubleshooting

### Common Issues

1. **Build Fails:**
   - Check `requirements.txt` for incompatible packages
   - Verify Python version in `.python-version`

2. **App Crashes:**
   - Check logs: `heroku logs --tail -a stroke-app`
   - Verify file paths in your Streamlit app

3. **Size Limit Exceeded:**
   - Use lightweight `requirements.txt` (already configured)
   - Remove heavy packages like XGBoost, LightGBM for deployment

### Useful Commands

```bash
# View logs
heroku logs --tail -a stroke-app

# Restart app
heroku restart -a stroke-app

# Check app status
heroku ps -a stroke-app
```

## 🎉 Success

Once deployed, your app will be available at:
**<https://stroke-app.herokuapp.com>**

## 📱 Features Available

Your deployed dashboard includes:

- Interactive stroke risk analysis
- Data visualizations with Plotly
- Professional healthcare styling
- Mobile-responsive design
- Essential ML analysis (using scikit-learn)

## 💡 Development vs Deployment

- **Local Development:** Use `requirements-full.txt` for all ML capabilities
- **Heroku Deployment:** Uses `requirements.txt` for size optimization

---

**Note:** Free Heroku dynos sleep after 30 minutes of inactivity. Consider upgrading to Hobby tier ($7/month) for 24/7 availability.
