# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy the **AI Academic Mentor** app to Streamlit Cloud with proper API key configuration.

---

## 📋 **Prerequisites**

1. ✅ **GitHub Account** - Your code must be on GitHub
2. ✅ **Streamlit Cloud Account** - Free at [share.streamlit.io](https://share.streamlit.io)
3. ✅ **Groq API Key** - Get it from [console.groq.com/keys](https://console.groq.com/keys)

---

## 🔐 **Step 1: Get Your Groq API Key**

1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your Groq account
3. Click **"Create API Key"**
4. **Copy** the API key (it starts with `gsk_...`)
5. **Save it securely** - you'll need it in Step 4

---

## 📤 **Step 2: Push Code to GitHub**

Your code is already on GitHub at:
```
https://github.com/KhuzaimaHassan/AI-Academic-Mentor.git
```

✅ **Already done!** The beautiful enhanced app is pushed.

---

## 🌐 **Step 3: Deploy to Streamlit Cloud**

### 3.1 Sign in to Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click **"Sign in with GitHub"**
3. Authorize Streamlit to access your repositories

### 3.2 Create New App

1. Click **"New app"** button
2. Fill in the deployment details:

   ```
   Repository: KhuzaimaHassan/AI-Academic-Mentor
   Branch: main
   Main file path: 9_Deployment/app.py
   ```

3. Click **"Advanced settings"** (Optional - for Python version):
   - Python version: `3.11` (or `3.10`)

4. **DON'T CLICK DEPLOY YET!** - First, add secrets ⬇️

---

## 🔑 **Step 4: Add Secrets (IMPORTANT!)**

Before deploying, you MUST add your Groq API key as a secret:

### 4.1 In the Deployment Settings:

1. Look for **"Advanced settings"** section
2. Find the **"Secrets"** section
3. Click on the **secrets editor**

### 4.2 Add Your API Key:

Copy and paste this into the secrets editor:

```toml
[groq]
GROQ_API_KEY = "your_actual_groq_api_key_here"
```

**⚠️ IMPORTANT:** Replace `your_actual_groq_api_key_here` with your actual Groq API key!

**Example:**
```toml
[groq]
GROQ_API_KEY = "gsk_1234567890abcdef1234567890abcdef"
```

### 4.3 Save and Deploy:

1. Click **"Save"**
2. Click **"Deploy!"**
3. Wait 2-5 minutes for deployment ⏳

---

## 📊 **Step 5: Verify Deployment**

Once deployed, your app will show:

✅ **Success Indicators:**
- Beautiful gradient purple UI
- Student dropdown visible
- "Generate AI Mentor Report" button works
- Console shows: `✅ Using Groq API key from Streamlit secrets`
- Console shows: `✅ Groq LLM initialized successfully`

❌ **Error Indicators:**
- `⚠️ Groq API key not configured`
- `❌ Failed to initialize Groq LLM`

**If you see errors:** Check that your API key is correctly added to Streamlit secrets.

---

## 🔧 **Step 6: Update Secrets (If Needed)**

To update your API key after deployment:

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Click on your app
3. Click **"⋮"** (three dots) → **"Settings"**
4. Go to **"Secrets"** section
5. Update the API key
6. Click **"Save"**
7. App will automatically restart ✨

---

## 📝 **Local Development Setup**

For local development, you can also use Streamlit secrets:

### Option 1: Use `.streamlit/secrets.toml` (Recommended)

1. The file `.streamlit/secrets.toml` is already created
2. Open it and replace `your_groq_api_key_here` with your actual key:

   ```toml
   [groq]
   GROQ_API_KEY = "gsk_1234567890abcdef1234567890abcdef"
   ```

3. Run the app: `streamlit run 9_Deployment/app.py`

**Note:** This file is in `.gitignore` and won't be pushed to GitHub.

### Option 2: Use `config.py` (Alternative)

1. Open `config.py`
2. Replace the placeholder with your API key
3. Run the app

**Note:** Make sure not to push your real API key to GitHub!

---

## 🎯 **API Key Priority**

The app checks for the API key in this order:

1. **Streamlit Secrets** (for Streamlit Cloud) ← **Highest Priority**
2. **config.py** (for local development)
3. **Environment Variable** `GROQ_API_KEY` (fallback)

---

## 🚨 **Troubleshooting**

### Issue: "Groq API key not configured"

**Solution:**
- Check that secrets are correctly formatted in TOML
- Ensure no extra spaces or quotes
- Verify API key is valid at [console.groq.com](https://console.groq.com)

### Issue: App crashes during "Generate AI Mentor Report"

**Possible Causes:**
1. Invalid API key
2. Missing data files (studentVle.csv)
3. Missing model files

**Solutions:**
- Check Streamlit Cloud logs for error details
- Ensure all required files are in the repository
- Verify model files are committed to GitHub

### Issue: "ModuleNotFoundError"

**Solution:**
- Check that `requirements.txt` includes all dependencies
- Streamlit Cloud will auto-install from `requirements.txt`

---

## 📦 **Required Files Checklist**

Ensure these files are in your GitHub repository:

- ✅ `9_Deployment/app.py` - Main Streamlit app
- ✅ `4_Src_Code/agentic_ai_pipeline.py` - AI pipeline (updated for secrets)
- ✅ `requirements.txt` - Python dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `6_Models/*.pkl` - Trained models
- ✅ `6_Models/vectorstore/` - RAG knowledge base
- ✅ `Data_csv's/*.csv` - Student data (except studentVle.csv)

---

## 🎉 **Expected Result**

Once deployed successfully, you'll have:

- **Live URL**: `https://your-app-name.streamlit.app`
- **Beautiful UI**: Purple gradient with modern styling
- **Working AI**: Generate mentor reports for students
- **Secure API**: Key stored safely in Streamlit secrets

---

## 📞 **Support & Resources**

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Groq Documentation**: [console.groq.com/docs](https://console.groq.com/docs)
- **Deployment Help**: [docs.streamlit.io/streamlit-community-cloud/deploy-your-app](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

---

## 🔒 **Security Best Practices**

1. ✅ **Never** commit API keys to GitHub
2. ✅ **Always** use Streamlit secrets for production
3. ✅ Keep `.streamlit/secrets.toml` in `.gitignore`
4. ✅ Rotate API keys periodically
5. ✅ Use different keys for dev and production

---

**Happy Deploying! 🚀**

