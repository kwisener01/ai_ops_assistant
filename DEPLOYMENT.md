# Deployment Guide

## Deploying to Streamlit Cloud

### Prerequisites

- GitHub account
- Streamlit Cloud account (https://streamlit.io/cloud)
- Repository pushed to GitHub

### Step-by-Step Deployment

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Add Streamlit dashboard"
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io/
   - Sign in with your GitHub account

3. **Deploy New App**
   - Click "New app"
   - Select your repository: `kwisener01/ai_ops_assistant`
   - Branch: `claude/fynix-systems-setup-016xtLGepeMrVxHaQh935vxn` (or `main` after merging)
   - Main file path: `streamlit_app.py`
   - App URL: Choose your custom URL (e.g., `fynix-ops-dashboard`, `fynix-business-ops`, or `fynix-kpi-dashboard`)

4. **Configure Advanced Settings** (Optional)
   - Python version: 3.11
   - Secrets: Add any API keys or credentials in Streamlit Cloud settings

5. **Deploy**
   - Click "Deploy!"
   - Wait for the app to build and deploy (usually 2-5 minutes)
   - Your app will be available at: `https://[your-chosen-url].streamlit.app/`

**Suggested URLs:**
- `fynix-ops-dashboard.streamlit.app` - Business operations dashboard
- `fynix-business-ops.streamlit.app` - Alternative naming
- `fynix-kpi-tracker.streamlit.app` - KPI-focused naming

> **Note:** The URL `work-assist.streamlit.app` is already in use for a different application. Choose a unique name for this business operations dashboard.

### Local Testing

Before deploying, test the app locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Updating the Deployed App

Any push to your GitHub repository will automatically trigger a redeploy on Streamlit Cloud.

```bash
git add .
git commit -m "Update dashboard features"
git push origin main
```

### Custom Domain

To use a custom domain (e.g., dashboard.fynix.systems):

1. Go to your app settings in Streamlit Cloud
2. Navigate to "Custom subdomain"
3. Follow the DNS configuration instructions
4. Add CNAME record pointing to Streamlit's servers

### Environment Variables

If you need to add environment variables or secrets:

1. Go to your app in Streamlit Cloud
2. Click on "⋮" menu → "Settings"
3. Go to "Secrets" section
4. Add your secrets in TOML format

Example:
```toml
[database]
host = "your-database-host"
username = "your-username"
password = "your-password"

[api]
key = "your-api-key"
```

### Troubleshooting

**App won't start:**
- Check the logs in Streamlit Cloud
- Verify all dependencies are in `requirements.txt`
- Ensure `streamlit_app.py` is in the root directory

**Import errors:**
- Make sure all Python modules are in the same directory
- Check that module names match import statements

**Performance issues:**
- Use `@st.cache_data` decorator for expensive computations
- Limit the amount of data loaded at once
- Consider using a database for large datasets

### Best Practices

1. **Version Control**
   - Keep your code in Git
   - Use meaningful commit messages
   - Tag releases

2. **Security**
   - Never commit API keys or passwords
   - Use Streamlit secrets for sensitive data
   - Enable XSRF protection

3. **Performance**
   - Cache data and computations
   - Optimize DataFrame operations
   - Use session state efficiently

4. **Monitoring**
   - Check app analytics in Streamlit Cloud
   - Monitor resource usage
   - Set up error notifications

### Support

For deployment issues:
- Streamlit Documentation: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: https://github.com/kwisener01/ai_ops_assistant/issues

### Next Steps After Deployment

1. Share the URL with your team
2. Set up data persistence (database or cloud storage)
3. Configure authentication if needed
4. Monitor usage and performance
5. Gather user feedback for improvements

---

**Fynix Systems - AI Operations Assistant**
For more information: https://fynix.systems/
