# PHML Agentic RAG System - Installation Guide

## 🚀 Quick Setup for Fresh Environment

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google API key for Gemini

### Step 1: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv phml_env

# Activate virtual environment
# On Windows:
phml_env\Scripts\activate
# On macOS/Linux:
source phml_env/bin/activate
```

### Step 2: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter conflicts, install core packages individually:
pip install streamlit
pip install llama-index
pip install llama-index-core
pip install llama-index-llms-gemini
pip install llama-index-embeddings-gemini
pip install google-generativeai
```

### Step 3: Set Up API Key
1. Get your Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Update the API key in `phml_agent_new.py`:
   ```python
   GOOGLE_API_KEY = "your_actual_api_key_here"
   ```

### Step 4: Prepare Data
Ensure you have documents in the `data/` directory:
```
data/
├── sample.md
└── (other PHML documents)
```

### Step 5: Run the Application
```bash
# Run the Streamlit app
streamlit run phml_agent_new.py

# Or test the agent functionality
python test_phml_agent_new.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # If you get import errors, try installing specific versions:
   pip install llama-index==0.10.53
   pip install llama-index-core==0.10.53
   ```

2. **Google API Issues**
   - Ensure your API key is valid
   - Check that you have enabled the Generative AI API
   - Verify billing is set up (if required)

3. **Package Conflicts**
   ```bash
   # Create a fresh environment and install minimal packages:
   pip install streamlit llama-index llama-index-llms-gemini google-generativeai
   ```

4. **Memory Issues**
   - Reduce document size in `data/` directory
   - Use smaller embedding models if needed

### Alternative Installation (Minimal)
If you encounter dependency conflicts, use this minimal setup:

```bash
pip install streamlit
pip install llama-index
pip install google-generativeai
```

Then modify the imports in the code to use only available components.

## 📁 File Structure
```
phml-ai/
├── phml_agent_new.py          # Main agentic RAG application
├── test_phml_agent_new.py     # Test script
├── requirements.txt           # Dependencies
├── INSTALLATION_GUIDE.md      # This file
├── data/
│   └── sample.md             # PHML knowledge base
└── README_PHML_Agent.md      # Documentation
```

## 🎯 Verification
After installation, verify everything works:

1. **Test imports:**
   ```python
   python -c "from llama_index.core.agent import ReActAgent; print('✅ ReAct Agent available')"
   ```

2. **Test Gemini connection:**
   ```python
   python -c "from llama_index.llms.gemini import Gemini; print('✅ Gemini LLM available')"
   ```

3. **Run test script:**
   ```bash
   python test_phml_agent_new.py
   ```

4. **Launch Streamlit app:**
   ```bash
   streamlit run phml_agent_new.py
   ```

## 🔄 Updates
To update the system:
```bash
pip install --upgrade llama-index llama-index-core google-generativeai
```

## 📞 Support
If you encounter issues:
1. Check the error messages carefully
2. Ensure all dependencies are installed
3. Verify your Google API key is working
4. Try the minimal installation approach
5. Check Python version compatibility (3.8+)

---
**Note:** This system requires an active internet connection for the Gemini API calls.
