# E-Consult Feedback Analysis🔭



A powerful, versatile feedback analysis tool that transforms comments into actionable insights across any field. Hosted on Streamlit, this interactive dashboard empowers businesses, educators, developers, researchers, and individuals to uncover sentiment trends and drive data-driven decision-making anywhere, anytime. 

## Live Demo

Access the live dashboard: **[Launch Dashboard](https://ashwinihebbali-sentiment-analysis-of-comments-receiv-app-oisvve.streamlit.app/)**

## Overview

Unleash the power of feedback with the E-Consult Feedback Analysis Dashboard! This versatile tool helps everyone discover positive, negative, and neutral sentiment trends across diverse categories. Upload your own CSV dataset or explore our sample dataset of 400 comments to uncover insights through vibrant visualizations, intuitive filters, and automated sentiment analysis powered by VADER.

## Features

### Core Capabilities
- **Automated Sentiment Analysis**: VADER-powered classification (positive, negative, neutral)
- **Multi-Domain Support**: Analyze feedback across 400+ categories including:
  - Healthcare (mental health, diagnostics, chronic care, telehealth)
  - Technology (mobile, API, performance, security)
  - Education, Business, Research, and more
- **Interactive Visualizations**:
  - Sentiment distribution pie charts
  - Domain-wise sentiment analysis bar charts
  - Word clouds for positive and negative comments
  - Sentiment intensity breakdown with VADER compound scores
- **Data Upload**: Import your own CSV datasets for custom analysis
- **Smart Filtering**: Filter by sentiment type and domain category
- **Keyword Search**: Search specific terms within comments
- **Downloadable Reports**: Export comprehensive insights as reports
- **Actionable Recommendations**: Auto-generated suggestions for each domain

### Key Metrics Dashboard
- Total comments analyzed
- Sentiment distribution with percentages
- VADER compound score averages
- Domain-specific sentiment trends
- Sample comments with sentiment scores

## 🔗Technology Stack

- **Platform**: Streamlit (Python web framework)
- **Sentiment Analysis**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly, Matplotlib, WordCloud
- **Deployment**: Streamlit Cloud

## Getting Started

### For Users (No Installation Required)
Simply visit the live dashboard and start analyzing:
1. Click the Streamlit app link above
2. Choose to use the default dataset or upload your own CSV
3. Explore visualizations and insights instantly
4. Download your analysis report

### For Developers (Local Setup)

#### Prerequisites
- Python 3.8 or higher
- pip package manager

#### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd feedback-analysis-dashboard
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Usage Guide

### Using the Default Dataset
The dashboard comes pre-loaded with 400 comments showcasing:
- **206 positive comments** (51.5%) - Strong feedback on accessibility, specialized care, innovation
- **194 negative comments** (48.5%) - Technical issues, privacy concerns, UX challenges
- **0 neutral comments** (0.0%) - Clear sentiment polarization
- Diverse domains from general healthcare to specialized areas

### Uploading Custom Datasets

1. Prepare your CSV file with these columns:
   - `sentiment`: "positive", "negative", or "neutral"
   - `domain`: Category or domain name
   - `comment`: The actual feedback text

2. Click "Upload CSV" in the sidebar
3. Select your file
4. View instant analysis across all visualizations

**CSV Format Example:**
```csv
sentiment,domain,comment
positive,general,Great service and quick response
negative,technical,App crashes frequently on mobile
neutral,billing,Standard pricing structure
```
## Key Insights from Default Dataset

### Strengths
- **Accessibility** (100% positive): Braille PDF exports, rural patient empowerment, inclusive design
- **Specialized Care**: Excellence in mental health, diagnostics, chronic care, rehabilitation
- **Innovation**: Strong performance in AI assistance, wearables, telehealth, genomics

### Challenges
- **Technical Issues**: App crashes, performance problems, compatibility issues
- **Privacy Concerns**: Unclear policies, data security questions
- **User Experience**: Functional but unremarkable features needing enhancement

## Interesting Facts

- **Accessibility domains** receive unanimous positive feedback with features like braille exports and rural patient support
- **Technical domains** show the highest concentration of negative feedback related to crashes and performance
- **Neutral feedback** often indicates "functional but unremarkable" features, suggesting innovation opportunities
- **Domain diversity**: The tool analyzes 400+ unique domains from general healthcare to Star Trek references (easter eggs for fun testing)
- **VADER accuracy**: Sentiment analysis shows clear compound score separation between positive (>0.5) and negative (<-0.5) feedback

## Data Requirements

### Minimum CSV Format
```
sentiment,domain,comment
```

### Supported Sentiment Values
- `positive` or `Positive`
- `negative` or `Negative`
- `neutral` or `Neutral`

### Domain Examples
Any category relevant to your analysis:
- Healthcare: general, medication, diagnostics, mental_health
- Technology: mobile, API, security, performance
- Business: billing, cost, features, support
- Custom: Any domain name relevant to your data

## Project Structure

```
feedback-analysis-dashboard/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── default_dataset.csv   # Sample dataset (400 comments)
├── utils/
│   ├── sentiment.py          # VADER sentiment analysis
│   ├── visualization.py      # Chart generation
│   └── recommendations.py    # Insight generation
├── README.md                 # This file
└── .streamlit/
    └── config.toml           # Streamlit configuration
```

## Dependencies

Key Python packages:
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `plotly`: Interactive visualizations
- `wordcloud`: Word cloud generation
- `vaderSentiment`: Sentiment analysis
- `matplotlib`: Additional plotting
- `numpy`: Numerical operations

Install all dependencies:
```bash
pip install streamlit pandas plotly wordcloud vaderSentiment matplotlib numpy
```

## Acknowledgments
Special thanks to Dr. Victor Ikechukwu for his invaluable guidance and support throughout the development of this project. Explore their work: [Victor-Ikechukwu](https://github.com/Victor-Ikechukwu).

---

## 📄 License
This project is licensed under the *MIT License*. See [LICENSE](LICENSE) for details.

---
🔥 If you found this project helpful, please ⭐ it on GitHub!
