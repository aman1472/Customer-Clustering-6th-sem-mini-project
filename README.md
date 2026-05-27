Project Title and Brief Description
Customer Segmentation and Analysis System Using Machine Learning

Customer Segmentation and Analysis System is an intelligent Machine Learning-based application developed to analyze customer purchasing behavior and group customers into meaningful segments. The system utilizes RFM (Recency, Frequency, Monetary) analysis along with the K-Means clustering algorithm to identify customer groups based on behavioral similarity.

The project helps businesses understand customer patterns, improve marketing strategies, identify high-value customers, and support data-driven decision-making. An interactive dashboard built using Streamlit is integrated into the system to visualize clustering insights and customer analytics efficiently.


Technology Stack and Tools Used
Programming Language
Python
Libraries and Frameworks
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Streamlit
Machine Learning Techniques
K-Means Clustering
RFM Analysis
Silhouette Score Evaluation
Development Environment
VS Code
Jupyter Notebook
Python Virtual Environment (venv)
Git/GitHub


Features and Functionalities Implemented
Data Preprocessing Module
Cleans and preprocesses raw retail transaction data
Removes null values and duplicate records
Handles data formatting and feature preparation
RFM Feature Engineering
Calculates Recency, Frequency, and Monetary values
Generates customer behavior features for clustering
Customer Segmentation
Applies K-Means clustering algorithm
Groups customers into distinct behavioral segments
Cluster Evaluation
Uses Silhouette Score to evaluate clustering quality
Generates cluster profiling insights
Interactive Dashboard
Visualizes customer segments using charts and graphs
Displays cluster distribution and customer analytics
Built using Streamlit for interactive exploration
Business Insight Generation
Identifies high-value and low-value customers
Supports targeted marketing and customer retention strategies


Installation/Execution Steps to Run the Project
1. Clone the Repository
git clone <repository-link>
2. Setup the Python Environment

Navigate to the project directory:

cd customer-segmentation-project

Create virtual environment:

python -m venv venv

Activate virtual environment:

Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
3. Install Required Libraries
pip install -r requirements.txt
4. Run the Data Processing Scripts
python src/data_preprocessing.py
python src/clustering.py
python src/evaluation.py
5. Run the Streamlit Dashboard
streamlit run app/dashboard.py
📌 Project Structure
Customer-Segmentation-System/
│
├── data/
│   ├── raw/
│   │   └── Online Retail.xlsx
│   │
│   └── processed/
│       ├── customer_features.csv
│       └── customer_clusters.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── recommendations.py
│
├── app/
│   └── dashboard.py
│
├── requirements.txt
└── README.md
📌 Team Members
Project Developer
Aman
📌 Project Output / Screenshots**
Screenshots Included:
Data preprocessing output
Cluster visualization graphs
Customer distribution charts
Streamlit dashboard interface
Silhouette Score evaluation results
📌 Future Enhancements
Real-time customer analytics
Advanced clustering algorithms
Cloud deployment support
Database integration
AI-powered recommendation engine
