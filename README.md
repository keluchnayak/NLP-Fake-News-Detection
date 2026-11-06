# 📰 Fake News Detection System (NLP & Machine Learning)

An end-to-end Natural Language Processing (NLP) and Machine Learning application that classifies news articles as **Real** or **Fake** using `CountVectorizer` and Supervised Classification models, deployed with an interactive **Gradio** web interface.

---

## 📌 Project Overview
This project processes raw news articles, cleans and vectorizes text features, trains multiple machine learning classifiers, and deploys the winning model for real-time predictions via a web application.

---

## 📁 Project Structure & Files
```
├── Fake.csv                      # Dataset containing fake news records
├── True.csv                      # Dataset containing real news records
├── fake_news_nlp_project.ipynb     # Main Jupyter Notebook containing Steps 1–8
├── model.pkl                     # Saved trained Naive Bayes classifier (via pickle)
├── vectorizer.pkl                # Saved fitted CountVectorizer (via pickle)
└── README.md                     # Project documentation
```

---

## ⚙️ 8-Step Machine Learning Pipeline

### **Step 1: Data Loading, Merging & EDA**
* Loaded `Fake.csv` and `True.csv`.
* Labeled records (`fake` vs. `true`) and concatenated into a unified DataFrame.
* Performed exploratory data analysis (EDA): checked dataset shape, null values, duplicates, and class balance.

### **Step 2: Target Label Encoding**
* Applied `LabelEncoder` from `scikit-learn`:
  * `0` $\rightarrow$ **Fake News**
  * `1` $\rightarrow$ **True / Real News**

### **Step 3: Stratified Train-Test Split**
* Split data using `train_test_split` with `test_size=0.2` (80% train / 20% test).
* Applied `stratify=y_encoded` to maintain class distribution across splits.

### **Step 4: Regex Text Preprocessing (`clean_text`)**
* Cleaned raw news text:
  1. Converted all text to lowercase.
  2. Applied regular expressions (`re.sub(r'[^a-z\s]', '', text)`) to remove numbers, punctuation, and special characters.
  3. Stripped extra whitespace (`re.sub(r'\s+', ' ', text)`).

### **Step 5: Feature Extraction (Bag of Words)**
* Extracted numerical features using `CountVectorizer()`.
* Learned vocabulary on training data (`fit_transform`) and transformed test data (`transform`) to prevent data leakage.

### **Step 6: Model Training & Comparison**
* Trained three distinct algorithms:
  1. **Multinomial Naive Bayes (`MultinomialNB`)** — Best for text probability modeling.
  2. **K-Nearest Neighbors (`KNeighborsClassifier`)**.
  3. **Decision Tree (`DecisionTreeClassifier`)**.

### **Step 7: Model Evaluation & Performance**
* Evaluated models across **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrices**.
* Evaluated individual predictions with sample actual vs. predicted label checks.

### **Step 8: Model Pickling & Gradio Deployment**
* Saved `model_nb` as `model.pkl` and `CountVectorizer` as `vectorizer.pkl`.
* Built an interactive web app using **Gradio** (`gr.Interface`) to allow users to input headlines or full text for real-time predictions.

---

## 🚀 How to Run the Project

### **1. Prerequisites**
Ensure you have the required Python packages installed:
```bash
pip install numpy pandas scikit-learn gradio
```

### **2. Running the Notebook**
Run the cells in `fake_news_detection.ipynb` sequentially from Step 1 to Step 8.

### **3. Launching the Gradio Web App**
Execute Cell 8 to launch the web interface:
```python
import gradio as gr

interface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=5, placeholder="Paste a news article body or headline here..."),
    outputs=gr.Textbox(label="Detection Result"),
    title="📰 Fake News Detection System"
)
interface.launch(share=True)
```
Open the generated local URL (or public link) in your browser to test live news inputs!
