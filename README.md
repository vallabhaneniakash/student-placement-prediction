# 🎓 Student Placement Prediction Using Machine Learning

A beginner-friendly machine learning project that predicts whether a student is likely to get placed, based on academic and skill-related attributes. It uses a **Random Forest Classifier** and includes a simple **Streamlit** web app for interactive predictions.

## 📌 Project Description

Campus placement is an important milestone for students. This project trains a classification model on student data (CGPA, IQ, communication skills, technical skills and internship experience) and predicts the placement outcome. The trained model is served through a clean Streamlit interface where anyone can enter student details and instantly see the result.

## 🛠️ Technologies Used

* **Python 3.9+**
* **Pandas** – data handling
* **NumPy** – numerical operations
* **Scikit-learn** – Random Forest model and evaluation
* **Matplotlib** – confusion matrix image
* **Streamlit** – web interface

## ✨ Features

* Small sample dataset included (`dataset.csv`, 60 students)
* Basic preprocessing (missing values and duplicates handling)
* Random Forest classification model
* Accuracy score and confusion matrix
* Interactive Streamlit app with placement prediction and confidence
* Model performance displayed inside the app

## 📂 Project Structure

```
student-placement-prediction/
├── dataset.csv          # Sample student placement dataset
├── train\_model.py       # Trains and evaluates the model
├── app.py               # Streamlit user interface
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .gitignore           # Files ignored by Git
└── screenshots/         # App screenshots for the README
```

## 📊 Dataset

|Column|Description|
|-|-|
|`CGPA`|Cumulative grade point average (5.5 – 9.8)|
|`IQ`|IQ score (80 – 140)|
|`Communication\_Skills`|Rating from 1 to 10|
|`Technical\_Skills`|Rating from 1 to 10|
|`Internship\_Experience`|1 = Yes, 0 = No|
|`Placed`|Target: 1 = Placed, 0 = Not Placed|

> \*\*Note:\*\* This is a small synthetic dataset created for learning and demonstration. Replace it with real data for real-world use, keeping the same column names.

## ⚙️ Installation

1. **Clone the repository**

```bash
   git clone https://github.com/<your-username>/student-placement-prediction.git
   cd student-placement-prediction
   ```

2. **(Optional) Create a virtual environment**

```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
   ```

## ▶️ How to Run

**Step 1 – Train the model** (prints accuracy and confusion matrix, creates `model.pkl`):

```bash
python train\_model.py
```

**Step 2 – Launch the web app:**

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## 📈 Model Results

The model is trained on 80% of the data and tested on the remaining 20%. On the included sample dataset it reaches about **83% accuracy**. Exact numbers are printed when you run `train\_model.py`, and the confusion matrix is saved as `confusion\_matrix.png`.

Because the dataset is small, results can vary if you change the data or random seed.

## 🖼️ Screenshots

Add your screenshots to the `screenshots/` folder and update the file names below.

|Streamlit App|Confusion Matrix|
|-|-|
|!\[App](screenshots/app.png)|!\[Confusion Matrix](confusion\_matrix.png)|

## 🚀 Future Improvements

* Use a larger, real-world dataset
* Compare with other models (Logistic Regression, Decision Tree)
* Deploy the app on Streamlit Community Cloud

## 👤 Author

**Your Name**
GitHub: vallabhaneniakash
LinkedIn: https://www.linkedin.com/in/akash-vallabhaneni-2080402b6

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

