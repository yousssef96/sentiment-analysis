# Sentiment Analysis of Women's Clothing Reviews

This project performs **sentiment analysis** on women's clothing e-commerce reviews, aiming to predict whether a customer recommends a product (positive/negative sentiment). The pipeline includes text preprocessing, feature extraction, multiple models, and an ensemble voting classifier.



## Dataset

The dataset used is the **Women’s Clothing E-Commerce Reviews** dataset. It contains anonymized customer reviews for various clothing products. References to the company in the text have been replaced with “retailer” to preserve privacy.

- **Rows:** 23,486 customer reviews  
- **Features:** 10 variables  

**Feature Description:**  

| Feature | Type | Description |
|---------|------|-------------|
| Clothing ID | Integer (Categorical) | Identifier for the specific product |
| Age | Positive Integer | Age of the reviewer |
| Title | String | Title of the review |
| Review Text | String | Body of the review |
| Rating | Ordinal Integer (1–5) | Customer’s product score |
| Recommended IND | Binary | Target variable; 1 = recommended, 0 = not recommended |
| Positive Feedback Count | Positive Integer | Number of other customers who found the review positive |
| Division Name | Categorical | High-level product division |
| Department Name | Categorical | Product department name |
| Class Name | Categorical | Product class name |

**Target Variable:**  
`Recommended IND` is used as the target for predictive modeling.

**License:**  
CC0: Public Domain  

**Note on class imbalance:**  
Although the target classes are unbalanced, oversampling techniques such as SMOTE were **not used**, because they cannot generate genuine text. Experiments showed oversampling decreased model performance. 



## Preprocessing

- Removal of HTML tags  
- Lowercasing and cleaning non-word characters  
- Handling emoticons  
- Tokenization and optional stemming  
- Stopword removal using NLTK's English stopwords  



## Feature Extraction

- TF-IDF vectorization of the preprocessed text  
- Hyperparameter tuning: n-grams, tokenizer choice, stopwords, IDF usage, and normalization  



## Models

1. **Logistic Regression**  
2. **XGBoost Classifier**  
3. **Majority Voting Ensemble** combining Logistic Regression and XGBoost  



## Evaluation

- **Metric:** Matthews Correlation Coefficient (MCC)  
- **Cross-Validation:** 5-fold stratified CV  
- **Confusion-matrix** 


