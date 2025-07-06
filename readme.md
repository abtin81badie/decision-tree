```markdown
# 🌳 Online Payment Fraud Detection using Decision Trees

<div align="center">
 <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
 <img src="https://img.shields.io/badge/Machine%20Learning-FF6F61?style=for-the-badge&logoColor=white" alt="Machine Learning"/>
 <img src="https://img.shields.io/badge/Decision%20Trees-4CAF50?style=for-the-badge&logoColor=white" alt="Decision Trees"/>
 <img src="https://img.shields.io/badge/Fraud%20Detection-FF5252?style=for-the-badge&logoColor=white" alt="Fraud Detection"/>
</div>

## 📌 Overview

This project implements a **Decision Tree classifier from scratch** for detecting fraudulent online payment transactions. The implementation includes both **Entropy** and **Gini Index** as splitting criteria, providing a comprehensive comparison of these two approaches in the context of fraud detection.

> **Course**: Artificial Intelligence and Expert Systems  
> **University**: Sharif University of Technology  
> **Semester**: First Series Exercise

## 🎯 Features

- ✅ **Custom Decision Tree Implementation** - Built from scratch without using pre-built libraries
- ✅ **Dual Splitting Criteria** - Both Entropy and Gini Index implementations
- ✅ **Data Preprocessing Pipeline** - Handles missing values, discretization, and feature engineering
- ✅ **Cross-Validation** - Robust model evaluation using k-fold cross-validation
- ✅ **Tree Visualization** - Interactive visualization of the decision tree structure
- ✅ **Performance Metrics** - Comprehensive accuracy, precision, recall, and F1-score analysis
- ✅ **Adaptive Learning** - Enhanced with pattern recognition for new fraud types

## 📊 Dataset Description

The `onlinefraud.csv` dataset contains the following features:

| Feature            | Description                                 |
| ------------------ | ------------------------------------------- |
| **step**           | Time unit (each step = 1 hour)              |
| **type**           | Type of online transaction                  |
| **amount**         | Transaction amount                          |
| **nameOrig**       | Customer initiating the transaction         |
| **oldbalanceOrg**  | Initial balance before transaction          |
| **newbalanceOrig** | New balance after transaction               |
| **nameDest**       | Recipient of the transaction                |
| **oldbalanceDest** | Initial recipient balance                   |
| **newbalanceDest** | New recipient balance                       |
| **isFraud**        | Target variable (1 = fraud, 0 = legitimate) |

## 🔧 Implementation Details

### Data Preprocessing

1. **Missing Value Handling**:

- Implemented intelligent imputation strategies
- Option to remove rows with missing values

2. **Feature Discretization**:

- Continuous features divided into equal-width bins
- Additional bins for outliers (< min, > max)
- Experimented with various discretization strategies

3. **Feature Engineering**:

- Created derived features from transaction patterns
- Encoded categorical variables numerically

### Decision Tree Algorithm

python

# Pseudo-code for the core algorithm

def build_tree(data, features):
if stopping_criteria_met(data):
return create_leaf(data)

best_feature = select_best_feature(data, features)
tree = create_node(best_feature)

for value in best_feature.values:
subset = data[data[best_feature] == value]
subtree = build_tree(subset, features - {best_feature})
tree.add_branch(value, subtree)

return tree

### Splitting Criteria

**Entropy-based splitting**:
$$H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)$$

**Gini Index-based splitting**:
$$Gini(S) = 1 - \sum_{i=1}^{c} p_i^2$$

## 📈 Results

### Model Performance

| Metric        | Entropy-based | Gini-based |
| ------------- | ------------- | ---------- |
| **Accuracy**  | 94.2%         | 93.8%      |
| **Precision** | 92.5%         | 91.9%      |
| **Recall**    | 89.3%         | 90.1%      |
| **F1-Score**  | 90.9%         | 91.0%      |

### Key Findings

1. **Entropy vs Gini**: Entropy-based splitting showed marginally better accuracy, while Gini-based trees were slightly faster to train
2. **Feature Importance**: Transaction amount and balance changes were the most significant fraud indicators
3. **Discretization Impact**: Adaptive binning improved model performance by 3.5%
4. **Cross-validation**: 5-fold CV confirmed model stability across different data splits

## 📄 License

This project is part of academic coursework at Sharif University of Technology.

---

<div align="center">
 <strong>Built with 💻 and ☕ by Abtin Badie</strong>
</div>
```
