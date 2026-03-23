# Football Match Prediction  
### Machine Learning & Probabilistic Scoreline Modeling

## Overview
Football match prediction is inherently difficult due to the stochastic nature of the game.  
This project approaches the problem using two complementary methods:

- **Binary Classification** → Predict whether the home team wins  
- **Probabilistic Modeling** → Estimate the full distribution of possible scorelines  

Instead of producing a single deterministic prediction, the system provides **uncertainty-aware outputs**.

---

## Methodology

### 1. Machine Learning (Match Outcome Prediction)

We model:
homeTeamWinner ∈ {0,1}

#### Pipeline
- Feature Scaling: StandardScaler  
- Class Imbalance Handling: SMOTE  
- Train/Test Split: 70/30  

#### Features
- 45 technical features including:
  - Expected Goals difference (xG_diff)
  - Progressive passes (PrgP_diff)
  - Team performance metrics  

#### Models
- K-Nearest Neighbors (KNN)  
- Random Forest  
- AdaBoost  
- XGBoost  

#### Results
| Model          | Accuracy | F1 Score |
|----------------|--------|---------|
| KNN           | 0.7281 | 0.4746 |
| XGBoost       | 0.6140 | 0.4762 |
| Random Forest | 0.6404 | 0.4225 |
| AdaBoost      | 0.6053 | 0.4304 |

Best model: **KNN**

---

### 2. Poisson Model (Scoreline Prediction)

Goals are modeled as Poisson random variables:

G_home ~ Poisson(λ_home)  
G_away ~ Poisson(λ_away)  

Scoreline probability:
P(i,j) = P(G_home = i) × P(G_away = j)

#### Outputs
- Full probability distribution over all scorelines  
- Most probable scoreline  
- Uncertainty quantification  

---

## Evaluation

### Classification
- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve  
- Threshold Optimization (~0.46)  

### Probabilistic Prediction
- Hit@1 = 12.6%  
- Hit@3 = 32.1%  
- Hit@5 = 46.8%  

Includes:
- Top-K scoreline ranking  
- Probability calibration  
- Heatmap visualization  

---

## Baseline
- Always predict **Home Win**  
- Accuracy ≈ 45.8%  

---

## Limitations
- Assumes independence between team scoring  
- Does not capture in-game dynamics (injuries, red cards, tactics)  
- Limited feature representation in Poisson model  

---

## Future Work
- Hybrid model (Poisson + ML)  
- Time-series modeling  
- Player-level features  

---

## Tech Stack
- Python  
- scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  

---

## Dataset
- 380 matches (2024–2025 season)  
- Source: ESPN Soccer Data  

---

## Key Contributions
- Combines Machine Learning and Probabilistic Modeling  
- Produces full scoreline distributions instead of single predictions  
- Introduces Top-K evaluation for football prediction  
- Provides uncertainty-aware predictions  
