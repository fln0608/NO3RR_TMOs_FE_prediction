# Machine Learning Model SHAP Interpretability Analysis

## 1. Core Functionality
This project focuses on using SHAP (Shapley Additive Explanations) to perform deep interpretability analysis on trained neural network models, revealing feature contributions and impact directions on prediction results.

## 2. Special Feature Grouping

Numerical Features (first 9): Direct SHAP value analysis

Supporter Shape Features (indices 9-12): Select activated one-hot encoded features

Catalyst Morphology Features (indices 13-19): Select activated one-hot encoded features

## 3. Visualization Outputs
The program generates two key SHAP plots: Feature Importance Scatter Plot and Bar Plot

## 4. Run
```
python shap.py
```

## 5. [Core SHAP analysis workflow](https://github.com/fln0608/NO3RR_TMOs_FE_prediction/blob/main/SHAP/shap.py#L105C9-L112C70)
```
background = X_train_tensor[:].clone().requires_grad_(True)
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(val_xt_requires_grad)
```


