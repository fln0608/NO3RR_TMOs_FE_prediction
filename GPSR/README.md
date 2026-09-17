# A genetic programming symbolic regression ((GPSR)) model for predicting the Faraday efficiency of ammonia
## Overview
This project uses **Genetic Programming Symbolic Regression (GPSR)** to discover an explicit mathematical relationship between catalyst descriptors and the target property **FE**.
The model automatically searches symbolic expressions using genetic programming and outputs an interpretable mathematical formula.

### 1. Change settings

Modify the [source code](https://github.com/fln0608/NO3RR_TMOs_FE_prediction/blob/main/GPSR/GPSR%20Model.py#L73-L95) for your own purpose.

### 2. Training and Validating the model's performance

2.1 Run
```
python GPSR Model.py
```
### 3. Note

The program has saved the [MinMaxScaler(https://github.com/fln0608/NO3RR_TMOs_FE_prediction/blob/main/GPSR/GPSR%20Model.py#L52) and [GPSR model](https://github.com/fln0608/NO3RR_TMOs_FE_prediction/blob/main/GPSR/GPSR%20Model.py#L135) after training.

The saved scaler file and model file are located in the working directory: [My_saved_pipelines_GPSR](https://github.com/fln0608/NO3RR_TMOs_FE_prediction/tree/main/My_saved_pipelines_GPSR).

