
# Faradaic Efficiency Prediction using the trained neural network model 

## 1. Core Functionality
Utilize a trained neural network (NN) model to rapidly predict the Faradaic efficiency (FE) of ammonia for unknown transition metal oxides (TMOs) catalysts in electrocatalytic nitrate reduction reactions (NO<sub>3</sub><sup>-</sup>RR).

## 2. Change Feature values
Modify the [source code](https://github.com/fln0608/NO3RR_TMOs_FE_prediction/blob/main/FE%20prediction/FE%20prediction.py#L63C1-L77C21) to predict the FE value for unknown catalysts

## 3. Run
```
python FE prediction.py
```
## 4. Output
```
# example output
Predicted FE for each fold: [0.7523, 0.7689, ...]
Predicted FE: 0.7612 ± 0.0154
```
## 5. All input feature information

| Feature name | Category | Input feature |
| :--- | :--- | :--- |
| Atomic number of the metal in metal oxides | TiO<sub>2</sub>, V<sub>2</sub>O<sub>5</sub>, Fe<sub>3</sub>O<sub>4</sub>, Fe<sub>2</sub>O<sub>3</sub>, Co<sub>3</sub>O<sub>4</sub>, CuO, Cu<sub>2</sub>O, ZnO |22, 23, 26, 27, 29, 30 |
|Valence state of metal ions | TiO<sub>2</sub>, V<sub>2</sub>O<sub>5</sub>, Fe<sub>3</sub>O<sub>4</sub>, Fe<sub>2</sub>O<sub>3</sub>, Co<sub>3</sub>O<sub>4</sub>, CuO, Cu<sub>2</sub>O, ZnO  | 4, 5, 2.67, 3, 2.67, 2, 1, 2 |
| Modified element|Fe, Co, Ni, Cu, Zn, Ru, Ag |26, 27, 28, 29, 30, 44, 47 |
| Mass fraction of modified element | 0-0.3345 | 0-0.3345 |
| Atomic number of the supporter atom | carbon cloth, carbon paper, 3D pinewood-derived carbon, Co foam, Ni foam, Cu foam, Cu mesh, Cu foil, Ti plate, Ti mesh, stainless steel mesh, stainless steel plate | 6, 27, 28, 29, 22, 26 |
| Nitrate concentration |0.001-3 M | 0.001-3 |
| Electrolyte pH | 0.1 M HCl, 0.5 M Na<sub>2</sub>SO<sub>4</sub>, 0.5 M K<sub>2</sub>SO<sub>4</sub>, 1.0 M PBS, 0.1 M PBS , 0.5 M PBS, 0.49 M K<sub>2</sub>SO<sub>4</sub>+0.01 M KOH, 0.1 M NaOH, 0.1 M KOH, 1.0 M NaOH , 1 M KOH |1, 7, 7.4, 12, 13, 14 |
|Applied potential | -1.2-0 V vs RHE |-1.2-0 |
|Modification method |doping, decoration |1, 0 |
|Shape of the supporter|porous foam, mesh, fiber, plate/foil |One-Hot Encoding |
|Catalyst morphology |nanoparticle, nanosphere, nanowire, nanorod, nanotube, nanobelt, nanosheet| One-Hot Encoding |
