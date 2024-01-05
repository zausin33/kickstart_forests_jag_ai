Kickstart forests
==================

Repository as starting point for the Use Case for the AI Career Kickstart Program 23/24 _Phase 2_: Development and Implementation of Trustworthy ML-Systems. 

## Data

Data for the observations of 1000 forests consisting on: 
-  **leaf area index [`lai`]**: a measure of forest structure that indicates how dense a forest is. The higher the value, the denser the forest. Values below 1 correspond to Woodlands, values above 1 describe actual forests, than can be divided into different classes (‘open forest’
‘moderate dense forest’, ‘dense forest’ and ‘very dense forest’ with ascending leaf density. Typical values in Germany are around 3-6.
- **`wetness`**: indicates how moist the soil is (values between 0% and 100%)
- **`treeSpecies`**: describes the species of trees in the forest. Forests consist of pines or beech trees and mostly a mix of both
- **`Sentinel_2A_{ABCD}`** columns with ABCD corresponding to a wavelength: reflection values corresponding to the bands of the actual satellite measurements 
- **`w{ABCD}`** columns with ABCD corresponding to a wavelength: contain the simulated sunlight reflection values for each wavelenth of the curve.

All Data was simulated by Rico Fischer, Team Leader 'Digital Twin and Forest Modelling', Julius Kühn-Institute(JKI) for Forest Protection, 

## Notebooks

### Data Exploration
Basic data exploration notebook

### Baseline model
Baseline model to predict `lai`. 
- No feature engineering
- Using a subset of the features available
- Simple regression models
- No hyperparameter optimization
