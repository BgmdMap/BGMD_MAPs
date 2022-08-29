# Leveraging Feature Bias for Scalable Misprediction Explanation of Machine Learning Models
This repository contains experimemnt data information and evaluation code for paper "Leveraging Feature Bias for Scalable Misprediction Explanation of Machine Learning Models". Here we present the evaluation data details and experiment results in Jupyternote books.

## Experiment results
### Bias Guided Misprediction Diagnoser (BGMD) 
The experiment results of comparing BGMD and EXPLAIN are under folder "BGMD_result".

### Mispredicted Area UPweight Sampling (MAPS)
The experiment results of comparing MAPS, JTT and SMOTE are under folder "MAPS_result".

## Model parameters



## Data
Data size is big, so all data is stored in https://drive.google.com/drive/folders/1-m34KJz5bRQ-QO3N5lefu9axPFlFEDJ-?usp=sharing. We used five fold cross validation using below datasets (80%-train, 20%-test) and reports median number in the paper.

### Merge conflict data

| Name               | Ruby | Python | Java | PHP |
|--------------------|----------------|---------------|------------|------------|
| Number of features |      28        |       28      |     28     |   28       |         
| Instance number    |       40,129   |       49,453  | 26,699     |      50,342  |             

### Bug report close time prediction data

| Name               | BRCTP |
|--------------------|-------|
| Number of features |   21  |
| Instance number    |  1,481|

### Kaggle Data
| Name               | Bank_marketing | Hotel Booking | Job Change | Spam Email | Water Quality |
|--------------------|----------------|---------------|------------|------------|---------------|
| Number of features |      20        |    59         | 13         |   100      |       9       |
| Instance number    |        6,797   |      7,135    |   5,748    |    9,000   |       5,940   |




