# Spam Detector

In this research,  we study the performances of two different models for spam filtering. Specifically, we use the scikit-learn pre-implemented models for Naive Bayes and Support Vector Machine (SVM) and evaluate the results on the TReC 2007 Spam Dataset. 

### Set Up Environment:

To create an environment with the environment.yml file, navigate to the folder containing the environment file and type ```conda env create -f environment.yml``` into your terminal. To activate your environment, do ```conda activate env-name```. The environment in the yml file is named ```IR```. 

> **Dependency**
>
> ```sys``` ```os```  ```pickle```   ```csv ``` ```time```
>
> ```pandas``` ```numpy``` ```collections``` ```re```
>
> ```nltk```   ```nltk.corpus ``` ```nltk.stem``` ```rake``` 
>
> ```keras``` ```keras.utils```  ```keras.callbacks```
>
> ```sklearn```

### Data Professing:

Process Trec 2007 Spam Dataset for spam filtering model

* Decoding Email:  ```DataProcessing.ipynb```
  * Decode email of ```byte``` type to ```string``` type
* Extract Contents: ```DataPreProcessing.ipynb```
  * Extracts sender and subject of each email
  * Filter email contents that is not English word and not English stop words by ```nltk corpus``` 
  * Save to ```txt``` file which contains each filtered email as a line

### Models: 

* Evaluation of Models:  ```eval_utils.py```
  * This program provides function ```evaluate```  that prints **Precision**, **Recall**, **F1 Score**, and **Accuracy** of the given model
* Models for Spam Detection: ```Naive_Bayes.ipynb```
  * This file contains three different Naive Bayes model and Support Vector Machine (SVM)
    * Gaussian Naive Bayes
    * Multinomial Naive Bayes
    * Complement Naive Bayes
  * We used 67% of dataset for training and 33% for evaluation

### Source:

* Gordon V. Cormack (2007) *TREC 2007 Spam Track Overview*