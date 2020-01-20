# Breast-Cancer-Detection
Detection of Breast Cancer in histopathological exams with Deep Learning in Python

This project is the basis of my Master's Dissertation at the Federal University of Paraiba. A first preliminary model has been made with open source medical data available [here](https://www.kaggle.com/paultimothymooney/breast-histopathology-images). The next models will implement patient data directly from the Federal Universities of Parana and Paraiba (Brazil). The model is a Convolutional Neural Network built on Keras, with Data Augmentation and Image Preprocessing done using OpenCV.

There are four files associated to this project:

* Plot_Image.py: Joins all the image patches into a single image, which is the basis for the manual histopathological diagnosis.
* Saving_Variables.py: Reads all images and performs an image filter with a 3x3 kernel to perform sharpening.
* Model.py: Building of the Convolutional Neural Network architecture and implementation of Data Augmentation. This model takes close to a day to run in a regular machine, so it is highly recommend to use a powerful machine or a cloud computing system, such as [vast.ai](https://vast.ai/)
* Model.hdf5: HDF5 file of the final model. Can be directly loaded and used, without the need to run the model training.

The specific architecture of the model was obtained after a vast trial including trial and error, grid search, and bayesian optimization. Adding more layers or increasing the number of parameters would make the model excessively slow to compute, and the combination of Batch Normalization, Dropouts, and Pooling Layers was the best found for the present architecture.

### Technical Notes

The manual exam diagnosis is made by analyzing and diagnosing every 50x50 pixels patch on the image (usually around 3000x3000 pixels). This means that every patient has around 3600 diagnostics. This method is very useful for the implementation of a Machine Learning algorithm, as even with relatively low accuracy and precision, some results with low confidence can be ignored, and even if only a fraction of results have high confidence (based on the results obtained from sigmoid transformations), it is possible to incredibly accurately predict a patient's diagnosis.

### Results

The following results are regarding the Test Set performance on each 50x50 patch (NOT THE OVERALL PATIENT DIAGNOSTIC!)

* Accuracy: 88.90%
* Precision: 87.90%
* Recall: 90.21%

Even though these are great results for a starting model, they are by far not enough to be used in the medical field. However, by narrowing the amount of cases covered, restricting to only those who the model recognizes great confidence (as previously explained), the results can get much better.

**Analyzing cases where the result of the sigmoid function gave over 80% confidence**:

* Covered cases: 80.03%
* Recall: 95.36%
* Precision: 93.39%
* Accuracy: 94.38%

**Analyzing cases where the result of the sigmoid function gave over 95% confidence**:

* Covered cases: 53.16%
* Recall: 97.87%
* Precision: 97.06%
* Accuracy: 97.68%

**Analyzing cases where the result of the sigmoid function gave over 99% confidence**:

* Covered cases: 25.24%
* Recall: 95.36%
* Precision: 93.39%
* Accuracy: 94.38%


**Analyzing cases where the result of the sigmoid function gave over 99.9% confidence**:

* Covered cases: 7.44%
* Recall: 99.39%
* Precision: 99.54%
* Accuracy: 99.70%

**Analyzing cases where the result of the sigmoid function gave over 99.99% confidence**:

* Covered cases: 2.66%
* Recall: 98.31%
* Precision: 100.00%
* Accuracy: 99.88%

Based on these results, it is possible that a valid strategy for diagnosis would be looking for malignant patches with high confidence. As it is unlikely that only a single, or even a few, patches are contaminated, having multiple high confidence diagnosis of cancer are very likely to correspond to an actual true result.
