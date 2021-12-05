# FDS-assignment-
FDS assignment on SVM(Breast cancer classification)

In this study, Our task is to classify tumors into malignant (cancer) or benign using features obtained from several cell images.

Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

What is a Support Vector Machine (SVM)?
A Support Vector Machine (SVM) is a binary linear classification whose decision boundary is explicitly constructed to minimize generalization error. It is a very powerful and versatile Machine Learning model, capable of performing linear or nonlinear classification, regression and even outlier detection.

SVM is well suited for classification of complex but small or medium sized datasets.

The confusion matrix
The confusion matrix is a table representing the performance of your model to classify labels correctly.

A confusion matrix for a binary classification task:

Predicted Negative	Predicted Positive
Actual Negative	True Negative (TN)	False Positive (FP)
Actual Positive	False Negative (FN)	True Positive (TP)
In a binary classifier, the "true" class is typically labeled with 1 and the "false" class is labeled with 0.

True Positive: A positive class observation (1) is correctly classified as positive by the model.

False Positive: A negative class observation (0) is incorrectly classified as positive.

True Negative: A negative class observation is correctly classified as negative.

False Negative: A positive class observation is incorrectly classified as negative.




Attribute Information:

ID number
Diagnosis (M = malignant, B = benign)
Ten real-valued features are computed for each cell nucleus:

Note:

1.0 (Orange) = Benign (No Cancer)

0.0 (Blue) = Malignant (Cancer)



Radius (mean of distances from center to points on the perimeter)
Texture (standard deviation of gray-scale values)
Perimeter
Area
Smoothness (local variation in radius lengths)
Compactness (perimeter^2 / area - 1.0)
Concavity (severity of concave portions of the contour)
Concave points (number of concave portions of the contour)
Symmetry
Fractal dimension ("coastline approximation" - 1)


