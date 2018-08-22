function [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% [trainedClassifier, validationAccuracy] = trainClassifier(trainingData)
% returns a trained classifier and its accuracy. This code recreates the
% classification model
% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'QualityAssessment', 'Prescreening', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 'Exudate1', 'Exudate2', 'Exudate3', 'Exudate4', 'Exudate5', 'Exudate6', 'Exudate7', 'Exudate8', 'EuclideanDist', 'DiameterOpticDisc', 'AMFMclass'};
predictors = inputTable(:, predictorNames);
response = inputTable.ClassLabel;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Data transformation: Select subset of the features
% This code selects the same subset of features as were used in the app.
includedPredictorNames = predictors.Properties.VariableNames([false false true true true false false false false false false false false false true true false false false]);
predictors = predictors(:,includedPredictorNames);
isCategoricalPredictor = isCategoricalPredictor([false false true true true false false false false false false false false false true true false false false]);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
classificationSVM = fitcsvm(...
    predictors, ...
    response, ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
featureSelectionFcn = @(x) x(:,includedPredictorNames);
svmPredictFcn = @(x) predict(classificationSVM, x);
trainedClassifier.predictFcn = @(x) svmPredictFcn(featureSelectionFcn(predictorExtractionFcn(x)));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'QualityAssessment', 'Prescreening', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 'Exudate1', 'Exudate2', 'Exudate3', 'Exudate4', 'Exudate5', 'Exudate6', 'Exudate7', 'Exudate8', 'EuclideanDist', 'DiameterOpticDisc', 'AMFMclass'};
trainedClassifier.ClassificationSVM = classificationSVM;
trainedClassifier.About = 'This struct is a trained model exported from Classification Learner R2017a.';
trainedClassifier.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the
% model.
inputTable = trainingData;
predictorNames = {'QualityAssessment', 'Prescreening', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 'Exudate1', 'Exudate2', 'Exudate3', 'Exudate4', 'Exudate5', 'Exudate6', 'Exudate7', 'Exudate8', 'EuclideanDist', 'DiameterOpticDisc', 'AMFMclass'};
predictors = inputTable(:, predictorNames);
response = inputTable.ClassLabel;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
