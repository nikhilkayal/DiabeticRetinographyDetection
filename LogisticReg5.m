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
% For logistic regression, the response values must be converted to zeros
% and ones because the responses are assumed to follow a binomial
% distribution.
% 1 or true = 'successful' class
% 0 or false = 'failure' class
% NaN - missing response.
successClass = double(1);
failureClass = double(0);
% Compute the majority response class. If there is a NaN-prediction from
% fitglm, convert NaN to this majority class label.
numSuccess = sum(response == successClass);
numFailure = sum(response == failureClass);
if numSuccess > numFailure
    missingClass = successClass;
else
    missingClass = failureClass;
end
successFailureAndMissingClasses = [successClass; failureClass; missingClass];
isMissing = isnan(response);
zeroOneResponse = double(ismember(response, successClass));
zeroOneResponse(isMissing) = NaN;
% Prepare input arguments to fitglm.
concatenatedPredictorsAndResponse = [predictors, table(zeroOneResponse)];
% Train using fitglm.
GeneralizedLinearModel = fitglm(...
    concatenatedPredictorsAndResponse, ...
    'Distribution', 'binomial', ...
    'link', 'logit');

% Convert predicted probabilities to predicted class labels and scores.
convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
scoresFcn = @(p) [1-p, p];
predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
featureSelectionFcn = @(x) x(:,includedPredictorNames);
logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
trainedClassifier.predictFcn = @(x) logisticRegressionPredictFcn(featureSelectionFcn(predictorExtractionFcn(x)));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'QualityAssessment', 'Prescreening', 'MA1', 'MA2', 'MA3', 'MA4', 'MA5', 'MA6', 'Exudate1', 'Exudate2', 'Exudate3', 'Exudate4', 'Exudate5', 'Exudate6', 'Exudate7', 'Exudate8', 'EuclideanDist', 'DiameterOpticDisc', 'AMFMclass'};
trainedClassifier.GeneralizedLinearModel = GeneralizedLinearModel;
trainedClassifier.SuccessClass = successClass;
trainedClassifier.FailureClass = failureClass;
trainedClassifier.MissingClass = missingClass;
trainedClassifier.ClassNames = {successClass; failureClass};
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
KFolds = 5;
cvp = cvpartition(response, 'KFold', KFolds);
% Initialize the predictions to the proper sizes
validationPredictions = response;
numObservations = size(predictors, 1);
numClasses = 2;
validationScores = NaN(numObservations, numClasses);
for fold = 1:KFolds
    trainingPredictors = predictors(cvp.training(fold), :);
    trainingResponse = response(cvp.training(fold), :);
    foldIsCategoricalPredictor = isCategoricalPredictor;
    
    % Data transformation: Select subset of the features
    % This code selects the same subset of features as were used in the app.
    includedPredictorNames = trainingPredictors.Properties.VariableNames([false false true true true false false false false false false false false false true true false false false]);
    trainingPredictors = trainingPredictors(:,includedPredictorNames);
    foldIsCategoricalPredictor = foldIsCategoricalPredictor([false false true true true false false false false false false false false false true true false false false]);
    
    % Train a classifier
    % This code specifies all the classifier options and trains the classifier.
    % For logistic regression, the response values must be converted to zeros
    % and ones because the responses are assumed to follow a binomial
    % distribution.
    % 1 or true = 'successful' class
    % 0 or false = 'failure' class
    % NaN - missing response.
    successClass = double(1);
    failureClass = double(0);
    % Compute the majority response class. If there is a NaN-prediction from
    % fitglm, convert NaN to this majority class label.
    numSuccess = sum(trainingResponse == successClass);
    numFailure = sum(trainingResponse == failureClass);
    if numSuccess > numFailure
        missingClass = successClass;
    else
        missingClass = failureClass;
    end
    successFailureAndMissingClasses = [successClass; failureClass; missingClass];
    isMissing = isnan(trainingResponse);
    zeroOneResponse = double(ismember(trainingResponse, successClass));
    zeroOneResponse(isMissing) = NaN;
    % Prepare input arguments to fitglm.
    concatenatedPredictorsAndResponse = [trainingPredictors, table(zeroOneResponse)];
    % Train using fitglm.
    GeneralizedLinearModel = fitglm(...
        concatenatedPredictorsAndResponse, ...
        'Distribution', 'binomial', ...
        'link', 'logit');
    
    % Convert predicted probabilities to predicted class labels and scores.
    convertSuccessProbsToPredictions = @(p) successFailureAndMissingClasses( ~isnan(p).*( (p<0.5) + 1 ) + isnan(p)*3 );
    returnMultipleValuesFcn = @(varargin) varargin{1:max(1,nargout)};
    scoresFcn = @(p) [1-p, p];
    predictionsAndScoresFcn = @(p) returnMultipleValuesFcn( convertSuccessProbsToPredictions(p), scoresFcn(p) );
    
    % Create the result struct with predict function
    featureSelectionFcn = @(x) x(:,includedPredictorNames);
    logisticRegressionPredictFcn = @(x) predictionsAndScoresFcn( predict(GeneralizedLinearModel, x) );
    validationPredictFcn = @(x) logisticRegressionPredictFcn(featureSelectionFcn(x));
    
    % Add additional fields to the result struct
    
    % Compute validation predictions
    validationPredictors = predictors(cvp.test(fold), :);
    [foldPredictions, foldScores] = validationPredictFcn(validationPredictors);
    
    % Store predictions in the original order
    validationPredictions(cvp.test(fold), :) = foldPredictions;
    validationScores(cvp.test(fold), :) = foldScores;
end

% Compute validation accuracy
correctPredictions = (validationPredictions == response);
isMissing = isnan(response);
correctPredictions = correctPredictions(~isMissing);
validationAccuracy = sum(correctPredictions)/length(correctPredictions);
