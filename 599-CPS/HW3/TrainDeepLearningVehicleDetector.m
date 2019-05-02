%% Prepare Data Set
% Use this block to parse KITTI dataset and store in matlab workspace
% variables. If done once, this block could be skipped or commented out.
% rootDataPath = 'D:\Grad-School\Assignments\599-CPS\HW3\HW3_Resources\KITTI_MOD_fixed\';
rootDataPath = 'KITTI_MOD_fixed';
trainingData = parseVehicleData(fullfile(rootDataPath, 'training'), '2011_09_26_drive_0059*');
testData = parseVehicleData(fullfile(rootDataPath, 'testing'), '2011_09_26_drive_*');

%% Scale the data to 240 x 240 size for resnet
% trainingData = scaleVehicleData(trainingData);
% d = load('D:\Grad-School\Assignments\599-CPS\HW3\HW3_Resources\KITTI_MOD_fixed\training_scaled\scaled_vehicleData_2011_09_26_drive_0057.mat');
% trainingData = d.table;

%% Unroll Vehicle data set
% unrolled_trainingData = unrollBoundingBoxes(trainingData);
% d = load('D:\Grad-School\Assignments\599-CPS\HW3\HW3_Resources\KITTI_MOD_fixed\training_scaled\unrolled_scaled_vehicleData_2011_09_26_drive_0057.mat');
% unrolled_trainingData = d.table;

%%
% The training data is stored in a table. The first column contains the
% path to the image files. The remaining columns contain the ROI labels for
% vehicles. 

% Display first few rows of the data set.
trainingData(1:4,:)
testData(1:4, :)

%% Create a Convolutional Neural Network (CNN)
% A CNN is the basis of the Faster R-CNN object detector. Create the CNN
% layer by layer using Deep Learning Toolbox(TM) functionality.
%
% Start with the |imageInputLayer| function, which defines the type and
% size of the input layer. For classification tasks, the input size is
% typically the size of the training images. For detection tasks, the CNN
% needs to analyze smaller sections of the image, so the input size must be
% similar in size to the smallest object in the data set. In this data set
% all the objects are larger than [16 16], so select an input size of [32
% 32]. This input size is a balance between processing time and the amount
% of spatial detail the CNN needs to resolve.

% Create image input layer.
inputLayer = imageInputLayer([32 32 3]);

%%
% Next, define the middle layers of the network. The middle layers are made
% up of repeated blocks of convolutional, ReLU (rectified linear units),
% and pooling layers. These layers form the core building blocks of
% convolutional neural networks.

% Define the convolutional layer parameters.
filterSize = [3 3];
numFilters = 32;

% Create the middle layers.
middleLayers = [
                
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)   
    reluLayer()     
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)  
    reluLayer() 
    maxPooling2dLayer(3, 'Stride',2)    
    
    ];

%%
% You can create a deeper network by repeating these basic layers. However,
% to avoid downsampling the data prematurely, keep the number of pooling
% layers low. Downsampling early in the network discards image information
% that is useful for learning.
% 
% The final layers of a CNN are typically composed of fully connected
% layers and a softmax loss layer. 

finalLayers = [

    % Add a fully connected layer with 64 output neurons. The output size
    % of this layer will be an array with a length of 64.
    fullyConnectedLayer(64)

    % Add ReLU nonlinearity.
    reluLayer

    % Add the last fully connected layer. At this point, the network must
    % produce outputs that can be used to measure whether the input image
    % belongs to one of the object classes or to the background. This
    % measurement is made using the subsequent loss layers.
    fullyConnectedLayer(width(trainingData))

    % Add the softmax loss layer and classification layer.
    softmaxLayer
    classificationLayer
    
    ];

%%
% Combine the input, middle, and final layers.
layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%% Configure Training Options
% |trainFasterRCNNObjectDetector| trains the detector in four steps. The first
% two steps train the region proposal and detection networks used in Faster
% R-CNN. The final two steps combine the networks from the first two steps
% such that a single network is created for detection [1]. Each training
% step can have different convergence rates, so it is beneficial to specify
% independent training options for each step. To specify the network
% training options use |trainingOptions| from Deep Learning Toolbox(TM).

checkpointpath = '/home/pkadubandi/GH/PradeepKadubandi/USC-CourseWork/599-CPS/HW3/checkpoints/resnet50/drive_0059/05-02-22-options-1-epoch-1024-region-proposals';

epochs = 1;
mini_batch = 1;

% Options for step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', checkpointpath);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', checkpointpath);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', checkpointpath);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', mini_batch, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', checkpointpath);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%%
% Here, the learning rate for the first two steps is set higher than the
% last two steps. Because the last two steps are fine-tuning steps, the
% network weights can be modified more slowly than in the first two steps.
% The mini-batch size must be 1 for Faster R-CNN training, which processes
% multiple image regions from one training image every iteration.
%
% In addition, |'CheckpointPath'| is set to a temporary location for all
% the training options. This name-value pair enables the saving of
% partially trained detectors during the training process. If training is
% interrupted, such as from a power outage or system failure, you can
% resume training from the saved checkpoint.

%%
latestCheckPointFile = fullfile(checkpointpath, 'faster_rcnn_stage_1_checkpoint__360__2019_05_01__14_12_42.mat');
latestCheckPoint = load(latestCheckPointFile);

%% Load saved detector
d = load(fullfile(checkpointpath, 'detector.mat'), 'detector');
detector = d.detector;

%% Train Faster R-CNN
% Now that the CNN and training options are defined, you can train the
% detector using |trainFasterRCNNObjectDetector|.
%
% During training, multiple image regions are processed from the training
% images. The number of image regions per image is controlled by
% |'NumRegionsToSample'|. The |'PositiveOverlapRange'| and
% |'NegativeOverlapRange'| name-value pairs control which image regions are
% used for training. Positive training samples are those that overlap with
% the ground truth boxes by 0.6 to 1.0, as measured by the bounding box
% intersection over union metric. Negative training samples are those that
% overlap by 0 to 0.3. The best values for these parameters should be
% chosen by testing the trained detector on a validation set. To choose the
% best values for these name-value pairs, test the trained detector on a
% validation set.
%
% For Faster R-CNN training, *the use of a parallel pool of MATLAB workers
% is highly recommended to reduce training time*.
% |trainFasterRCNNObjectDetector| automatically creates and uses a parallel
% pool based on your parallel preferences defined in
% <docid:vision_gs#bugsb2y-1 Computer Vision System Toolbox Preferences>. Ensure that the
% use of the parallel pool is enabled prior to training.
%
% A CUDA-capable NVIDIA(TM) GPU with compute capability 3.0 or higher is
% highly recommended for training.
%
% To save time while running this example, a pretrained network is loaded
% from disk. To train the network yourself, set the |doTrainingAndEval|
% variable shown here to true.

% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network. 
doTrainingAndEval = true;

if doTrainingAndEval
    % Set random seed to ensure example training reproducibility.
    rng(0);
    
    % Train Faster R-CNN detector. Select a BoxPyramidScale of 1.2 to allow
    % for finer resolution for multiscale object detection.
    [detector, train_info] = trainFasterRCNNObjectDetector(trainingData, 'resnet50', options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'NumRegionsToSample', [1024 512 1024 512], ...
        'BoxPyramidScale', 1.2);
    
    save(fullfile(checkpointpath, 'detector.mat'), 'detector');
    save(fullfile(checkpointpath, 'train_info.mat'), 'train_info');
else
    % Load pretrained detector for the example.
    detector = data.detector;
end

%%
% To quickly verify the training, run the detector on a test image.

% Read a test image.
I = imread(testData.imageFilename{1});

% Run the detector.
[bboxes, scores] = detect(detector, I);

% Annotate detections in the image.
I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
figure
imshow(I)

%% Insepct data
disp('inspecting training data');
inspect_data(trainingData);
disp('inspecting test data');
inspect_data(testData);

%% Evaluate Detector Using Test Set
% Testing a single image showed promising results. To fully evaluate the
% detector, testing it on a larger set of images is recommended. Computer
% Vision System Toolbox(TM) provides object detector evaluation functions
% to measure common metrics such as average precision
% (|evaluateDetectionPrecision|) and log-average miss rates
% (|evaluateDetectionMissRate|). Here, the average precision metric is
% used. The average precision provides a single number that incorporates
% the ability of the detector to make correct classifications (precision)
% and the ability of the detector to find all relevant objects (recall).
%
% The first step for detector evaluation is to collect the detection
% results by running the detector on the test set. To avoid long evaluation
% time, the results are loaded from disk. Set the |doTrainingAndEval| flag
% from the previous section to true to execute the evaluation locally.
doTrainingAndEval = true;

if doTrainingAndEval
    % Run detector on each image in the test set and collect results.
    resultsStruct = struct([]);
    for i = 1:height(testData)
        
        % Read the image.
        I = imread(testData.imageFilename{i});
        
        % Run the detector.
        [bboxes, scores, labels] = detect(detector, I);
        
        % Collect the results.
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
    end
    
    % Convert the results into a table.
    results = struct2table(resultsStruct);
else
    results = data.results;
end

% Extract expected bounding box locations from test data.
expectedResults = testData(:, 2:end);

% Evaluate the object detector using average precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, expectedResults);

%%
% The precision/recall (PR) curve highlights how precise a detector is at
% varying levels of recall. Ideally, the precision would be 1 at all recall
% levels. In this example, the average precision is 0.6. The use of
% additional layers in the network can help improve the average precision,
% but might require additional training data and longer training time.

% Plot precision/recall curve
figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))

%% Plot training loss
figure
hold on
plot(train_info(1).TrainingLoss, 'b-')
plot(train_info(2).TrainingLoss, 'r-')
plot(train_info(3).TrainingLoss, 'g-')
plot(train_info(4).TrainingLoss, 'm-')
xlabel('Iteration')
ylabel('Training Loss')
legend('Stage 1', 'Stage 2', 'Stage 3', 'Stage 4')
title('Training Loss With Iterations')
hold off
%%
% Display one of the images from the data set to understand the type of
% images it contains.
disp(trainingData.vehicle{10})
disp(length(trainingData.vehicle{10}))
I = imread(trainingData.imageFilename{10});
[h_o, w_o, ~] = size(I);
h_s = 240;
w_s = 240;
J = imresize(I, [h_s, w_s]);
for i = 1:length(trainingData.vehicle{10})
    original_bbox = trainingData.vehicle{10}(i, :);
    scaled_bbox = [original_bbox(1)*w_s/w_o, original_bbox(2)*h_s/h_o, original_bbox(3)*w_s/w_o, original_bbox(4)*h_s/h_o];
    I = insertShape(I, 'Rectangle', original_bbox);
    J = insertShape(J, 'Rectangle', scaled_bbox);
end
figure
imshow(I)
figure
imshow(J)

%% Use already scaled training data
index = 10;
I = imread(trainingData.imageFilename{index});
for i = 1:length(trainingData.vehicle{index})
    original_bbox = trainingData.vehicle{index}(i, :);
    I = insertShape(I, 'Rectangle', original_bbox);
end
figure
imshow(I)

%% Supporting Functions
% Function for parsing KITTI data
function vehicleData = parseVehicleData(rootFolder, filePattern)
    boxesFolder = fullfile(rootFolder, 'boxes');
    imagesFolder = fullfile(rootFolder, 'images');
    fileSearchPattern = fullfile(boxesFolder, filePattern);
    allFiles = dir(fileSearchPattern);
    
    table = cell2table(cell(length(allFiles), 2), 'VariableNames', {'imageFilename' 'vehicle'});
    
    for i = 1:length(allFiles)
        fullpath = fullfile(allFiles(i).folder, allFiles(i).name);
        [~, filename, ~] = fileparts(fullpath);
        pngFileName = strcat(filename, '.png');
        pngFileName = fullfile(imagesFolder, pngFileName);
        
        table{i,1} = {pngFileName};
        
        boxes = [];
        try
            f = fopen(fullpath);
            while true
                line = fgetl(f);
                if ~ischar(line); break; end

                C = strsplit(line, ' ');
                width = str2num(C{6}) - str2num(C{4});
                height = str2num(C{5}) - str2num(C{3});
                if (width > 0.0 && height > 0.0)
                    box = [str2num(C{4}), str2num(C{3}), width, height];
                    boxes = [boxes; box];
                else
                    fprintf('skipping row %f , box %f because of bad data \n', i, length(boxes));
                end
            end
            table{i,2} = {boxes};
            fclose(f);
        catch e
            fprintf ('Error when processing file %s\n', fullpath);
            fprintf(e.message);
            fclose(f);
        end
    end
    
    vehicleData = table;
end

% Function for scaling images and bounding boxes to work with resnet size.
% Call this once to save images and bounding boxes files and reuse them.
% Change the hardcoded values in function before calling it.
function vehicleData = scaleVehicleData(unscaledData)
    rootFolder = 'D:\Grad-School\Assignments\599-CPS\HW3\HW3_Resources\KITTI_MOD_fixed\training_scaled\';
    % mkdir(rootFolder); Does not create the whole path
    vehicleDataFile = 'scaled_vehicleData_2011_09_26_drive_0057';
    h_s = 240;
    w_s = 240;
    
    table = cell2table(cell(336, 2), 'VariableNames', {'imageFilename' 'vehicle'});
    
    for i = 1:336 %Image at 338 index has some issue which is not yet clear...
        boxes = [];
        fullpath = unscaledData.imageFilename{i};
        I = imread(fullpath);
        [~, fileName, ext] = fileparts(fullpath);
        destFullPath = fullfile(rootFolder, 'images', strcat(fileName, ext));
        [h_o, w_o, ~] = size(I);
        I = imresize(I, [h_s, w_s]);
        imwrite(I, destFullPath);
        table{i,1} = {destFullPath};
        for j = 1:length(unscaledData.vehicle{i})
            original_bbox = unscaledData.vehicle{i}(j, :);
            scaled_bbox = [original_bbox(1)*w_s/w_o, original_bbox(2)*h_s/h_o, original_bbox(3)*w_s/w_o, original_bbox(4)*h_s/h_o];
            boxes = [boxes; scaled_bbox];
        end
        table{i,2} = {boxes};
    end
    
    save(fullfile(rootFolder, vehicleDataFile), 'table');
    vehicleData = table;
end

function expandedVehicleData = unrollBoundingBoxes(unexpandedVehicleData)
    rootFolder = 'D:\Grad-School\Assignments\599-CPS\HW3\HW3_Resources\KITTI_MOD_fixed\training_scaled\';
    % mkdir(rootFolder); Does not create the whole path
    vehicleDataFile = 'unrolled_scaled_vehicleData_2011_09_26_drive_0057';

    % A bit hacky way to assign table size, I manually calculated the total
    % entries in a separate script before executing this method.
    total = 2174;
    table = cell2table(cell(total, 2), 'VariableNames', {'imageFilename' 'vehicle'});
    
    tIndex = 1;
    for i = 1:336 %another hack - this is specific to existing variable
        fullpath = unexpandedVehicleData.imageFilename{i};
        for j = 1:length(unexpandedVehicleData.vehicle{i})
            original_bbox = unexpandedVehicleData.vehicle{i}(j, :);
            table{tIndex,1} = {fullpath};
            table{tIndex,2} = {original_bbox};
            tIndex = tIndex+1;
        end
    end
    
    save(fullfile(rootFolder, vehicleDataFile), 'table');
    expandedVehicleData = table;    
end

function inspect_data(data)
    root = '/home/pkadubandi/GH/PradeepKadubandi/USC-CourseWork/599-CPS/HW3';
    for i = 1:height(data)
        I = imread(fullfile(root, data.imageFilename{i}));
        [h, w, ~] = size(I);
        v_dims = size(data.vehicle{i});
        for j = 1:v_dims(1)
            bbox = data.vehicle{i}(j, :);
            if (bbox(3) <= 0 || bbox(4) <= 0)
                fprintf('Row %f: Box %f contains non positive width = %f or height = %f \n', i, j, bbox(3), bbox(4));
            end
            if (bbox(1) < 0 || bbox(2) < 0 || bbox(1) > w || bbox(2) > h)
                fprintf('Row %f: Box %f contains a pixel (%f, %f) out of bounds \n', i, j, bbox(1), bbox(2));
            end
        end
    end
end

%% Summary
% This example showed how to train a vehicle detector using deep learning.
% You can follow similar steps to train detectors for traffic signs,
% pedestrians, or other objects.
%
% To learn more about deep learning, see <docid:vision_doccenter#bvd8yot-1
% Deep Learning for Computer Vision>.

%% References
% [1] Ren, Shaoqing, et al. "Faster R-CNN: Towards Real-Time Object
% detection with Region Proposal Networks." _Advances in Neural Information
% Processing Systems._ 2015.