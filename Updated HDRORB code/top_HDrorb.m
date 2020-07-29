clear all
%HD recall of reactive behavior
%data format - columns for each feature/channel of data 
%====Features and Label===
load('input_data.mat')
features=data_all;
%no labels for recall
%f_label_a_binary=data_all(:,215);
%f_label_v_binary=data_all(:,216);

%for only EMG and force sensors, use force sensor as condition to predict
%EMG
%% select = 0 for regular, select = 1 for known condition tracking in training
select = 1;
%% set these:
%maxL_condition_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]; % # of vectors in CiM
%maxL_result_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]; % # of vectors in CiM
maxL_condition_list = [10]; % # of vectors in CiM
maxL_result_list = [10]; % # of vectors in CiM
%count_list = [20, 40, 80, 120, 160, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
count_list = 20;
%maxL_condition_list = 20;
%maxL_result_list = 20;
num_channels = 64;
%num_result_channels_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100];
num_result_channels_list = [1];
num_condition_channels_list = [2];
%num_condition_channels_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100];
D = 10000; %dimension of the hypervectors
features_result = [];
features_condition = [];
learningrate=0.5; % percentage of the dataset used to train the algorithm

%% need to change this for EMG and force sensor data, maybe mav?
%all values become positive (assuming have negative values) (move range to between 0 and max+abs(min)
for k=1:num_channels
    features(:,k)=features(:,k)-min(features(:,k));
end
accuracy_list = zeros(length(count_list), min(length(maxL_condition_list), length(maxL_result_list)));
error_due_to_repeat = zeros(length(count_list), min(length(maxL_condition_list), length(maxL_result_list)));

for g = 1:max(length(num_result_channels_list), length(num_condition_channels_list))
%for g = 1:length(count_list)
%    count = count_list(g);
%     num_result_channels = num_result_channels_list(1);
%     num_condition_channels = num_condition_channels_list(1);
    if (length(num_result_channels_list) > length(num_condition_channels_list))
        num_result_channels = num_result_channels_list(g);
        num_condition_channels = num_condition_channels_list(1);
    elseif (length(num_result_channels_list) == length(num_condition_channels_list))
        num_result_channels = num_result_channels_list(g);
        num_condition_channels = num_condition_channels_list(g);
    else
        num_result_channels = num_result_channels_list(1);
        num_condition_channels = num_condition_channels_list(g);
    end
    count = count_list;
    % move range to between 0 and 1
    %for k=1:num_channels
    %    features(:,k)=features(:,k)/max(features(:,k));
    %end

    % need to do this to get negative values, features in the range of -0.4 and 0.6
     %for i=1:num_channels
     % features(:,i)=features(:,i)-0.4;
     %end

    %% designate condition and result channels
    %features_result=features(:,1:num_result_channels);
    %features_result=features(1:10,33-num_result_channels:32); %5 channels GSR
    %features_condition=features(226:235,33:num_condition_channels+32); %5 channels ECG
    %% works
%     count = count/10;
%     features_result = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     features_condition = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     for i = 2:1:count
%         features_result = [features_result; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     end
%     features_result_1 = features_result;
%     for i = 2:1:num_result_channels
%         features_result = [features_result features_result_1];
%     end
%     for i = 2:1:count
%         features_condition = [features_condition; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     end
%     features_condition_1 = features_condition;
%     for i = 2:1:num_condition_channels
%         features_condition = [features_condition features_condition_1];
%     end
    %features_result = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
    %features_condition = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
    % for i = 1:1:count
    %     features_condition = [features_condition; 1; 0.9; 0.8; 0.7; 0.6; 0.5; 0.4; 0.3; 0.2; 0.1];
    % end

    %% randomly select values
    %features_result = rand(1000,1);
    %features_condition = rand(1000,1);


    %% randomly select, but same result and condition
    %features_result = rand(1000,1);
    %features_condition = features_result;

    %% randomly select pairs for training and then select from previous dataset for recall
%     features_result = rand(count*learningrate,1);
%     features_condition = rand(count*learningrate,1);
%     for i = 1:1:(count - count*learningrate)
%         x = randi([1 count*learningrate],1,1);
%         pair_result = features_result(x);
%         pair_condition = features_condition(x);
%         features_result = [features_result ; pair_result]; %#ok<AGROW>
%         features_condition = [features_condition ; pair_condition]; %#ok<AGROW>
%     end

    % multiple channels randomly select pairs for training and then select from previous dataset for recall
    features_result = rand(count*learningrate,num_result_channels);
    features_condition = rand(count*learningrate,num_condition_channels);
    for i = 1:1:(count - count*learningrate)
        x = randi([1 count*learningrate],1,1);
        pair_result = features_result(x,:);
        pair_condition = features_condition(x,:);
        features_result = [features_result ; pair_result]; %#ok<AGROW>
        features_condition = [features_condition ; pair_condition]; %#ok<AGROW>
    end

    %% verifies that the recall parts are all 
%     matching = zeros((count),1);
%     for i = (count*learningrate+1):1:count
%         if sum(features_condition(1:count*learningrate) == features_condition(i)) == 0
%             fprintf ('not a repeat for condition\n');
%         else
%             [row_condition, ~] = find(features_condition(1:count*learningrate)==features_condition(i));
%         end
%         if sum(features_result(1:count*learningrate) == features_result(i)) == 0
%             fprintf ('not a repeat for result\n');
%         else
%             [row_result, ~] = find(features_result(1:count*learningrate) == features_result(i));
%         end
%         if (row_condition ~= row_result)
%             %fprintf ('not matching\n');
%             matching(i) = 0;
%         else
%             %fprintf ('matching\n');
%             matching(i) = 1;
%         end
%     end
%     if sum(matching) ~= (count - count*learningrate)
%         fprintf('pairs arent matching\n');
%     end

    %% =======HDC============
    if (select == 1)
            HD_functions_HDrorb_knowncondition;     % load HD functions with known condition clause
    else
        HD_functions_HDrorb;     % load HD functions
    end
  for j=1:min(length(maxL_condition_list), length(maxL_result_list))
    learningFrac = learningrate; 
    maxL_condition = maxL_condition_list(j);
    maxL_result = maxL_result_list(j);
    classes = 2; % level of classes
    precision = 20; % no use
    ngram = 10; % for temporal encode
    precision_condition = maxL_condition / max(features_condition(:));
    precision_result = maxL_result / max(features_result(:));

    %% Encoding
    % Also generates continuous item memory for each channel that can be used
    % for encoding multiple values, quantized to nearest value of 1 until max
    % signal value set by maxL (should change these given the pre-processing
    % done to the features to get them between -0.4 and 0.6). Can check EMG
    % classification CiM method given mav pre-processing. Can use different
    % quantization for force sensors and EMG sensors
    [chAMcondition, iMchcondition] = initItemMemories (D, maxL_condition, num_condition_channels,0);
    [chAMresult, iMchresult] = initItemMemories (D, maxL_result, num_result_channels,1);

    %% designate training vs. recall data
    features_condition_training = features_condition(1:floor(count*learningrate),:); %for now..
    features_result_training = features_result(1:floor(count*learningrate),:); %for now..
    features_condition_recall = features_condition(floor(count*learningrate)+1:count,:); %for now..
    features_result_recall = features_result(floor(count*learningrate)+1:count,:); %for now..
    quantized_result_training = int64 (features_result_training * precision_result);
    quantized_condition_training = int64 (features_condition_training * precision_condition);
    quantized_result_recall = int64 (features_result_recall * precision_result);
    quantized_condition_recall = int64 (features_condition_recall * precision_condition);
    size_training = size(features_condition_training);
    length_training = size_training(1);
    size_recall = size(features_condition_recall);
    length_recall = size_recall(1);


    %removed downsampling

    %Sparse biopolar mapping
    %creates matrix of random hypervectors with element values 1, 0, and -1,
    %matrix is has feature (channel) numbers of binary D size hypervectors
    %Should be the S vectors
    %q=0.7;
    %projM1=projBRandomHV(D,num_condition_channels,q);
    %projM2=projBRandomHV(D,num_result_channels,q);

    %for N = 1 : ngram
    % creates ngram for data, rotates through and 
    %for N = 1 : ngram
    %N

    %% Valence

    fprintf ('HDC for RORB\n');
    if (select == 1)
        [prog_HV, prog_HVlist, result_AM, known_condition_integers, result_integers] = hdcrorbtrain (length_training, features_condition_training, features_result_training, chAMcondition, chAMresult, iMchcondition, iMchresult, D, precision_condition, precision_result, num_condition_channels, num_result_channels); 
    else
        [prog_HV, result_AM] = hdcrorbtrain (length_training, features_condition_training, features_result_training, chAMcondition, chAMresult, iMchcondition, iMchresult, D, precision_condition, precision_result, num_condition_channels, num_result_channels); 
    end
    
    [actuator_values] = hdcrorbpredict (length_recall, prog_HV, result_AM, features_condition_recall, chAMcondition, chAMresult, iMchcondition, iMchresult, D, precision_condition, precision_result, num_condition_channels, num_result_channels); 
    expected = int64 (features_result_recall .* precision_result);

    %% check the number of repeat condition samples
%     non_repeat_errors = [];
     row_wrong = [];
     for i = 1:1:length_recall
        if (expected(i,:) ~= actuator_values(i,:))
            row_wrong = [row_wrong i];
        end
     end
     [~,u,c] = unique(quantized_condition_training,'rows');
     [n,~,~] = histcounts(c,numel(u));
    repeat_bin_address = find(n > 1);
    repeat_address_condition = u(repeat_bin_address);
    repeat_values_condition = quantized_condition_training(repeat_address_condition,:);
    [length_repeat, ~] = size(repeat_values_condition);
    wrong_recall_values_condition = quantized_condition_recall(row_wrong,:);
    match_count_condition = 0;
    [length_wrong, ~] = size(wrong_recall_values_condition);
    non_repeat_errors = [];
    for i = 1:1:length_wrong
        x = 0;
        for o = 1:1:length_repeat
            if (repeat_values_condition(o,:) == wrong_recall_values_condition(i,:))
                x = 1;
            end
        end
        if (x == 1)
            match_count_condition = match_count_condition+1;
        else
            non_repeat_errors = [non_repeat_errors; wrong_recall_values_condition(i,:)]; %#ok<AGROW>
        end
    end
    [~,u,c] = unique(quantized_result_training,'rows');
    [n,~,bin] = histcounts(c,numel(u));
    repeat_bin_address = find(n > 1);
    repeat_address_result = u(repeat_bin_address);
    repeat_values_result = quantized_result_training(repeat_address_result,:);
    [length_repeat_result, ~] = size(repeat_values_result);
    match_count_result = 0;
    wrong_recall_values_result = quantized_result_recall(row_wrong,:);
    [length_wrong_result, ~] = size(wrong_recall_values_result);
    for i = 1:1:length_wrong_result
        for o = 1:1:length_repeat_result
            x = x+(repeat_values_result(o,:) == wrong_recall_values_result(i,:));
        end
        if (x > 0)
            match_count_result = match_count_result+1;
        end
    end

    %% check whether the result samples were the same for all of the repeat condition samples
    result_repeat_count = zeros(length_repeat,4);
    location = [];
    for i = 1:1:length_repeat
        for o = 1:1:length_training
            x = (repeat_values_condition(i,:) == quantized_condition_training(o,:));
            if (x == 1)
                location = [location o];
            end
        end
        %location = find(quantized_condition_training == repeat_values_condition(i));
        unique_values = unique(quantized_result_training(location),'rows');
        if (length(unique_values) == length(location))
            result_repeat_count(i,1) = length(location);
            result_repeat_count(i,2) = 0;
        else
            result_repeat_count(i,1) = length(location);
            result_repeat_count(i,2) = length(location) - length(unique_values);
        end
        result_repeat_count(i,3) = result_repeat_count(i,1) - result_repeat_count(i,2);
        if (result_repeat_count(i,3) > 0)
            result_repeat_count(i,4) = 0;
        else

            result_repeat_count(i,4) = 1;
        end
    end
    number_unique_condition = sum(result_repeat_count(:,4));
    training_sample_count = count*learningrate;
    recall_sample_count = count*learningrate;

    training_sample_count
    num_condition_channels
    num_result_channels
    recall_sample_count; 
    error_due_to_repeat(g, j) = length(row_wrong) == match_count_condition;

    maxL_condition
    maxL_result;
    [expected_length, ~] = size(expected);
    accuracy_count = 0;
    for h = 1:1:expected_length
        channel_match_count = 0;
        for u = 1:1:num_result_channels
            match = (expected(h,u) == actuator_values(h,u));
            channel_match_count = channel_match_count + match;
        end
        if channel_match_count == num_result_channels
            accuracy_count = accuracy_count + 1;
        end
    end
    accuracy_2 = accuracy_count/expected_length*100
    accuracy = sum(sum(actuator_values==expected))/numel(expected)*100
    accuracy_list(g, j) = accuracy;
    accuracy_list_2(g, j) = accuracy_2;
    %fprintf('%d of the %d recall samples were wrong, %d of these matched the repeats in the condition samples where %d of the repeated condition values also had the same result value for all the samples, and %d of these matched the repeats in the result samples\n',length(row_wrong), length(quantized_result_recall), match_count_condition, number_unique_condition, match_count_result);
    fprintf('%d of the %d recall samples were wrong, %d of these matched the repeats in the condition samples where %d of the repeated condition values also had the same result sample for all the samples\n',length(row_wrong), length(quantized_result_recall), match_count_condition, number_unique_condition);

    %accuracy(N,2) = acc1;
    %acc1

    %acc_ngram_1(N,j)=acc1;
    %acc_ngram_1(N,j)=acc1;
    end
end

