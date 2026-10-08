% clear; clc;

%% =========================================================
% Paths
% ==========================================================

baseDir = 'C:\Lab\brain_age_test_retest\output';

resnetFile  = fullfile(baseDir, 'ResNet.xlsx');
sfcnRegFile = fullfile(baseDir, 'SFCN_reg.xlsx');
sfcnSmFile  = fullfile(baseDir, 'SFCN_sm.xlsx');

%% =========================================================
% OAS1 subjects overlapping with ResNet training data
% ==========================================================

OAS1_RESNET_OVERLAP = [
    "OAS1_0061"
    "OAS1_0092"
    "OAS1_0111"
    "OAS1_0150"
    "OAS1_0156"
    "OAS1_0202"
    "OAS1_0230"
    "OAS1_0353"
    "OAS1_0368"
    "OAS1_0379"
];

%% =========================================================
% Model configuration
% Model name | file | MMRR sheet | OAS1 sheet
% ==========================================================

models = {
    'SFCN-reg T1WI',    sfcnRegFile, 'MMRR_T1WI',     'OAS1_T1WI'
    'SFCN-reg GMP',     sfcnRegFile, 'MMRR_GMP',      'OAS1_GMP'
    'SFCN-reg WMP',     sfcnRegFile, 'MMRR_WMP',      'OAS1_WMP'

    'SFCN-sm T1WI',     sfcnSmFile,  'MMRR_T1WI',     'OAS1_T1WI'
    'SFCN-sm GMP',      sfcnSmFile,  'MMRR_GMP',      'OAS1_GMP'
    'SFCN-sm WMP',      sfcnSmFile,  'MMRR_WMP',      'OAS1_WMP'

    'ResNet T1',        resnetFile,  'MMRR_T1',       'OAS1_T1'
    'ResNet T1 B0',     resnetFile,  'MMRR_T1_B0',    'OAS1_T1_B0'
    'ResNet T1 SEX',    resnetFile,  'MMRR_T1_SEX',   'OAS1_T1_SEX'
    'ResNet T1 SEX B0', resnetFile,  'MMRR_T1_SEX_B0','OAS1_T1_SEX_B0'
};

nModels = size(models,1);

%% =========================================================
% Containers
% ==========================================================

MMRR_MAD = nan(nModels,1);
MMRR_ICC = nan(nModels,1);
MMRR_MAE = nan(nModels,1);
MMRR_N   = nan(nModels,1);

OAS1_MAD_all = nan(nModels,1);
OAS1_ICC_all = nan(nModels,1);
OAS1_MAE_all = nan(nModels,1);
OAS1_N_all   = nan(nModels,1);

OAS1_MAD_excl = nan(nModels,1);
OAS1_ICC_excl = nan(nModels,1);
OAS1_MAE_excl = nan(nModels,1);
OAS1_N_excl   = nan(nModels,1);

%% =========================================================
% Main loop
% ==========================================================

for k = 1:nModels

    modelName = models{k,1};
    fileName  = models{k,2};
    mmrrSheet = models{k,3};
    oas1Sheet = models{k,4};

    fprintf('\n========================================\n');
    fprintf('%s\n', modelName);
    fprintf('========================================\n');

    %% -----------------------------------------------------
    % MMRR
    % ------------------------------------------------------

    T = readtable(fileName, ...
        'Sheet', mmrrSheet, ...
        'VariableNamingRule','preserve');

    ID   = string(T{:,1});
    age  = double(T{:,2});
    pred = double(T{:,3});

    % MMRR ID example:
    % sub_113_30
    % sub_113_33
    %
    % Subject = sub_113

    subject = strings(size(ID));

    for i = 1:length(ID)
        token = regexp(ID(i), '^(sub_\d+)_', ...
            'tokens', 'once');

        if ~isempty(token)
            subject(i) = string(token{1});
        end
    end

    [Y, trueAge, subjectNames] = makePairs( ...
        subject, ID, age, pred);

    MMRR_N(k)   = size(Y,1);
    MMRR_MAD(k) = mean(abs(Y(:,1) - Y(:,2)), 'omitnan');
    MMRR_ICC(k) = ICC21(Y);
    MMRR_MAE(k) = mean(abs(pred - age), 'omitnan');

    fprintf('MMRR  N   = %d pairs\n', MMRR_N(k));
    fprintf('MMRR  MAD = %.3f\n', MMRR_MAD(k));
    fprintf('MMRR  ICC = %.3f\n', MMRR_ICC(k));
    fprintf('MMRR  MAE = %.3f\n', MMRR_MAE(k));


    %% -----------------------------------------------------
    % OAS1 - all 20 subjects
    % ------------------------------------------------------

    T = readtable(fileName, ...
        'Sheet', oas1Sheet, ...
        'VariableNamingRule','preserve');

    ID   = string(T{:,1});
    age  = double(T{:,2});
    pred = double(T{:,3});

    % OAS1 ID example:
    % OAS1_0061_MR1
    % OAS1_0061_MR2
    %
    % Subject = OAS1_0061

    subject = regexprep(ID, '_MR\d+$', '');

    [Y_all, trueAge_all, subjectNames_all] = makePairs( ...
        subject, ID, age, pred);

    OAS1_N_all(k)   = size(Y_all,1);
    OAS1_MAD_all(k) = mean(abs(Y_all(:,1) - Y_all(:,2)), ...
                           'omitnan');
    OAS1_ICC_all(k) = ICC21(Y_all);
    OAS1_MAE_all(k) = mean(abs(pred - age), 'omitnan');


    %% -----------------------------------------------------
    % OAS1 - exclude ResNet training overlap
    % ------------------------------------------------------

    keep = ~ismember(subject, OAS1_RESNET_OVERLAP);

    ID_excl      = ID(keep);
    age_excl     = age(keep);
    pred_excl    = pred(keep);
    subject_excl = subject(keep);

    [Y_excl, trueAge_excl, subjectNames_excl] = makePairs( ...
        subject_excl, ID_excl, age_excl, pred_excl);

    OAS1_N_excl(k)   = size(Y_excl,1);

    OAS1_MAD_excl(k) = ...
        mean(abs(Y_excl(:,1) - Y_excl(:,2)), 'omitnan');

    OAS1_ICC_excl(k) = ICC21(Y_excl);

    OAS1_MAE_excl(k) = ...
        mean(abs(pred_excl - age_excl), 'omitnan');


    fprintf('\nOAS1 all\n');
    fprintf('N   = %d pairs\n', OAS1_N_all(k));
    fprintf('MAD = %.3f\n', OAS1_MAD_all(k));
    fprintf('ICC = %.3f\n', OAS1_ICC_all(k));
    fprintf('MAE = %.3f\n', OAS1_MAE_all(k));

    fprintf('\nOAS1 excluding overlap\n');
    fprintf('N   = %d pairs\n', OAS1_N_excl(k));
    fprintf('MAD = %.3f\n', OAS1_MAD_excl(k));
    fprintf('ICC = %.3f\n', OAS1_ICC_excl(k));
    fprintf('MAE = %.3f\n', OAS1_MAE_excl(k));

end

%% =========================================================
% Result table - numerical
% ==========================================================

Model = string(models(:,1));

Result = table( ...
    Model, ...
    MMRR_N, ...
    MMRR_MAD, ...
    MMRR_ICC, ...
    MMRR_MAE, ...
    OAS1_N_all, ...
    OAS1_MAD_all, ...
    OAS1_ICC_all, ...
    OAS1_MAE_all, ...
    OAS1_N_excl, ...
    OAS1_MAD_excl, ...
    OAS1_ICC_excl, ...
    OAS1_MAE_excl);

disp(Result);

%% =========================================================
% Presentation table
%
% OAS1:
% all subjects (excluding overlap)
% ==========================================================

MMRR_MAD_text = compose('%.2f', MMRR_MAD);
MMRR_ICC_text = compose('%.3f', MMRR_ICC);
MMRR_MAE_text = compose('%.2f', MMRR_MAE);

OAS1_MAD_text = compose('%.2f (%.2f)', ...
    OAS1_MAD_all, OAS1_MAD_excl);

OAS1_ICC_text = compose('%.3f (%.3f)', ...
    OAS1_ICC_all, OAS1_ICC_excl);

OAS1_MAE_text = compose('%.2f (%.2f)', ...
    OAS1_MAE_all, OAS1_MAE_excl);

Summary = table( ...
    Model, ...
    MMRR_MAD_text, ...
    MMRR_ICC_text, ...
    MMRR_MAE_text, ...
    OAS1_MAD_text, ...
    OAS1_ICC_text, ...
    OAS1_MAE_text, ...
    'VariableNames', { ...
    'Model', ...
    'MMRR_MAD', ...
    'MMRR_ICC', ...
    'MMRR_MAE', ...
    'OAS1_MAD_All_Excluded', ...
    'OAS1_ICC_All_Excluded', ...
    'OAS1_MAE_All_Excluded'});

disp(Summary);

%% =========================================================
% Save
% ==========================================================

outputFile = fullfile(baseDir, ...
    'MMRR_OAS1_MAD_ICC_MAE.xlsx');

writetable(Result, outputFile, ...
    'Sheet', 'Raw_results');

writetable(Summary, outputFile, ...
    'Sheet', 'Summary');

fprintf('\nSaved:\n%s\n', outputFile);


%% =========================================================
% Local function:
% create scan1 / scan2 pairs
% ==========================================================

function [Y, trueAge, subjectNames] = makePairs( ...
    subject, ID, age, pred)

    subjectNames = unique(subject);
    subjectNames(subjectNames == "") = [];

    Y       = [];
    trueAge = [];
    validSubjects = strings(0,1);

    for s = 1:length(subjectNames)

        sub = subjectNames(s);
        idx = find(subject == sub);

        if length(idx) ~= 2
            warning('%s has %d scans; expected 2.', ...
                sub, length(idx));
            continue;
        end

        % Sort IDs to keep scan order reproducible
        [~, ord] = sort(ID(idx));
        idx = idx(ord);

        Y(end+1,:) = [pred(idx(1)), pred(idx(2))];
        trueAge(end+1,1) = age(idx(1));
        validSubjects(end+1,1) = sub;
    end

    subjectNames = validSubjects;
end


%% =========================================================
% ICC(2,1)
%
% Two-way random effects
% Absolute agreement
% Single measurement
% ==========================================================

function ICC = ICC21(Y)

    % Rows    = subjects
    % Columns = measurements (scan1, scan2)

    Y = Y(all(~isnan(Y),2),:);

    n = size(Y,1);
    k = size(Y,2);

    if n < 2 || k < 2
        ICC = NaN;
        return;
    end

    grandMean = mean(Y(:));

    rowMean = mean(Y,2);
    colMean = mean(Y,1);

    % Mean square for subjects
    SSR = k * sum((rowMean - grandMean).^2);
    MSR = SSR / (n - 1);

    % Mean square for measurements
    SSC = n * sum((colMean - grandMean).^2);
    MSC = SSC / (k - 1);

    % Residual
    residual = Y ...
        - rowMean ...
        - colMean ...
        + grandMean;

    SSE = sum(residual(:).^2);
    MSE = SSE / ((n - 1) * (k - 1));

    % ICC(2,1):
    % two-way random, absolute agreement, single measurement

    ICC = (MSR - MSE) / ...
        (MSR ...
        + (k - 1) * MSE ...
        + (k * (MSC - MSE) / n));
end