% clear; clc;

baseDir = 'C:\Lab\brain_age_test_retest\output';

models = {
    'SFCN-reg T1WI',   fullfile(baseDir,'SFCN_reg.xlsx'), 'Maclaren_T1WI'
    'SFCN-reg GMP',    fullfile(baseDir,'SFCN_reg.xlsx'), 'Maclaren_GMP'
    'SFCN-reg WMP',    fullfile(baseDir,'SFCN_reg.xlsx'), 'Maclaren_WMP'

    'SFCN-sm T1WI',    fullfile(baseDir,'SFCN_sm.xlsx'),  'Maclaren_T1WI'
    'SFCN-sm GMP',     fullfile(baseDir,'SFCN_sm.xlsx'),  'Maclaren_GMP'
    'SFCN-sm WMP',     fullfile(baseDir,'SFCN_sm.xlsx'),  'Maclaren_WMP'

    'ResNet T1',       fullfile(baseDir,'ResNet.xlsx'),    'Maclaren_T1'
    'ResNet T1 B0',    fullfile(baseDir,'ResNet.xlsx'),    'Maclaren_T1_B0'
    'ResNet T1 SEX',   fullfile(baseDir,'ResNet.xlsx'),    'Maclaren_T1_SEX'
    'ResNet T1 SEX B0',fullfile(baseDir,'ResNet.xlsx'),    'Maclaren_T1_SEX_B0'
};

nModels = size(models,1);

%% =========================================================
%  Result containers
% ==========================================================

SD_intra = nan(nModels,3);
SD_total = nan(nModels,3);

%% =========================================================
%  Calculate each model
% ==========================================================

for k = 1:nModels

    modelName = models{k,1};
    fileName  = models{k,2};
    sheetName = models{k,3};

    T = readtable(fileName, 'Sheet', sheetName, ...
        'VariableNamingRule','preserve');

    subjectID = string(T{:,1});
    predAge   = T{:,3};

    fprintf('\n%s\n', modelName);

    for s = 1:3

        subName = sprintf('sub-%02d', s);

        % Select this subject
        idx = startsWith(subjectID, subName + "_");
        id_sub   = subjectID(idx);
        pred_sub = predAge(idx);

        % Extract run number
        runNum = nan(length(id_sub),1);

        for i = 1:length(id_sub)
            token = regexp(id_sub(i), 'run-(\d+)', ...
                'tokens', 'once');

            if ~isempty(token)
                runNum(i) = str2double(token{1});
            end
        end

        % Sort run01 -> run40
        [runNum, order] = sort(runNum);
        pred_sub = pred_sub(order);

        % Check
        if length(pred_sub) ~= 40
            warning('%s: %s has %d scans, expected 40.', ...
                modelName, subName, length(pred_sub));
            continue;
        end

        if ~isequal(runNum(:)', 1:40)
            warning('%s: %s run numbers are incomplete.', ...
                modelName, subName);
            continue;
        end

        %% -------------------------------------------------
        % SD_intra
        % run01-run02, run03-run04, ..., run39-run40
        % --------------------------------------------------

        scan1 = pred_sub(1:2:end);
        scan2 = pred_sub(2:2:end);

        m = length(scan1);

        SD_intra(k,s) = sqrt( ...
            sum((scan1 - scan2).^2) / (2*m) );

        %% -------------------------------------------------
        % SD_total
        % Standard deviation of all 40 predictions
        % --------------------------------------------------

        SD_total(k,s) = std(pred_sub);

        fprintf('  Subject %d: %.3f / %.3f\n', ...
            s, SD_intra(k,s), SD_total(k,s));
    end
end

%% =========================================================
%  Mean across 3 subjects
% ==========================================================

Mean_SD_intra = mean(SD_intra, 2, 'omitnan');
Mean_SD_total = mean(SD_total, 2, 'omitnan');

Difference = Mean_SD_total - Mean_SD_intra;

%% =========================================================
%  Create table
% ==========================================================

Subject1 = compose('%.3f / %.3f', ...
    SD_intra(:,1), SD_total(:,1));

Subject2 = compose('%.3f / %.3f', ...
    SD_intra(:,2), SD_total(:,2));

Subject3 = compose('%.3f / %.3f', ...
    SD_intra(:,3), SD_total(:,3));

Result = table( ...
    string(models(:,1)), ...
    Subject1, ...
    Subject2, ...
    Subject3, ...
    Mean_SD_intra, ...
    Mean_SD_total, ...
    Difference, ...
    'VariableNames', { ...
    'Model', ...
    'Subject1_Intra_Total', ...
    'Subject2_Intra_Total', ...
    'Subject3_Intra_Total', ...
    'Mean_SD_intra', ...
    'Mean_SD_total', ...
    'Difference'});

disp(Result);