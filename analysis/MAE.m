% clear; clc; close all;
% 
% %% =========================================================
% % Paths
% % =========================================================
% baseDir = 'C:\Lab\brain_age_test_retest\output';
% 
% resnetFile  = fullfile(baseDir, 'ResNet.xlsx');
% sfcnRegFile = fullfile(baseDir, 'SFCN_reg.xlsx');
% sfcnSmFile  = fullfile(baseDir, 'SFCN_sm.xlsx');
% 
% %% =========================================================
% % Special subjects to highlight
% % =========================================================
% specialSubs = ["sub_502", "sub_742"];
% 
% %% =========================================================
% % Model configuration for MMRR
% % label shown on subplot | file | sheet
% % =========================================================
% sfcnRegModels = {
%     'SFCN-reg-T1WI', sfcnRegFile, 'MMRR_T1WI'
%     'SFCN-reg-GM',   sfcnRegFile, 'MMRR_GMP'
%     'SFCN-reg-WM',   sfcnRegFile, 'MMRR_WMP'
% };
% 
% sfcnSmModels = {
%     'SFCN-sm-T1WI', sfcnSmFile, 'MMRR_T1WI'
%     'SFCN-sm-GM',   sfcnSmFile, 'MMRR_GMP'
%     'SFCN-sm-WM',   sfcnSmFile, 'MMRR_WMP'
% };
% 
% resnetModels = {
%     'ResNet-T1',        resnetFile, 'MMRR_T1'
%     'ResNet-T1-B0',     resnetFile, 'MMRR_T1_B0'
%     'ResNet-T1-SEX',    resnetFile, 'MMRR_T1_SEX'
%     'ResNet-T1-SEX-B0', resnetFile, 'MMRR_T1_SEX_B0'
% };
% 
% %% =========================================================
% % Get common axis limits for all MMRR plots
% % x: chronological age ±1
% % y: predicted age ±1
% % =========================================================
% allModels = [sfcnRegModels; sfcnSmModels; resnetModels];
% 
% allAge  = [];
% allPred = [];
% 
% for i = 1:size(allModels,1)
%     fileName  = allModels{i,2};
%     sheetName = allModels{i,3};
% 
%     T = readtable(fileName, 'Sheet', sheetName, ...
%         'VariableNamingRule','preserve');
% 
%     age  = double(T{:,2});
%     pred = double(T{:,3});
% 
%     allAge  = [allAge; age];
%     allPred = [allPred; pred];
% end
% 
% xLim = [floor(min(allAge))-1,  ceil(max(allAge))+1];
% yLim = [floor(min(allPred))-1, ceil(max(allPred))+1];
% 
% %% =========================================================
% % Figure 1: MMRR - SFCN-reg
% % =========================================================
% figure('Color','w','Position',[100 100 1500 420]);
% tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
% 
% for i = 1:size(sfcnRegModels,1)
%     nexttile;
%     plotMMRRSheet(sfcnRegModels{i,2}, sfcnRegModels{i,3}, ...
%         sfcnRegModels{i,1}, xLim, yLim, specialSubs);
% end
% 
% %% =========================================================
% % Figure 2: MMRR - SFCN-sm
% % =========================================================
% figure('Color','w','Position',[100 560 1500 420]);
% tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
% 
% for i = 1:size(sfcnSmModels,1)
%     nexttile;
%     plotMMRRSheet(sfcnSmModels{i,2}, sfcnSmModels{i,3}, ...
%         sfcnSmModels{i,1}, xLim, yLim, specialSubs);
% end
% 
% %% =========================================================
% % Figure 3: MMRR - ResNet
% % =========================================================
% figure('Color','w','Position',[100 1020 1100 760]);
% tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
% 
% for i = 1:size(resnetModels,1)
%     nexttile;
%     plotMMRRSheet(resnetModels{i,2}, resnetModels{i,3}, ...
%         resnetModels{i,1}, xLim, yLim, specialSubs);
% end
% 
% %% =========================================================
% % Local function
% % =========================================================
% function plotMMRRSheet(fileName, sheetName, modelLabel, xLim, yLim, specialSubs)
% 
%     T = readtable(fileName, 'Sheet', sheetName, ...
%         'VariableNamingRule','preserve');
% 
%     ID   = string(T{:,1});
%     age  = double(T{:,2});
%     pred = double(T{:,3});
% 
%     % Extract subject name, e.g. sub_502 from sub_502_30
%     subject = strings(size(ID));
%     for k = 1:length(ID)
%         token = regexp(ID(k), '^(sub_\d+)', 'tokens', 'once');
%         if ~isempty(token)
%             subject(k) = string(token{1});
%         end
%     end
% 
%     % General points
%     isSpecial = ismember(subject, specialSubs);
%     is502 = subject == "sub_502";
%     is742 = subject == "sub_742";
% 
%     hold on;
% 
%     % All normal points
%     scatter(age(~isSpecial), pred(~isSpecial), 28, ...
%         'filled', 'MarkerFaceColor', [0 0.4470 0.7410], ...
%         'MarkerEdgeColor', 'none');
% 
%     % Highlighted points
%     scatter(age(isSpecial), pred(isSpecial), 36, ...
%         'filled', 'MarkerFaceColor', 'r', ...
%         'MarkerEdgeColor', 'none');
% 
%     % y = x line
%     xx = linspace(xLim(1), xLim(2), 100);
%     plot(xx, xx, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);
% 
%     % Regression line
%     p = polyfit(age, pred, 1);
%     yy = polyval(p, xx);
%     plot(xx, yy, '-', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2);
% 
%     % Annotate special subjects
%     idx502 = find(is502);
%     for j = 1:length(idx502)
%         text(age(idx502(j)) + 0.15, pred(idx502(j)), 'sub 502', ...
%             'Color', 'r', 'FontSize', 10, 'HorizontalAlignment','left');
%     end
% 
%     idx742 = find(is742);
%     for j = 1:length(idx742)
%         text(age(idx742(j)) + 0.15, pred(idx742(j)), 'sub 742', ...
%             'Color', 'r', 'FontSize', 10, 'HorizontalAlignment','left');
%     end
% 
%     % Axes and labels
%     xlim(xLim);
%     ylim(yLim);
% 
%     xlabel('Chronological Age (years)', 'FontSize', 11);
%     ylabel('Predicted Brain Age (years)', 'FontSize', 11);
% 
%     title(modelLabel, 'FontSize', 12, 'FontWeight', 'normal');
% 
%     set(gca, 'FontSize', 10, 'Box', 'on', 'LineWidth', 1);
%     grid on;
% 
%     hold off;
% end
clear; clc; close all;

%% =========================================================
% Paths
% =========================================================
baseDir = 'C:\Lab\brain_age_test_retest\output';

resnetFile  = fullfile(baseDir, 'ResNet.xlsx');
sfcnRegFile = fullfile(baseDir, 'SFCN_reg.xlsx');
sfcnSmFile  = fullfile(baseDir, 'SFCN_sm.xlsx');

%% =========================================================
% OAS1 subjects overlapping with ResNet training data
% Only MR1 will be highlighted in red
% =========================================================
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
% =========================================================
sfcnRegModels = {
    'SFCN-reg-T1WI', sfcnRegFile, 'OAS1_T1WI'
    'SFCN-reg-GM',   sfcnRegFile, 'OAS1_GMP'
    'SFCN-reg-WM',   sfcnRegFile, 'OAS1_WMP'
};

sfcnSmModels = {
    'SFCN-sm-T1WI', sfcnSmFile, 'OAS1_T1WI'
    'SFCN-sm-GM',   sfcnSmFile, 'OAS1_GMP'
    'SFCN-sm-WM',   sfcnSmFile, 'OAS1_WMP'
};

resnetModels = {
    'ResNet-T1',        resnetFile, 'OAS1_T1'
    'ResNet-T1-B0',     resnetFile, 'OAS1_T1_B0'
    'ResNet-T1-SEX',    resnetFile, 'OAS1_T1_SEX'
    'ResNet-T1-SEX-B0', resnetFile, 'OAS1_T1_SEX_B0'
};

%% =========================================================
% Common axis limits for ALL OAS1 plots
% =========================================================
allModels = [sfcnRegModels; sfcnSmModels; resnetModels];

allAge  = [];
allPred = [];

for i = 1:size(allModels,1)

    T = readtable( ...
        allModels{i,2}, ...
        'Sheet', allModels{i,3}, ...
        'VariableNamingRule','preserve');

    age  = double(T{:,2});
    pred = double(T{:,3});

    allAge  = [allAge; age];
    allPred = [allPred; pred];
end

% Chronological age: data range +/- 1 year
xLim = [floor(min(allAge))-1, ceil(max(allAge))+1];

% Predicted age: common range across all models +/- 1 year
yLim = [floor(min(allPred))-1, ceil(max(allPred))+1];

fprintf('OAS1 common x-axis: %.0f to %.0f\n', xLim(1), xLim(2));
fprintf('OAS1 common y-axis: %.0f to %.0f\n', yLim(1), yLim(2));

%% =========================================================
% Figure 1: SFCN-reg
% =========================================================
figure( ...
    'Color','w', ...
    'Position',[100 100 1500 430]);

tiledlayout(1,3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for i = 1:size(sfcnRegModels,1)

    nexttile;

    plotOAS1Sheet( ...
        sfcnRegModels{i,2}, ...
        sfcnRegModels{i,3}, ...
        sfcnRegModels{i,1}, ...
        xLim, yLim, ...
        OAS1_RESNET_OVERLAP);
end

%% =========================================================
% Figure 2: SFCN-sm
% =========================================================
figure( ...
    'Color','w', ...
    'Position',[100 570 1500 430]);

tiledlayout(1,3, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for i = 1:size(sfcnSmModels,1)

    nexttile;

    plotOAS1Sheet( ...
        sfcnSmModels{i,2}, ...
        sfcnSmModels{i,3}, ...
        sfcnSmModels{i,1}, ...
        xLim, yLim, ...
        OAS1_RESNET_OVERLAP);
end

%% =========================================================
% Figure 3: ResNet
% =========================================================
figure( ...
    'Color','w', ...
    'Position',[100 100 1100 760]);

tiledlayout(2,2, ...
    'TileSpacing','compact', ...
    'Padding','compact');

for i = 1:size(resnetModels,1)

    nexttile;

    plotOAS1Sheet( ...
        resnetModels{i,2}, ...
        resnetModels{i,3}, ...
        resnetModels{i,1}, ...
        xLim, yLim, ...
        OAS1_RESNET_OVERLAP);
end


%% =========================================================
% Local function
% =========================================================
function plotOAS1Sheet( ...
    fileName, sheetName, modelLabel, ...
    xLim, yLim, overlapSubjects)

    %% Read data
    T = readtable( ...
        fileName, ...
        'Sheet', sheetName, ...
        'VariableNamingRule','preserve');

    ID   = string(T{:,1});
    age  = double(T{:,2});
    pred = double(T{:,3});

    %% -----------------------------------------------------
    % Extract subject ID
    %
    % Expected examples:
    % OAS1_0061_MR1
    % OAS1_0061_MR2
    % ------------------------------------------------------
    subject = regexprep(ID, '_MR\d+$', '');

    %% -----------------------------------------------------
    % Identify MR1 from overlapping subjects
    % ------------------------------------------------------
    isOverlap = ismember(subject, overlapSubjects);

    isMR1 = contains(ID, '_MR1');

    isOverlapMR1 = isOverlap & isMR1;

    %% -----------------------------------------------------
    % Plot
    % ------------------------------------------------------
    hold on;

    % Normal scans = blue
    scatter( ...
        age(~isOverlapMR1), ...
        pred(~isOverlapMR1), ...
        28, ...
        'filled', ...
        'MarkerFaceColor',[0 0.4470 0.7410], ...
        'MarkerEdgeColor','none');

    % Overlap MR1 = red
    scatter( ...
        age(isOverlapMR1), ...
        pred(isOverlapMR1), ...
        38, ...
        'filled', ...
        'MarkerFaceColor','r', ...
        'MarkerEdgeColor','none');

    %% -----------------------------------------------------
    % Identity line y = x
    % ------------------------------------------------------
    xyMin = min([xLim(1), yLim(1)]);
    xyMax = max([xLim(2), yLim(2)]);

    xx = [xyMin xyMax];

    plot( ...
        xx, xx, ...
        '--', ...
        'Color',[0.5 0.5 0.5], ...
        'LineWidth',1.2);

    %% -----------------------------------------------------
    % Axes
    % ------------------------------------------------------
    xlim(xLim);
    ylim(yLim);

    xlabel( ...
        'Chronological Age (years)', ...
        'FontSize',11);

    ylabel( ...
        'Predicted Brain Age (years)', ...
        'FontSize',11);

    title( ...
        modelLabel, ...
        'FontSize',12, ...
        'FontWeight','normal');

    set(gca, ...
        'FontSize',10, ...
        'Box','on', ...
        'LineWidth',1);

    grid on;

    hold off;
end
