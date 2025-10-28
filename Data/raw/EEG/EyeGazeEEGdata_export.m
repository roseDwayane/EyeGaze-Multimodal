% EEGLAB EYE數據分epoch導出為CSV (批次處理版)
% 作者: Kong-Yi Chang (Cary)
% 日期: 2025-10-28
% 用途: 批次將EEGLAB格式的EYE數據按epoch分割並根據event類型存成不同命名的CSV檔

clear; clc;

% 添加EEGLAB路徑 (請根據實際安裝位置修改)
% addpath('C:\path\to\eeglab');
% eeglab nogui;

%% 批次處理參數設定
dataRootPath = 'H:\共用雲端硬碟\CNElab_徐浩哲_Howard_slapjack\A.Data';
outputFolder = 'G:\共用雲端硬碟\CNElab_林佳誼_Gaze\B.GazeImage\01.data\EEGseg';

% 設定要處理的Pair範圍
pairRange = 12:40;

% 設定要處理的參與者
participants = {'A', 'B'};

%% 開始批次處理
totalProcessed = 0;
totalSkipped = 0;

for pairNum = pairRange
    pairName = sprintf('Pair-%d', pairNum);
    fprintf('\n========================================\n');
    fprintf('處理 %s\n', pairName);
    fprintf('========================================\n');
    
    for p = 1:length(participants)
        participant = participants{p};
        
        %% 設定路徑
        pairFolder = fullfile(dataRootPath, pairName, 'preprocessing', 'EYE');
        filename = [participant '_EYE_7_Epoch.set'];
        filepath = fullfile(pairFolder, filename);
        
        % 檢查檔案是否存在
        if ~exist(filepath, 'file')
            fprintf('  ⚠ 檔案不存在，跳過: %s\n', filepath);
            totalSkipped = totalSkipped + 1;
            continue;
        end
        
        %% 載入EEGLAB資料
        fprintf('\n  正在載入: %s - Participant %s\n', pairName, participant);
        try
            EEG = pop_loadset('filename', filename, 'filepath', pairFolder);
        catch ME
            fprintf('  ✗ 載入失敗: %s\n', ME.message);
            totalSkipped = totalSkipped + 1;
            continue;
        end
        
        % 檢查資料結構
        fprintf('  資料維度: %d channels x %d timepoints x %d epochs\n', ...
            size(EEG.data, 1), size(EEG.data, 2), size(EEG.data, 3));
        
        %% 建立event與epoch的對應關係
        epochEventTypes = cell(EEG.trials, 1);
        for epochIdx = 1:EEG.trials
            % 取得這個epoch的第一個event索引
            firstEventIdx = EEG.epoch(epochIdx).event(1);
            
            % 從EEG.event中取得對應的event type
            epochEventTypes{epochIdx} = EEG.event(firstEventIdx).type;
        end
        
        %% 為每種event type建立計數器
        countA711 = 0;
        countB711 = 0;
        count511 = 0;
        count611 = 0;
        
        %% 確保輸出資料夾存在
        if ~exist(outputFolder, 'dir')
            mkdir(outputFolder);
            fprintf('  建立輸出資料夾: %s\n', outputFolder);
        end
        
        %% 遍歷每個epoch並存檔
        savedCount = 0;
        for epochIdx = 1:EEG.trials
            % 取得當前epoch的資料 (channels x timepoints)
            epochData = EEG.data(:, :, epochIdx);
            
            % 取得event type
            eventType = epochEventTypes{epochIdx};
            
            if isempty(eventType)
                continue;
            end
            
            % 將event type轉換為字串
            if isnumeric(eventType)
                eventType = num2str(eventType);
            end
            
            % 移除前導空格
            eventType = strtrim(eventType);
            
            %% 根據event type決定檔名
            outputFilename = '';
            
            switch eventType
                case {'A710', 'A711'}
                    countA711 = countA711 + 1;
                    trialNum = sprintf('%02d', countA711);
                    if strcmp(participant, 'A')
                        outputFilename = sprintf('%s-A-Single-EYE_trial%s_player.csv', pairName, trialNum);
                    else % participant == 'B'
                        outputFilename = sprintf('%s-A-Single-EYE_trial%s_observer.csv', pairName, trialNum);
                    end
                    
                case {'B710', 'B711'}
                    countB711 = countB711 + 1;
                    trialNum = sprintf('%02d', countB711);
                    if strcmp(participant, 'A')
                        outputFilename = sprintf('%s-B-Single-EYE_trial%s_observer.csv', pairName, trialNum);
                    else % participant == 'B'
                        outputFilename = sprintf('%s-B-Single-EYE_trial%s_player.csv', pairName, trialNum);
                    end
                    
                case {'510', '511'}
                    count511 = count511 + 1;
                    trialNum = sprintf('%02d', count511);
                    if strcmp(participant, 'A')
                        outputFilename = sprintf('%s-Comp-EYE_trial%s_playerA.csv', pairName, trialNum);
                    else % participant == 'B'
                        outputFilename = sprintf('%s-Comp-EYE_trial%s_playerB.csv', pairName, trialNum);
                    end
                    
                case {'610', '611'}
                    count611 = count611 + 1;
                    trialNum = sprintf('%02d', count611);
                    if strcmp(participant, 'A')
                        outputFilename = sprintf('%s-Coop-EYE_trial%s_playerA.csv', pairName, trialNum);
                    else % participant == 'B'
                        outputFilename = sprintf('%s-Coop-EYE_trial%s_playerB.csv', pairName, trialNum);
                    end
                    
                otherwise
                    continue;
            end
            
            % 檢查檔名是否成功生成
            if isempty(outputFilename)
                continue;
            end
            
            % 完整的輸出路徑
            outputPath = fullfile(outputFolder, outputFilename);
            
            % 存成CSV檔 (保持 channels x timepoints 格式)
            try
                writematrix(epochData, outputPath);
                savedCount = savedCount + 1;
            catch ME
                fprintf('  ✗ 存檔失敗: %s - %s\n', outputFilename, ME.message);
            end
        end
        
        fprintf('  ✓ %s - Participant %s 完成: 已存檔 %d 個 epochs\n', pairName, participant, savedCount);
        fprintf('    統計: A711=%d, B711=%d, 511=%d, 611=%d\n', countA711, countB711, count511, count611);
        totalProcessed = totalProcessed + 1;
    end
end

%% 顯示總結
fprintf('\n========================================\n');
fprintf('批次處理完成！\n');
fprintf('成功處理: %d 個檔案\n', totalProcessed);
fprintf('跳過/失敗: %d 個檔案\n', totalSkipped);
fprintf('========================================\n');