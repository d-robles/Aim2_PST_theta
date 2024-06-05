%training wheels

clear all
close all
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
%% Input Information

exp = 'PST';
%PSTS0110Hz0001
% *************************************************************************
subs = {'0001', '0002', '0006', '0013'};
%subs = {'100' '101' '102'}; %to test on just one sub 
nsubs = length(subs); % number of subjects being run
% *************************************************************************
conds =  {'pos_all' 'neg_all'};
nconds = length(conds);

session = {'S01'};
tmstarget = {'10Hz'};
% *************************************************************************

Pathname = 'D:\PST_2a\raw_files\segments';

% Location of electrode information
% electrode_loc = 'M:\Analysis\Electrodelocs\Vamp_EOG_electrode_locs.ced';

% *************************************************************************
% A few electrodes
 electrode = [8 11]; %Bikeout data has dif electrode map. Pz = 7, Fz = 9
 elec_names = {'FCz'; 'Cz'};


% Multiple electrodes
% electrode = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15];
% elec_names = {'Oz';'P7';'T7';'P3';'C3';'F3';'Pz';'Cz';'Fz';'P4';'C4';'F4';'P8';'T8';'FP2'};

% *************************************************************************
% Pick the type of trials
%   1 = targets
%   2 = standards
%   3 = standards and targets
perms = 3;
trialevent = {'pos_all';'neg_all'};
%trialevent = {'Standards'};
%trialevent = {'Targets'};

% *************************************************************************
% If using eeglab to plot ersp
elab_plot = 'Off'; % No

% *************************************************************************
% Set baseline
baseln = [-1000 -500];
%baseln = [NaN];

% *************************************************************************
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

%time frequency parameters
tf_epochslim = [-1  1.9];
tf_cycles = [1 0.5];
timesout = 200;
analfreq = [1.6 30];
pratio = 2;
%
%-------------------------------------------------------------------------
%% TF Analysis
% -------------------------------------------------------------------------
% clear ersp freqs itc powbase times

i_count = 0;

for i_sub = 1:nsubs
    for i_cond = 1:nconds
        
        i_count = i_count + 1; % counter to select data from ALLEEG
        
        % Filename = [subs{i_sub} '_' exp '_' conds{i_cond}];
        Filename = ['PSTS0110Hz' subs{i_sub} '_'];
            
        if perms == 1 % Load target data
            EEG = pop_loadset('filename',Filename,'filepath','M:\Data\bike\BikeOut\segments_fft_JK\');
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
            
            for i_chan = 1:length(electrode)
                EEG = eeg_checkset(EEG);
                % Caption if creating plots with pop_newtimef
%                 i_cap = strcat(exp, '_', subs(i_sub), '_', elec_names(i_chan),...
%                     '_', conds(i_cond));
                % If plotting with the pop_newtimef function
                if strcmp(elab_plot, 'Yes')
                    figure
                end
                % FFT with output ersp & itc in the order of each subj, cond, 
                % trial type (perm), and channel.
                [ersp(i_sub,i_cond,1,i_chan,:,:),itc(i_sub,i_cond,1,i_chan,:,:),powbase,times,freqs] =...
                    pop_newtimef(EEG, 1, i_chan, tf_epochslim*1000, tf_cycles, 'topovec', i_chan,...
                    'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'baseline', baseln,...
                    'freqs', analfreq, 'plotersp', elab_plot, 'plotitc', elab_plot,...
                    'padratio', 8, 'timesout', timesout);
            end
            
        elseif perms == 2 % Load standards data
            EEG = pop_loadset('filename',[Filename '_fft_Standards.set'],'filepath','M:\Data\bike\BikeOut\segments_fft_JK\');
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
            
            for i_chan = 1:length(electrode)
                EEG = eeg_checkset(EEG);
                % Caption if creating plots with pop_newtimef
%                 i_cap = strcat(exp, '_', subs(i_sub), '_', elec_names(i_chan),...
%                     '_', conds(i_cond));
                % If plotting with the pop_newtimef function
                if strcmp(elab_plot, 'Yes')
                    figure
                end
                % FFT with output ersp & itc in the order of each subj, cond, 
                % trial type (perm), and channel.
                [ersp(i_sub,i_cond,1,i_chan,:,:),itc(i_sub,i_cond,1,i_chan,:,:),powbase,times,freqs] =...
                    pop_newtimef(EEG, 1, i_chan, tf_epochslim*1000, tf_cycles, 'topovec', i_chan,...
                    'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'baseline', baseln,...
                    'freqs', analfreq, 'plotphase', 'Off', 'plotersp', elab_plot,...
                    'plotitc', elab_plot, 'padratio', 8, 'timesout', timesout);
            end
            
        elseif perms == 3
            
            % Load positive data 
             EEG = pop_loadset('filename',[Filename 'pos_all.set'],'filepath','D:\PST_2a\raw_files\segments');
             %EEG = pop_resample(EEG, 128);
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
            for i_chan = 1:length(electrode)
                EEG = eeg_checkset(EEG);
                % Caption if creating plots with pop_newtimef
%                 i_cap = strcat(exp, '_', subs(i_sub), '_', elec_names(i_chan),... commented this out for now DR
%                     '_', conds(i_cond));
                % If plotting with the pop_newtimef function
                if strcmp(elab_plot, 'Yes')
                    figure
                end
                % FFT with output ersp & itc in the order of each subj, cond, 
                % trial type (perm), and channel.
                [ersp(i_sub,i_cond,1,i_chan,:,:),itc(i_sub,i_cond,1,i_chan,:,:),powbase,times,freqs] =...
                    pop_newtimef(EEG, 1, i_chan, tf_epochslim*1000, tf_cycles, 'topovec', i_chan,...
                    'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'baseline', baseln,...
                    'freqs', analfreq, 'plotphase', 'Off', 'plotersp', elab_plot,...
                    'plotitc', elab_plot, 'padratio', pratio, 'timesout', timesout);
            end
            
            % Load negative data
            EEG = pop_loadset('filename',[Filename 'neg_all.set'],'filepath','D:\PST_2a\raw_files\segments');
            %EEG = pop_resample(EEG, 128);
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
            for i_chan = 1:length(electrode)
                EEG = eeg_checkset(EEG);
                % Caption if creating plots with pop_newtimef
%                 i_cap = strcat(exp, '_', subs(i_sub), '_', elec_names(i_chan),...
%                     '_', conds(i_cond));
                % If plotting with the pop_newtimef function
                if strcmp(elab_plot, 'Yes')
                    figure
                end
                % FFT with output ersp & itc in the order of each subj, cond, 
                % trial type (perm), and channel.
                % (participants x conditions x events x electrodes x frequencies x timepoints)

                [ersp(i_sub,i_cond,2,i_chan,:,:),itc(i_sub,i_cond,2,i_chan,:,:),powbase,times,freqs] =...
                    pop_newtimef(EEG, 1, i_chan, tf_epochslim*1000, tf_cycles, 'topovec', i_chan,...
                    'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, 'baseline', baseln,...
                    'freqs', analfreq, 'plotphase', 'Off', 'plotersp', elab_plot,...
                    'plotitc', elab_plot, 'padratio', pratio, 'timesout', timesout);
            end
            
        end     
          
    end
 
end

eeglab redraw

% -------------------------------------------------------------------------
% /////////////////////////////////////////////////////////////////////////
% -------------------------------------------------------------------------
%      %* NOT NEEDED FOR ERSP ANALYSIS FOR DISSERTATION ATM
%% Plotting functions

%  Skate_ERSP_Plot(ALLEEG, 1, EEG, elec_names, electrode, electrode_loc,...
%      ersp, exp, Filename, freqs, itc, Pathname, perms, powbase, subs, times,...
%      trialevent)
%  
% Skate_ERSP_Plot(freqs,ersp)

% -------------------------------------------------------------------------
% /////////////////////////////////////////////////////////////////////////
% -----------------------------------------------------------------------
%    
% %%
% %QUICK PLOTS
% %conds =  {'In' 'Out'};
% %targets = 1
% %stand =2
% CLim = [-1.5 1.5];
% 
% for i_event = 1:length(trialevent)
% 
%     for ch_ersp = 1:length(electrode)
% 
%         % ERSP values by electrode
%         % (participants x conditions x events x electrodes x frequencies x timepoints)
%         in_ersp_chan  =  squeeze(mean(ersp(:,1,i_event,ch_ersp,:,:),1));
%         out_ersp_chan = squeeze(mean(ersp(:,2,i_event,ch_ersp,:,:),1));
% 
%         figure; 
%         colormap('redblue')
% 
%         % Subplot 1: out
%         subplot(2,1,1); 
%         imagesc(times,freqs,in_ersp_chan,CLim);
%         title({['ERSP: ' char(trialevent{i_event}) ', Inside']; char(elec_names(ch_ersp))}); 
%         set(gca,'Ydir','Normal')
%         line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
%         ylabel('Freq (Hz)');
%         colorbar
%         % Subplot 1: in
%         subplot(2,1,2); 
%         imagesc(times,freqs,out_ersp_chan,CLim);
%         title({['ERSP: ' char(trialevent{i_event}) ', Outside']; char(elec_names(ch_ersp))}); 
%         set(gca,'Ydir','Normal')
%         line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5) 
%         ylabel('Freq (Hz)');
%         colorbar
% %         % Subplot 1: out-in
% %         subplot(3,1,3); 
% %         imagesc(times,freqs,npref_ersp_chan-pref_ersp_chan,CLim); 
% %         title({['ERSP: ' char(trialevent{i_event}) ', np-p']; char(elec_names(ch_ersp))});
% %         set(gca,'Ydir','Normal')
% %         line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5) 
% %         ylabel('Freq (Hz)'); xlabel('Time (ms)');
% %         colorbar
% 
%         %Overall subplot title
%     %     supertitle(['ERSP: ' char(elec_names(ch_ersp))],'FontSize',12)
% 
%         clear in_ersp_chan out_ersp_chan 
%     end
% 
%     clear ch_ersp 
% 
% end
% 
% clear i_event

%% ***************************************************************
%
%THESE ARE THE MAIN ERSP PLOTS FOR MANUSCRIPT. 
%
%****************************************************************

%PLOTS USING DIFFERENCE ERSPS AND AVERAGED BY PREFERENCE 
%erp_diff_out = squeeze(erp_out(:,1,:,:,:)-erp_out(:,2,:,:,:));
%time_window = find(EEG.times>-200,1)-1:find(EEG.times>1000,1)-2;

%averaging ERSP windows and subtracting targets-standards
% (participants x conditions x events x electrodes x frequencies x timepoints)
%conds =  {'In' 'Out'};
%%targets = 1
%stand =2
%ersp_dif = squeeze(ersp(:,:,1,:,:,:)-ersp(:,:,2,:,:,:));
electrode = 1; 
ersp_pos  =  squeeze(mean(ersp(:,1,1,electrode,:,:),1));
ersp_neg  =  squeeze(mean(ersp(:,2,2,electrode,:,:),1));
in_ersp_dif = squeeze(ersp_pos-ersp_neg);

out_ersp_targ = squeeze(mean(mean(ersp(:,2,1,electrode,:,:),1),2));
out_ersp_stand = squeeze(mean(mean(ersp(:,2,2,electrode,:,:),1),2));
out_ersp_dif = squeeze(out_ersp_targ-out_ersp_stand);

%save variables to save time

CLim = [-1.5 1.5];
figure;
colormap('redblue') 

% Subplot 1: INSIDE
subplot(3,1,2);
imagesc(times,freqs,ersp_pos,CLim);
title('ERSP:Targets, positive @Pz');
set(gca,'Ydir','Normal')
line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
line([300 700],[7 7],'Color','k','LineStyle','-','LineWidth',.1)
line([300 700],[13 13],'Color','k','LineStyle','-','LineWidth',.1)
line([300 300],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
line([700 700],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
ylabel('Freq (Hz)');
colorbar
% Subplot 1: in stand
subplot(3,1,1);
imagesc(times,freqs,ersp_neg,CLim);
title('ERSP:Standards, negative');
set(gca,'Ydir','Normal')
line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
line([300 700],[7 7],'Color','k','LineStyle','-','LineWidth',.1)
line([300 700],[13 13],'Color','k','LineStyle','-','LineWidth',.1)
line([300 300],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
line([700 700],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
ylabel('Freq (Hz)');
colorbar
%Subplot 1: in difference
subplot(3,1,3);
imagesc(times,freqs,in_ersp_dif,CLim);
title('ERSP:Targets-Standards, Inside');
set(gca,'Ydir','Normal')
line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
line([300 700],[7 7],'Color','k','LineStyle','-','LineWidth',.1)
line([300 700],[13 13],'Color','k','LineStyle','-','LineWidth',.1)
line([300 300],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
line([700 700],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
ylabel('Freq (Hz)'); xlabel('Time (ms)');
colorbar

%%%%%%%%%%
%OUTSIDE
%%%%%%%%%%
CLim = [-1.5 1.5];
figure;
colormap('redblue')

% Subplot 1: TARG
subplot(3,1,2);
imagesc(times,freqs,out_ersp_targ,CLim);
title('ERSP:Targets, Outside @ Pz');
set(gca,'Ydir','Normal')
line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
line([300 700],[7 7],'Color','k','LineStyle','-','LineWidth',.1)
line([300 700],[13 13],'Color','k','LineStyle','-','LineWidth',.1)
line([300 300],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
line([700 700],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
ylabel('Freq (Hz)');
colorbar
% Subplot 1: stan
subplot(3,1,1);
imagesc(times,freqs,out_ersp_stand,CLim);
title('ERSP:Standards, Outside');
set(gca,'Ydir','Normal')
line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
line([300 700],[7 7],'Color','k','LineStyle','-','LineWidth',.1)
line([300 700],[13 13],'Color','k','LineStyle','-','LineWidth',.1)
line([300 300],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
line([700 700],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
ylabel('Freq (Hz)');
colorbar
% Subplot 1: tar-stan out
subplot(3,1,3);
imagesc(times,freqs,out_ersp_dif,CLim);
title('ERSP:Targets-Standards, Outside');
set(gca,'Ydir','Normal')
line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
line([300 700],[7 7],'Color','k','LineStyle','-','LineWidth',.1)
line([300 700],[13 13],'Color','k','LineStyle','-','LineWidth',.1)
line([300 300],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
line([700 700],[7 13],'Color','k','LineStyle','-','LineWidth',.1)
ylabel('Freq (Hz)'); xlabel('Time (ms)');
colorbar
% 
% %in out  difference wave 
% InOut_grand_diff_pz = squeeze(in_ersp_dif-out_ersp_dif);
% figure;
% CLim = [-1.5 1.5];
% colormap('redblue')
% subplot(3,1,1);
% imagesc(times,freqs,InOut_grand_diff_pz,CLim);
% title('ERSP:In-Out, Target-Standards');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[12 12],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)'); xlabel('Time (ms)');
% colorbar

%%
%******************************
%    BAR PLOTS - TO AVERAGE ALPHA POWER IN AN EARLY AND LATE TIME WINDOWS
%*******************************
% (participants x conditions x events x electrodes x frequencies x timepoints)
%conds =  {'In' 'Out'};
%%targets = 1
%stand =2

alpha_bins = find(freqs >= 7 & freqs <= 13);
%time_window = find(times>-650,1)-1:find(times>-300,1)-2; 
time_window = find(times>300,1)-1:find(times>700,1)-2; %pretrial range: [-500 -100] ...

electrode = 2;

%inside stands 
inside_bar_stan_mean = squeeze(mean(ersp(:,1,2,electrode,alpha_bins,time_window),1));
inside_bar_stan_mplot = mean (mean(inside_bar_stan_mean,2));
%pt_sask_bar_stan_sd = squeeze(std(ersp(:,1,1,electrode,alpha_bins,pretrial_window),1));
inside_bar_stan_sdplot = mean(std (inside_bar_stan_mean)/sqrt(nsubs),2);

% inside targs, M and SD
inside_bar_tar_mean = squeeze(mean(ersp(:,1,1,electrode,alpha_bins,time_window),1));
inside_bar_tar_mplot = mean (mean(inside_bar_tar_mean,2));
%pt_sask_bar_tar_sd = squeeze(std(ersp(:,1,2,electrode,alpha_bins,pretrial_window),1));
inside_bar_tar_sdplot = mean(std (inside_bar_tar_mean)/sqrt(nsubs),2);


%outside stands 
outside_bar_stan_mean = squeeze(mean(ersp(:,2,2,electrode,alpha_bins,time_window),1));
outside_bar_stan_mplot = mean (mean(outside_bar_stan_mean,2));
%pt_sask_bar_stan_sd = squeeze(std(ersp(:,1,1,electrode,alpha_bins,pretrial_window),1));
outside_bar_stan_sdplot = mean(std (outside_bar_stan_mean)/sqrt(nsubs),2);

% outside targs, M and SD
traffic_bar_tar_mean = squeeze(mean(ersp(:,2,1,electrode,alpha_bins,time_window),1));
traffic_bar_tar_mplot = mean (mean(traffic_bar_tar_mean,2));
%pt_sask_bar_tar_sd = squeeze(std(ersp(:,1,2,electrode,alpha_bins,pretrial_window),1));
traffic_bar_tar_sdplot = mean(std (traffic_bar_tar_mean)/sqrt(nsubs),2);

%stands
allconds_m_stan = [inside_bar_stan_mplot   outside_bar_stan_mplot];
allconds_sd_stan = [inside_bar_stan_sdplot  outside_bar_stan_sdplot];
%targs
allconds_m_targ = [inside_bar_tar_mplot   traffic_bar_tar_mplot];
allconds_sd_targ = [inside_bar_tar_sdplot  traffic_bar_tar_sdplot];

%********************************************
% PRE/POST PLOTS
%********************************************
close all
conds_plot = {'Inside'; 'Outside';}; 
figure;
subplot (1,2,1)
set(gcf,'color','w');
set(gcf, 'Position',  [100, 500, 1000, 400])
barweb(allconds_m_stan,allconds_sd_stan);
%ylim([-0.2 0.08])
ylim([-0.28 0.12])
ylabel('Power dB')
%title('Pre-trial ERSP power, Standards')
title('Post-trial ERSP power, Standards')
subplot(1,2,2)
barweb(allconds_m_targ,allconds_sd_targ);
%ylim([-0.2 0.08])
ylim([-0.45 0.12])
ylabel('Power dB')
%title('Pre-trial ERSP power, Targets')
title('Post-trial ERSP power, Targets')
legend(conds_plot)
%%
%********************************
%bar plots over frequency

% alpha_bins = find(freqs >= 7 & freqs <= 13);
% %time_window = find(times>-650,1)-1:find(times>-300,1)-2; 
% time_window = find(times>-650,1)-1:find(times>1550,1)-2; %pretrial range: [-500 -100] ...
% 
% electrode = 2;
% 
% %inside stands 
% inside_bar_stan_mean = squeeze(mean(ersp(:,1,2,electrode,alpha_bins,time_window),1));
% inside_bar_stan_mplot = mean (mean(inside_bar_stan_mean,2));
% %pt_sask_bar_stan_sd = squeeze(std(ersp(:,1,1,electrode,alpha_bins,pretrial_window),1));
% inside_bar_stan_sdplot = mean(std (inside_bar_stan_mean)/sqrt(nsubs),2);
% 
% % inside targs, M and SD
% inside_bar_tar_mean = squeeze(median(ersp(:,1,1,electrode,alpha_bins,time_window),1));
% inside_bar_tar_mplot = mean (mean(inside_bar_tar_mean,2));
% %pt_sask_bar_tar_sd = squeeze(std(ersp(:,1,2,electrode,alpha_bins,pretrial_window),1));
% inside_bar_tar_sdplot = mean(std (inside_bar_tar_mean)/sqrt(nsubs),2);
% 
% 
% %outside stands 
% outside_bar_stan_mean = squeeze(median(ersp(:,2,2,electrode,alpha_bins,time_window),1));
% outside_bar_stan_mplot = mean (mean(outside_bar_stan_mean,2));
% %pt_sask_bar_stan_sd = squeeze(std(ersp(:,1,1,electrode,alpha_bins,pretrial_window),1));
% outside_bar_stan_sdplot = mean(std (outside_bar_stan_mean)/sqrt(nsubs),2);
% 
% % outside targs, M and SD
% traffic_bar_tar_mean = squeeze(median(ersp(:,2,1,electrode,alpha_bins,time_window),1));
% traffic_bar_tar_mplot = mean (mean(traffic_bar_tar_mean,2));
% %pt_sask_bar_tar_sd = squeeze(std(ersp(:,1,2,electrode,alpha_bins,pretrial_window),1));
% traffic_bar_tar_sdplot = mean(std (traffic_bar_tar_mean)/sqrt(nsubs),2);
% 
% %stands
% allconds_m_stan = [inside_bar_stan_mplot   outside_bar_stan_mplot];
% allconds_sd_stan = [inside_bar_stan_sdplot  outside_bar_stan_sdplot];
% %targs
% allconds_m_targ = [inside_bar_tar_mplot   traffic_bar_tar_mplot];
% allconds_sd_targ = [inside_bar_tar_sdplot  traffic_bar_tar_sdplot];
% 
% %********************************************
% % PRE/POST PLOTS
% %********************************************
% close all
% conds_plot = {'Inside'; 'Outside';}; 
% figure;
% subplot (1,2,1)
% set(gcf,'color','w');
% set(gcf, 'Position',  [100, 500, 1000, 400])
% barweb(allconds_m_stan,allconds_sd_stan);
% %ylim([-0.2 0.08])
% ylim([0 0.2])
% ylabel('Power dB')
% %title('Pre-trial ERSP power, Standards')
% title('Alpha power over time window, Standards')
% subplot(1,2,2)
% barweb(allconds_m_targ,allconds_sd_targ);
% %ylim([-0.2 0.08])
% ylim([-0.35 0.2])
% ylabel('Power dB')
% %title('Pre-trial ERSP power, Targets')
% title('Alpha power over time window, Targets')
% legend(conds_plot)


%%
%*****************************************

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%************************************************
% FMUT STATS ARE DONE HERE. TO CONDUCT A REPEATED MEASURES ANOVA AT EACH TIME POINT,...
...USING 10,000 PERMUTATIONS. 
%*THE test_results_<conditionname> structure, hold the h, p and, fmax and f values for each...
...time point for which this anova run. 

%****** reminder not to mess up the conditions
%targets = 1
%stand =2

%**************************************************
%averaging ERSP windows and subtracting targets-standards
% (participants x conditions x events x electrodes x frequencies x timepoints)
% FMUTdata  - 4D matrix of electrode x time points x conditions x subjects
%**********************************************************

nperm = 1e4; %Number of permutations
alpha = 0.05;
size (ersp)
%ersp_stat = permute(ersp,[4 6 2 1 3 5]);
%% 
%**********************************************
%first testing for the alpha average 
% Averaging by the alpha frequency, the remaining 5D alpha_ersp dimensions are...
...then changed using 'permute' to fit into the 4D or 5D Matrix used for the...
...fmut analysis. 

% **************************************
%              IMPORTANT
% calc_Fmax needs the dims, nperm and alpha parameters to run the ANOVAS
%this is the main function to create the test_result_<condname> variable that contains...
...the h,p, f and other relevant stats at each time point. 
%***************************************************


%remember: ERSP % (participants x conditions x events x electrodes x frequencies x timepoints)
alpha_bins = find(freqs >= 7 & freqs <= 13);
alpha_ersp  =  squeeze(mean(ersp(:,:,:,:,alpha_bins,:),5));

size (alpha_ersp)     %subs x conds x events, electrodes x timepoints    
alpha_ersp_fmut = permute(alpha_ersp,[4 5 2 3 1]); %inthis case, elec x time...
...in/out x targ/stan, subs

size (alpha_ersp_fmut)     %4D matrix of electrode x time points x conditions x subjects
% dims = [3 4]; 
% test_results_alpha = calc_Fmax(alpha_ersp_fmut,[],dims,nperm,alpha); 
% test_results_alpha.h
%********
%early vs late alpha 
%early_window = find(times>-650,1)-1:find(times>-300,1)-2; 
%earliest_window = find(times>-500,1)-1:find(times>-100,1)-2; 
%imme_window = find(times>0,1)-1:find(times>300,1)-2;

% electrode = 1;
%test1 = alpha_ersp_fmut(electrode,early_window,:,:,:); 
%alpha_earliest = alpha_ersp_fmut(electrode,earliest_window,:,:,:); 
%alpha_imme = alpha_ersp_fmut(electrode,imme_window,:,:,:); 
%clear electrode 

% test_results_alpha_early = calc_Fmax(alpha_early,[],dims,nperm,alpha);  
% test_results_alpha_earliest = calc_Fmax(alpha_earliest,[],dims,nperm,alpha);
% test_results_alpha_imme = calc_Fmax(alpha_imme,[],dims,nperm,alpha); 

electrode = 2;
late_window = find(times>300,1)-1:find(times>700,1)-2;
alpha_late = alpha_ersp_fmut(electrode,late_window,:,:,:); %sig in several timepoints 
size (alpha_late)

dims = [3 4];
test_results_alpha_late = calc_Fmax(alpha_late,[],dims,nperm,alpha); 
test_results_alpha_late.h %SIG
test_results_alpha_late.p


%%
%Averaged over whole time window. 

%remember: ERSP % (participants x conditions x events x electrodes x frequencies x timepoints)
alpha_bins = find(freqs >= 7 & freqs <= 13);
epoch_ersp  =  squeeze(mean(ersp(:,:,:,:,:,:),6));

size (epoch_ersp)     %subs1 x conds2 x events3, electrodes4 x freqs5(treat it like timepoints)
alpha_ersp_fmut = permute(epoch_ersp,[4 5 2 3 1]); %inthis case, elec x time...
...in/out x targ/stan, subs
size (alpha_ersp_fmut)     %4D matrix of electrode x time points x conditions x subjects


%********
dims = [3 4]; 
electrode =2;
test1 = alpha_ersp_fmut(electrode,alpha_bins,:,:,:); 
size (test1) %electrode x freqs x conds x events x subs 

test_results = calc_Fmax(test1,[],dims,nperm,alpha); 
test_results.h %sig
test_results.p


% %%
% %********************************************************
% % targs and standards separately 
% %targets = 1
% %stand =2
% %**********************************
% alpha_ersp_targ  =  squeeze(mean(ersp(:,:,1,:,alpha_bins,:),5)); %target is 2 in the rest of studies but 1 here....
% size (alpha_ersp_targ) %subs x conds x electrodes x timepoints    
% alpha_ersp_targ_fmut = permute(alpha_ersp_targ,[3 4 2 1 ]);% FMUTdata...
% dims = 3; 
% test_results_targ = calc_Fmax(alpha_ersp_targ_fmut,[],dims,nperm,alpha);
% clear dims 
% 
% alpha_ersp_stand  =  squeeze(mean(ersp(:,:,2,:,alpha_bins,:),5));
% size (alpha_ersp_stand) %subs x conds x  electrodes x timepoints    
% alpha_ersp_stand_fmut = permute(alpha_ersp_stand,[3 4 2 1 ]);% FMUTdata...
% dims = 3; 
% test_results_stand = calc_Fmax(alpha_ersp_stand_fmut,[],dims,nperm,alpha);
% clear dims 
% 
% %*********************************
% %difference ERSP 
% alpha_ersp_dif = squeeze(alpha_ersp(:,:,1,:,:) - alpha_ersp(:,:,2,:,:)) ; %targets (1) - standards (2)
% size (alpha_ersp_dif)     
% alpha_ersp_dif_fmut = permute(alpha_ersp_dif,[3 4 2 1]);
% size (alpha_ersp_dif_fmut)     
% dims = 3; 
% test_results_dif = calc_Fmax(alpha_ersp_dif_fmut,[],dims,nperm,alpha);
% clear dims 



%****************************
% OLD TEXT TO BE CLEANED OR REUSED

%Individual plots
% size(ersp)
%ERSP (participants x conditions x events x electrodes x frequencies x timepoints)

%% ***********************************
% inside
close all
electrode = 1;
for i_sub = 1:nsubs
    figure;
    CLim = [-1.5 1.5];
    colormap('redblue')
    for i_event = 1:length(trialevent)
        subplot (3,1,i_event);
        imagesc(times,freqs,squeeze(ersp(i_sub,1,i_event,1,:,:)),CLim);
        title({['ERSP: ' char(trialevent{i_event}) ', Inside']; char(subs(i_sub))});
        set(gca,'Ydir','Normal')
        line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
        ylabel('Freq (Hz)');
        colorbar
    end
end

% close all
% electrode = 1;
% for i_sub = 1:nsubs
%     figure;
%     CLim = [-1.5 1.5];
%     colormap('redblue')
%         subplot (2,1,1);
%         imagesc(times,freqs,squeeze( mean(ersp(i_sub,1,:,1,:,:),3)),CLim);
%         title({'ERSP: Inside'; char(subs(i_sub))});
%         set(gca,'Ydir','Normal')
%         line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
%         ylabel('Freq (Hz)');
%         colorbar
% end
% 

%% ******************************************************
%Traffic 
close all
electrode = 1;
for i_sub = 1:nsubs
    figure;
    CLim = [-1.5 1.5];
    colormap('redblue')
    for i_event = 1:length(trialevent)
        subplot (3,1,i_event);
        imagesc(times,freqs,squeeze(ersp(i_sub,2,i_event,1,:,:)),CLim);
        title({['ERSP: ' char(trialevent{i_event}) ', Outside']; char(subs(i_sub))});
        set(gca,'Ydir','Normal')
        line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
        ylabel('Freq (Hz)');
        colorbar
    end
end

% close all
% electrode = 1;
% for i_sub = 1:nsubs
%     figure;
%     CLim = [-1.5 1.5];
%     colormap('redblue')
%     subplot (2,1,1);
%     imagesc(times,freqs,squeeze (mean(ersp(i_sub,2,:,1,:,:),3)),CLim);
%     title({['ERSP:Outside']; char(subs(i_sub))});
%     set(gca,'Ydir','Normal')
%     line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
%     ylabel('Freq (Hz)');
%     colorbar
% end


%% *******************************************************************************
% %  THETA PLOTS 
% 
% pref_ersp_stand_fz  =  squeeze(mean(mean(ersp(:,[1,2],1,1,:,:),1),2));
% pref_ersp_targ_fz  =  squeeze(mean(mean(ersp(:,[1,2],2,1,:,:),1),2));
% pref_ersp_dif_fz = squeeze(pref_ersp_targ_fz-pref_ersp_stand_fz);
% 
% npref_ersp_stand_fz = squeeze(mean(mean(ersp(:,[3,4],1,1,:,:),1),2));
% npref_ersp_targ_fz = squeeze(mean(mean(ersp(:,[3,4],2,1,:,:),1),2));
% npref_ersp_dif_fz = squeeze(npref_ersp_targ_fz-npref_ersp_stand_fz);
% 
% 
% CLim = [-1.5 1.5];
% figure;
% colormap('jet')
% 
% % Subplot 1: pref
% subplot(3,1,1);
% imagesc(times,freqs,pref_ersp_stand_fz,CLim);
% title('ERSP:Standards, Preferred @Fz');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)');
% colorbar
% % Subplot 1: in
% subplot(3,1,2);
% imagesc(times,freqs,pref_ersp_targ_fz,CLim);
% title('ERSP:Targets, Preferred');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)');
% colorbar
% % Subplot 1: out-in
% subplot(3,1,3);
% imagesc(times,freqs,pref_ersp_dif_fz,CLim);
% title('ERSP:Targets-Standards, Preferred');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)'); xlabel('Time (ms)');
% colorbar
% 
% %NON-PREF
% CLim = [-1.5 1.5];
% figure;
% colormap('jet')
% 
% % Subplot 1: pref
% subplot(3,1,1);
% imagesc(times,freqs,npref_ersp_stand_fz,CLim);
% title('ERSP:Standards, Non Preferred @ Fz');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)');
% colorbar
% % Subplot 1: in
% subplot(3,1,2);
% imagesc(times,freqs,npref_ersp_targ_fz,CLim);
% title('ERSP:Targets, Non Preferred');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)');
% colorbar
% % Subplot 1: out-in
% subplot(3,1,3);
% imagesc(times,freqs,npref_ersp_dif_fz,CLim);
% title('ERSP:Targets-Standards, Non Preferred');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)'); xlabel('Time (ms)');
% colorbar
% 
% pnp_grand_diff_fz = squeeze(pref_ersp_dif_fz-npref_ersp_dif_fz);
% figure
% subplot(3,1,1);
% imagesc(times,freqs,pnp_grand_diff_fz,CLim);
% title('ERSP:Preferred-Unpreferred, Target-Standards');
% set(gca,'Ydir','Normal')
% line([0 0],[min(freqs) max(freqs)],'Color','m','LineStyle','--','LineWidth',1.5)
% line([min(times) max(times)],[4 4],'Color','k','LineStyle','--','LineWidth',.1)
% line([min(times) max(times)],[8 8],'Color','k','LineStyle','--','LineWidth',.1)
% ylabel('Freq (Hz)'); xlabel('Time (ms)');
% colorbar



