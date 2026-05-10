function Fig_3_plot

clear; 
%close all;

fig1 = figure('Units','normalized','Position',[0.05 0.05 0.55 0.9], ...
              'Color',[1 1 1]);

% ================= PARAMETERS ===============================
epsilon = 0.02; 
a       = 0.9; %0.5; %0.7;%
b       = 0.04; %0.02; %0.05;%

dx = 0.25; 
L  = 100; 
m  = round(L/dx);

if abs(m*dx - L) > 1e-12
    error('Choose dx so that L/dx is an integer.');
end

dt = 0.01*dx^2;

dt_stab = dx^2/4;
if dt > dt_stab
    warning('dt = %g may be too large for dx = %g. Suggested dt <= %g.', dt, dx, dt_stab);
end

snap_times = [5 10 15 20 50 100 200];
snap_steps = round(snap_times / dt);

T_max = 300;
N_max = round(T_max / dt);

% ================= PRECOMPUTED CONSTANTS ====================
inv_dx2 = 1 / dx^2;
inv_eps = 1 / epsilon;
inv_a   = 1 / a;

% ================= INITIAL CONDITIONS =======================
lambda1_init = zeros(m);
lambda1_init(:,ceil(m/2)+1:m) = 0.9;

lambda2_init = zeros(m);
lambda2_init(1:ceil(m/2),:) = 0.9;

mu1_init = ones(m);
mu2_init = ones(m);

mu1 = mu1_init;
mu2 = mu2_init;

lambda1 = lambda1_init;
lambda2 = lambda2_init;

% Snapshot storage
nSnaps = numel(snap_steps) + 1;
snapshots = cell(1, nSnaps);
snap_count = 0;
next_snap_idx = 1;

% Progress setup
tStart = tic;
progressEvery = max(1, round(N_max/100));  % print about 100 updates

fprintf('Starting computation:\n');
fprintf('a = %.5f, b = %.5f, dx = %.5f, m = %d, dt = %.10g\n', ...
    a, b, dx, m, dt);
fprintf('Total iterations: %d, final time: %.5f\n\n', N_max, N_max*dt);

% ================= MAIN LOOP ================================
Max = zeros(1,N_max);

for i = 1:N_max

    % Save old values to preserve explicit Euler update.
    mu1_old = mu1;
    mu2_old = mu2;

    lambda1_old = lambda1;
    lambda2_old = lambda2;

    L_mu1 = lattice_fast(mu1_old, inv_dx2);
    L_l1  = lattice_fast(lambda1_old, inv_dx2);

    f_mu1_val = inv_eps * ( ...
        (1 - 2*lambda1_old) .* ...
        (lambda1_old - inv_a*(lambda2_old + b)) .* mu1_old + ...
        lambda1_old .* (1 - lambda1_old) .* ...
        (mu1_old - inv_a*mu2_old) );

    f_l1_val = inv_eps * lambda1_old .* (1 - lambda1_old) .* ...
        (lambda1_old - inv_a*(lambda2_old + b));

    mu1 = mu1_old + dt * (f_mu1_val + L_mu1);
    mu2 = mu2_old + dt * (mu1_old - mu2_old);

    lambda1 = lambda1_old + dt * (f_l1_val + L_l1);
    lambda2 = lambda2_old + dt * (lambda1_old - lambda2_old);

    Max(i) = max(abs(mu2),[],'all');

    % Save snapshots without plotting
    if next_snap_idx <= numel(snap_steps) && i == snap_steps(next_snap_idx)
        snap_count = snap_count + 1;
        snapshots{snap_count} = mu2;
        fprintf('Saved snapshot at iteration %d / %d, T = %.5f\n', ...
            i, N_max, i*dt);
        next_snap_idx = next_snap_idx + 1;
    end

    % Progress notification
    if mod(i,progressEvery) == 0 || i == 1 || i == N_max
        elapsed = toc(tStart);
        frac = i / N_max;
        eta = elapsed * (1 - frac) / frac;

        fprintf(['Iteration %d / %d (%.2f%%) | ', ...
                 'a = %.5f, b = %.5f | ', ...
                 'T = %.5f / %.5f | ', ...
                 'max|mu2| = %.5g | ', ...
                 'elapsed %.2f min | ETA %.2f min\n'], ...
                 i, N_max, 100*frac, ...
                 a, b, ...
                 i*dt, N_max*dt, ...
                 Max(i), ...
                 elapsed/60, eta/60);
    end
end

% - Vizualization of the dynamics of mu2 -
T = 100; 
N = round(T / dt);
Max = Max(1:N);

interv = T/(500*dt);
Mu2_max_vaizdavimui(1) = 1; 

for kk = 2:(1+T/(interv*dt)) 
    Mu2_max_vaizdavimui(kk) = max(Max(interv*(kk-2)+1:interv*(kk-2)+20));
end

[maximum,i_max] = max(Mu2_max_vaizdavimui); 
maximum = round(maximum,2);

runtime_sec = toc(tStart);

% Save final snapshot if not already included
snap_count = snap_count + 1;
snapshots{snap_count} = mu2;

fprintf('\nComputation finished. Runtime: %.3f seconds, %.2f minutes\n', ...
    runtime_sec, runtime_sec/60);

% ================= PLOTTING ================================
sub = gobjects(1,8);

for k = 1:min(8, snap_count)
    sub(k) = subplot(2,4,k,'Parent',fig1);
    imagesc(snapshots{k});
    axis image;
    set(gca,'XTick',[],'YTick',[]);

    % ===== PADIDINAMI GRAFIKAI =====
    pos = get(sub(k),'Position');
    set(sub(k),'Position',[pos(1) pos(2) pos(3)*1.15 pos(4)*1.15]);

    clim_here = caxis;
    max_abs_color = max(abs(clim_here));
    caxis([-max_abs_color max_abs_color]);

    colormap(bluewhitered(256));
    colorbar('FontName','Times New Roman','FontSize',20, 'Color', [0 0 0]);
end

% ================= COSMETICS ================================
pos1 = get(sub(5),'Position'); set(sub(5),'Position',[pos1(1)-0.11 pos1(2)-0.03 pos1(3) pos1(4)]);
pos1 = get(sub(1),'Position'); set(sub(1),'Position',[pos1(1)-0.11 pos1(2)-0.08 pos1(3) pos1(4)]);
pos1 = get(sub(6),'Position'); set(sub(6),'Position',[pos1(1)-0.13 pos1(2)-0.03 pos1(3) pos1(4)]);
pos1 = get(sub(2),'Position'); set(sub(2),'Position',[pos1(1)-0.13 pos1(2)-0.08 pos1(3) pos1(4)]);
pos1 = get(sub(7),'Position'); set(sub(7),'Position',[pos1(1)-0.14 pos1(2)-0.03 pos1(3) pos1(4)]);
pos1 = get(sub(3),'Position'); set(sub(3),'Position',[pos1(1)-0.14 pos1(2)-0.08 pos1(3) pos1(4)]);
pos1 = get(sub(4),'Position'); set(sub(4),'Position',[pos1(1)-0.15 pos1(2)-0.08 pos1(3) pos1(4)]);
pos2 = get(sub(8),'Position'); set(sub(8),'Position',[pos2(1)-0.15 pos2(2)-0.03 pos1(3) pos1(4)]);

annotation(fig1,'textbox',[0.07 0.025 0.055 0.055],'String',{'(e)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.07 0.45 0.055 0.055],'String',{'(a)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.245 0.025 0.055 0.055],'String',{'(f)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.245 0.45 0.055 0.055],'String',{'(b)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.45 0.025 0.055 0.055],'String',{'(g)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.45 0.45 0.055 0.055],'String',{'(c)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.645 0.025 0.055 0.055],'String',{'(h)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');
annotation(fig1,'textbox',[0.645 0.45 0.055 0.055],'String',{'(d)'},'FontName', ...
           'Times New Roman','FontSize',24,'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');

% -------------- vaizduoja iteracijas iki T=100 ----------------------
fig3 = figure('Units','normalized','Position',[0.5 0.1 0.25 0.5],'Color',[1 1 1]);

set(gca, 'TickLabelInterpreter', 'latex');
xticks_array = [0+1 25+1 50+1 75+1 100+1 250+1 500];
xticks_label_array = {'0' '5' '10' '15' '20' '50' '100'};

sub1_1 = subplot(1,1,1);
ax = gca;
ax.Color = [1 1 1];
ax.XColor = [0 0 0];
ax.YColor = [0 0 0];
ax.FontSize = 22;
ax.FontName = 'Times New Roman';
yyaxis left
plot(Mu2_max_vaizdavimui,'LineWidth',1.5,'LineStyle','-' ,'Color',[0 0 0]);
ax = gca;
ax.YColor = [0 0 0];
y_lim = ylim;  
y_lim(2) = maximum;
hold on; 
plot([i_max i_max],y_lim,'LineStyle','--' ,'Color',[0 0 0])

xlim([0 kk]);
xticks(xticks_array);
xticklabels(xticks_label_array);

yyaxis right
plot(sign(Mu2_max_vaizdavimui).*(log10(1+abs(Mu2_max_vaizdavimui))), ...
    'LineWidth',1.5,'LineStyle','-' ,'Color',[0 0 0.6]);
ax = gca;
ax.YColor = [0 0 0.6];

annotation(fig3,'textbox',[0.01 0.9 0.22 0.055], ...
    'String',{'max |{\it\mu}_2^{(\it{t})}|'}, ...
    'FontName','Times New Roman','FontSize',22, ...
    'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');

annotation(fig3,'textbox',[0.86 0.03 0.07 0.07], ...
    'String',{'\it T  '}, ...
    'FontName','Times New Roman','FontSize',22, ...
    'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');

annotation(fig3,'textbox',[0.71 0.9 0.25 0.055], ...
    'String',{'log_{10} (1+max |{\it\mu}^{(\it{t})}_2|)'}, ...
    'FontName','Times New Roman','FontSize',22, ...
    'Color',[0 0 0.6], ...
    'FitBoxToText','off','Color',[0 0 0],'LineStyle','none');

set(gca,'FontName','Times New Roman','FontSize',22);     
pos1 = get(sub1_1,'Position'); 
set(sub1_1,'Position',[pos1(1)-0.06 pos1(2)-0.02 pos1(3) pos1(4)-0.1]);

% ================= FUNCTIONS ================================

function L = lattice_fast(U, inv_dx2)

L = zeros(size(U));

L(2:end-1,2:end-1) = ...
    U(2:end-1,1:end-2) + U(2:end-1,3:end) + ...
    U(1:end-2,2:end-1) + U(3:end,2:end-1) - ...
    4*U(2:end-1,2:end-1);

L(:,1) = ...
    U(:,2) + U(:,2) + ...
    [U(1,1); U(1:end-1,1)] + ...
    [U(2:end,1); U(end,1)] - ...
    4*U(:,1);

L(:,end) = ...
    U(:,end-1) + U(:,end-1) + ...
    [U(1,end); U(1:end-1,end)] + ...
    [U(2:end,end); U(end,end)] - ...
    4*U(:,end);

L(1,:) = ...
    [U(1,1), U(1,1:end-1)] + ...
    [U(1,2:end), U(1,end)] + ...
    U(2,:) + U(2,:) - ...
    4*U(1,:);

L(end,:) = ...
    [U(end,1), U(end,1:end-1)] + ...
    [U(end,2:end), U(end,end)] + ...
    U(end-1,:) + U(end-1,:) - ...
    4*U(end,:);

L(1,1) = 2*U(1,2) + 2*U(2,1) - 4*U(1,1);
L(1,end) = 2*U(1,end-1) + 2*U(2,end) - 4*U(1,end);
L(end,1) = 2*U(end,2) + 2*U(end-1,1) - 4*U(end,1);
L(end,end) = 2*U(end,end-1) + 2*U(end-1,end) - 4*U(end,end);

L = L * inv_dx2;

end

end
