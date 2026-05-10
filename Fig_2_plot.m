%close all;
clear all;
%load 'Divergence_matrix.mat';

delete(gcp('nocreate'));
parpool("threads");

h_a = 0.001;
h_b = 0.0005;

L  = 100;
dx = 0.25;

m = round(L/dx);

if abs(m*dx - L) > 1e-12
    error('Choose dx so that L/dx is an integer.');
end

epsilon = 0.02;

a0 = 0.2; a1 = 1;
b0 = 0;   b1 = 0.14;

T = 100;

% Time step adapted to dx.
% This keeps the same diffusion CFL ratio as the original dx=1, dt=0.01.
dt = 0.01 * dx^2;

% Conservative diffusion stability check for explicit 2D Laplacian.
dt_stab = dx^2 / 4;
if dt > dt_stab
    warning('dt = %g may be too large for dx = %g. Suggested dt <= %g.', ...
        dt, dx, dt_stab);
end

% Adjust N so that final time is exactly T.
N = ceil(T/dt);
dt = T/N;

fprintf('Using dx = %g, m = %d, dt = %.10g, N = %d, final time = %.10g\n', ...
    dx, m, dt, N, N*dt);

Masyvas_diverg = Fig_2_divergence_matrix_computation_fast( ...
    h_a, h_b, dx, m, dt, epsilon, a0, a1, b0, b1, N);

fig1 = figure('Position',[100 1500 800 500],'Color',[1 1 1]);
sub = subplot(1,1,1);

imagesc(a0:h_a:a1, b0:h_b:b1, abs(Masyvas_diverg'));
set(gca,'YDir','normal');

colormap(parula(5));
colorbar('Ticks', [0.1 0.3 0.5 0.7 0.9], ...
    'TickLabels', {'(v)', '(iv)', '(iii)', '(ii)','(i)'});

pos1 = get(sub(1),'Position');
set(sub(1),'Position',[pos1(1)-0.01 pos1(2)-0.01 pos1(3) pos1(4)]);

annotation(fig1,'textbox',[0.89 0.03 0.055 0.055], ...
    'String',{'\it a'}, ...
    'FontName','Times New Roman', ...
    'FontSize',40, ...
    'FitBoxToText','off', ...
    'LineStyle','none');

annotation(fig1,'textbox',[0.01 0.96 0.2 0.055], ...
    'String',{'\it b'}, ...
    'FontName','Times New Roman', ...
    'FontSize',40, ...
    'FitBoxToText','off', ...
    'LineStyle','none');

set(gca,'FontName','Times New Roman','FontSize',28);


function Divergence_matrix = Fig_2_divergence_matrix_computation_fast( ...
    h_a, h_b, dx, m, dt, epsilon, a0, a1, b0, b1, N)

aVals = a0:h_a:a1;
bVals = b0:h_b:b1;

nA = numel(aVals);
nB = numel(bVals);

totalJobs = nA * nB;
Divergence_matrix = -ones(nA,nB);

mu1_init = ones(m);
mu2_init = ones(m);

lambda1_init = zeros(m);
lambda1_init(:,ceil(m/2)+1:m) = 0.9;

lambda2_init = zeros(m);
lambda2_init(1:ceil(m/2),:) = 0.9;

if isempty(gcp('nocreate'))
    parpool("threads");
end

progressCount = 0;
tStart = tic;

q = parallel.pool.DataQueue;
afterEach(q, @updateProgress);

fprintf('Starting computation: %d parameter pairs, %d time steps each.\n', ...
    totalJobs, N);

inv_dx2 = 1 / dx^2;
inv_eps = 1 / epsilon;

parfor idx = 1:totalJobs

    [i,j] = ind2sub([nA,nB],idx);

    a = aVals(i);
    b = bVals(j);

    mu1 = mu1_init;
    mu2 = mu2_init;

    lambda1 = lambda1_init;
    lambda2 = lambda2_init;

    maximum = max(abs(mu2),[],'all');

    inv_a = 1 / a;

    for n = 1:N

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

        curmax = max(abs(mu2),[],'all');

        if curmax > maximum
            maximum = curmax;
        end
    end

    finalmax = max(abs(mu2),[],'all');

    if finalmax >= 1000 || any(isnan(mu2),'all')
        val = 1;
    elseif maximum >= 1000
        val = 0.75;
    elseif maximum >= 100
        val = 0.5;
    elseif maximum >= 10
        val = 0.25;
    else
        val = 0;
    end

    Divergence_matrix(idx) = val;

    send(q, [idx, i, j, a, b]);
end

fprintf('Computation finished.\n');

    function updateProgress(data)

        idx = data(1);
        i   = data(2);
        j   = data(3);
        a   = data(4);
        b   = data(5);

        progressCount = progressCount + 1;

        elapsed = toc(tStart);
        frac = progressCount / totalJobs;
        eta = elapsed * (1 - frac) / frac;

        fprintf(['Done %d / %d jobs (%.2f%%) | ', ...
                 'current idx=%d, i=%d, j=%d | ', ...
                 'a=%.5f, b=%.5f | ', ...
                 'elapsed %.2f min | ETA %.2f min\n'], ...
                 progressCount, totalJobs, 100*frac, ...
                 idx, i, j, a, b, ...
                 elapsed/60, eta/60);
    end

end


function L = lattice_fast(U, inv_dx2)

L = zeros(size(U));

% Interior points
L(2:end-1,2:end-1) = ...
    U(2:end-1,1:end-2) + U(2:end-1,3:end) + ...
    U(1:end-2,2:end-1) + U(3:end,2:end-1) - ...
    4*U(2:end-1,2:end-1);

% Left and right boundaries
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

% Top and bottom boundaries
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

% Corners
L(1,1) = ...
    2*U(1,2) + 2*U(2,1) - 4*U(1,1);

L(1,end) = ...
    2*U(1,end-1) + 2*U(2,end) - 4*U(1,end);

L(end,1) = ...
    2*U(end,2) + 2*U(end-1,1) - 4*U(end,1);

L(end,end) = ...
    2*U(end,end-1) + 2*U(end-1,end) - 4*U(end,end);

L = L * inv_dx2;

end
