clear all; close all; clc
scale_down = 16; % for fast prototyping 
%% first search a good seed design with the scaled down design problem
fb = FilterBankStruct( );
fb.T = 256/scale_down; 
fb.B = 160/scale_down; 
Lh = 5*fb.T;
Lg = 5*fb.T;
fb.tau0 = 4*fb.B -1; 
eta = 1e4;
lambda = 0;
fb.symmetry=[-1,0,0]; 

best_cost = inf;
best_seed = fb;
for num_trial = 1 : 20
    [h, g] = fbd_random_initial_guess(Lh, Lg, fb.B, fb.tau0);
    fb.h = h;   fb.g = g;
    [fb, cost, recon_err, iter] = FilterBankDesign(fb, eta, lambda, 1000);
    fprintf('Trial: %g; cost: %g; reconstruction error: %g; iterations %g\n', num_trial, cost, recon_err, iter)
    if cost < best_cost
        best_cost = cost;
        best_seed = fb;
    end
end

%% then scale up this seed design, and use it as the initial guess
fb = FilterBankStruct( );
fb.T = 256; 
fb.B = 160; 
Lh = 5*fb.T;
Lg = 5*fb.T;
fb.tau0 = 4*fb.B -1;
fb.symmetry=[-1,0,0];
fb.h = kron(best_seed.h, ones(scale_down, 1));
fb.g = kron(best_seed.g, ones(scale_down, 1));
[fb, cost, recon_err, iter] = FilterBankDesign(fb, eta, lambda, 1000);

%% save the final design 
fprintf('Cost: %g; reconstruction error: %g; iterations %g\n', cost, recon_err, iter)
figure; 
plot(fb.h); hold on; plot(fb.g); legend('analysis', 'synthesis')
save my_dft_fb fb