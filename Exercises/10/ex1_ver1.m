clc;
clear all;
%% Generate data

  % Lock random seed
  rng(123,'twister');

  gauss_rnd = @(m,S) m + chol(S)'*randn(size(m));
  rmse = @(x,y) sqrt(mean((x(:)-y(:)).^2));

  % Define parameters
  steps = 256;  % Number of time steps
  w     = 0.5;  % Angular velocity
  q     = 0.01; % Process noise spectral density
  r     = 0.1;  % Measurement noise variance

  % This is the transition matrix
  A = 1;

  % This is the process noise covariance
  Q = 1;

  % This is the measurement model
  H = 1;
  
  % This is the measurement noise covariance
  R = 1;
 
  % This is the true initial value
  x0 = 0; 

  % Simulate data
  X = zeros(size(A,1),steps);  % The true signal
  Y = zeros(size(H,1),steps);  % Measurements
  T = 1:steps;         % Time
  x = x0;
  for k=1:steps
    x = gauss_rnd(A*x,Q);
    y = gauss_rnd(x(1),R);
    X(:,k) = x;
    Y(:,k) = y;
  end
% 
   % Visualize
   figure; clf;
     plot(T,X(1,:),'--',T,Y,'o');
     legend('True signal','Measurements');
     xlabel('Time step'); title('\bf Simulated data')
    
%% Baseline solution

% Baseline solution. The estimates
% of x_k are stored as columns of
% the matrix EST1.
  
% Calculate baseline estimate

    m1 = 0;  % Initialize first step with a guess
    EST1 = zeros(size(A,1),steps);
    for k=1:steps
      m1(1) = Y(k);
      EST1(:,k) = m1;
    end

    % Visualize results
    figure; clf;

     % Plot the signal and its estimate
     plot(T,X(1,:),'--',T,EST1(1,:),'-',T,Y,'o');
     legend('True signal','Estimated signal','Measurements');
     xlabel('Time step'); title('\bf Baseline solution')
   % Compute error
   err1 = rmse(X,EST1)
   
   


%%
k=256;
[W,M,V] = Init_EM(Y',k);
Ln = Likelihood(Y',k,W,M,V); % Initialize log likelihood
Lo = 2*Ln;
%%%% EM algorithm %%%%
niter = 0;
maxiter = 256;
while (abs(100*(Ln-Lo)/Lo)>0.1) & (niter<=maxiter),
    E = Expectation(Y',k,W',M',V); % E-step    
    [W,M,V] = Maximization(Y',k,E);  % M-step
    Lo = Ln;
    Ln = Likelihood(X,k,W,M,V);
    niter = niter + 1;
end 
L = Ln;

