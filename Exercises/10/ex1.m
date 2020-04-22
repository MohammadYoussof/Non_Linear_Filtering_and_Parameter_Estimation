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
   

    %% Step 1 : Initial guess
    data = Y;
    temp=randperm(length(data));
    piecap(1)=0.5;
    meucap1(1)=data(temp(1));
    meucap2(1)=data(temp(2));
    sigmacap1(1)=var(data);
    sigmacap2(1)=var(data);
    
    for i = 1:256
    %% Step 2 : Expectation Step; computes the responsibilities
    Qq1=gauss_dist(data,meucap1(i),sigmacap1(i));
    Qq2=gauss_dist(data,meucap2(i),sigmacap2(i));
    log_likelihood(i)=sum(log(((1-piecap(i))*Qq1) + (piecap(i)*Qq2)));
    
    responsibilities(i,:)=(piecap(i)*Qq2)./(((1-piecap(i))*Qq1)+(piecap(i)*Qq2));
    
    %% Step 3 : Maximization Step; compute the weighted means and variances 
    
    meucap1(i+1)=sum((1-responsibilities(i,:)).*data)/sum(1-responsibilities(i,:));
    meucap2(i+1)=sum((responsibilities(i,:)).*data)/sum(responsibilities(i,:));
    
    sigmacap1(i+1)=sum((1-responsibilities(i,:)).*((data-meucap1(i)).^2))/sum(1-responsibilities(i,:));
    sigmacap2(i+1)=sum((responsibilities(i,:)).*((data-meucap2(i)).^2))/sum(responsibilities(i,:));
    
    piecap(i+1)=sum(responsibilities(i,:))/length(data);

    end
    figure
    plot(log_likelihood)
    xlabel('Iteration');
    ylabel('Observed Data Log-likelihood');
    grid minor

%%
%[W,M,V,L] = EM_GM(X',256,0.5,steps,1,[]);