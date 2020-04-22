
%% Generate data

clc;
clear all;

  % Lock random seed
  rng(123,'twister');

  gauss_rnd = @(m,S) m + chol(S)'*randn(size(m));
  rmse = @(x,y) sqrt(mean((x(:)-y(:)).^2));

  % Define parameters
  steps = 100;  % Number of time steps
  w     = 0.5;  % Angular velocity
  q     = 0.01; % Process noise spectral density
  r     = 0.1;  % Measurement noise variance

  % This is the transition matrix
  A = [cos(w)    sin(w)/w; 
       -w*sin(w) cos(w)];

  % This is the process noise covariance
  Q = [0.5*q*(w-cos(w)*sin(w))/w^3 0.5*q*sin(w)^2/w^2;
       0.5*q*sin(w)^2/w^2          0.5*q*(w+cos(w)*sin(w))/w];

  % This is the measurement model
  H = [1 0];
 
  % This is the true initial value
  x0 = [0;0.1]; 

  % Simulate data
  X = zeros(size(A,1),steps);  % The true signal
  Y = zeros(size(H,1),steps);  % Measurements
  T = 1:steps;         % Time
  x = x0;
  for k=1:steps
    x = gauss_rnd(A*x,Q);
    y = gauss_rnd(x(1),r);
    X(:,k) = x;
    Y(:,k) = y;
  end
% 
%   % Visualize
%   figure; clf;
%     plot(T,X(1,:),'--',T,Y,'o');
%     legend('True signal','Measurements');
%     xlabel('Time step'); title('\bf Simulated data')
 

%% Baseline solution

  % Baseline solution. The estimates
  % of x_k are stored as columns of
  % the matrix EST1.
  
  % Calculate baseline estimate
  m1 = [0;1];  % Initialize first step with a guess
  EST1 = zeros(2,steps);
  for k=1:steps
    m1(2) = Y(k)-m1(1);
    m1(1) = Y(k);
    EST1(:,k) = m1;
  end

  % Visualize results
  figure; clf;
  
  % Plot the signal and its estimate
  subplot(2,1,1);
    plot(T,X(1,:),'--',T,EST1(1,:),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf Baseline solution')
  
  % Plot the derivative and its estimate
  subplot(2,1,2);
    plot(T,X(2,:),'--',T,EST1(2,:),'-');
    legend('True derivative','Estimated derivative');
    xlabel('Time step')

  % Compute error
  err1 = rmse(X,EST1)
  
  % Report and pause
  fprintf('This is the base line estimate. Press enter.\n');
%% Kalman Filter Solution
    m2 = [0;1];  % Initialize first step
    P2 = eye(size(A,1)); % Some uncertanty in covariance  
    kf_m = zeros(size(m2,1),size(Y,2)); %Allocate space for mean 
    kf_P = zeros(size(P2,1),size(P2,2),size(Y,2)); %Allocate space for covariance

    % Run Kalman filter
    for k=1:steps
      % Replace these with the Kalman filter equations
      %%Prediction
      m2_pred = A*m2;
      P2_pred = A*P2*A' + Q;
   
      %Update
      vk = Y(k) - H*m2_pred;
      Sk = H*P2_pred*H' + r;
      Kk = P2_pred*H'/Sk;
   
      m2 = m2_pred + Kk*vk;
      P2 = P2_pred - (Kk*Sk*Kk');
       % Store the results
      kf_m(:,k) = m2;
      kf_P(:,:,k) = P2;
        
    end

    % Visualize results
    figure; clf

    % Plot the signal and its estimate
    subplot(2,1,1);
    plot(T,X(1,:),'--',T,kf_m(1,:),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf Kalman filter')

    % Plot the derivative and its estimate
    subplot(2,1,2);
    plot(T,X(2,:),'--',T,kf_m(2,:),'-');
    legend('True derivative','Estimated derivative');
    xlabel('Time step')

    % Compute error
    err2 = rmse(X,kf_m)
    
 %% RTS Smoothing Solution
    ms = kf_m(:,end);
    Ps = kf_P(:,:,end);
    rts_m = zeros(size(m2,1),size(Y,2));
    rts_P = zeros(size(P2,1),size(P2,2),size(Y,2));
    rts_m(:,end) = ms;
    rts_P(:,:,end) = Ps;
    for k=size(kf_m,2)-1:-1:1
      mp = A*kf_m(:,k);
      Pp = A*kf_P(:,:,k)*A'+Q;
      Gk = kf_P(:,:,k)*A'/Pp; 
      ms = kf_m(:,k) + Gk*(ms - mp);
      Ps = kf_P(:,:,k) + Gk*(Ps - Pp)*Gk';
      rts_m(:,k) = ms;
      rts_P(:,:,k) = Ps;
    end
    % Visualize results
    figure; clf

    % Plot the signal and its estimate
    subplot(2,1,1);
    plot(T,X(1,:),'--',T,kf_m(1,:),'-',T,rts_m(1,:),'-',T,Y,'o');
    legend('True signal','Filtered signal','Smoothened signal','Measurements');
 
    xlabel('Time step'); title('\bf Kalman Smoothing')
    % Plot the derivative and its estimate
    subplot(2,1,2);
    plot(T,X(2,:),'--',T,kf_m(2,:),'-',T,rts_m(2,:),'-');
    legend('True derivative','Filtered derivative','Smoothened derivative');
    xlabel('Time step')

    % Compute error
    err3 = rmse(X,rts_m)  