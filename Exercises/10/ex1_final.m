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
  R = 0.5;
 
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

    m1 = 0;  % Initialize first step with a guess
    EST1 = zeros(size(A,1),steps);
    for k=1:steps
      m1(1) = Y(k);
      EST1(:,k) = m1;
    end

   % Compute error
   err1 = rmse(X,EST1);
  
 %% Kalman Filter Estimate and  RTS Smoother
   
    xmin = 0.1;
    xmax = 1.5;
    R = zeros(steps+1,1);
    log_likelihood = zeros(steps,1);
    R(1,:) = xmin + (xmax-xmin).*rand(1,1); % Initial guess
     
    for i=1:steps   
        m2 = 0;  % Initialize first step
        P2 = eye(size(A,1)); % Some uncertanty in covariance  
        kf_m = zeros(size(m2,1),size(Y,2)); %Allocate space for mean 
        kf_P = zeros(size(P2,1),size(P2,2),size(Y,2)); %Allocate space for covariance
        % Run Kalman filter
        for k=1:steps
          %%Prediction
          m2_pred = A*m2;
          P2_pred = A*P2*A' + Q;

          %Update
          vk = Y(k) - H*m2_pred;
          Sk = H*P2_pred*H' + R(i,:);
          Kk = P2_pred*H'/Sk;

          m2 = m2_pred + Kk*vk;
          P2 = P2_pred - (Kk*Sk*Kk');

          % Store the results
          kf_m(:,k) = m2;
          kf_P(:,:,k) = P2;  
        end

        % Compute error
        err2 = rmse(X,kf_m);

        %%%%%%% RTS Smoother

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
        % Compute error
        err3 = rmse(X,rts_m);
        % Compute log likelihood
        sigma = 0;
        for j=2:steps
            sigma = sigma + rts_P(:,:,j) + (rts_m(:,j))*(rts_m(:,j)');
        end
        sigma = sigma/steps;
        phi = 0;
        for j=2:steps
            phi = phi + rts_P(:,:,j-1) + (rts_m(:,j-1))*(rts_m(:,j-1)');
        end
        phi = phi/steps;
        B_l = 0;
        for j=2:steps
            B_l = B_l + (Y(j) * rts_m(:,j)');
        end
        B_l = B_l/steps;
        C_l = 0;
        for j=2:steps
            C_l = C_l + ((rts_P(:,:,j) * Gk) + (rts_m(:,j) * rts_m(:,j-1)'));
        end
        C_l = C_l/steps;
        
        D_l = 0;
        for j=2:steps
            D_l = D_l + (Y(j)*Y(j)');
        end
        D_l = D_l/steps;
        prev_log_likelihood =0;
        log_likelihood(i,:) = (-1/2*log(abs(2*pi*kf_P(:,:,1)))) - ((steps/2)*log(2*pi*Q)) ...
                          -((steps/2)*log(2*pi*R(i,:))) ...
                          - (1/2*trace(inv(kf_P(:,:,1))*(rts_P(:,:,1)+((rts_m(:,1)-kf_m(:,1))*(rts_m(:,1)-kf_m(:,1))'))))...
                          - (steps/2*trace(inv(Q)*(sigma - (C_l*A') - (A*C_l)+ (A*sigma*A'))))...
                          - (steps/2*trace(inv(R(i,:))*(D_l - (B_l*H') - (H*B_l')+ (H*sigma*H'))));

        if i==1
            R(i+1,:) = xmin + (xmax-xmin).*rand(1,1); 
        else
            if(log_likelihood(i-1,:) < log_likelihood(i,:))
                R(i+1,:) = xmin + (xmax-xmin).*rand(1,1); 
            else
                R(i+1,:) = xmin + (xmax-xmin).*rand(1,1);
                log_likelihood(i,:) = log_likelihood(i-1,:);
            end
        end        
    end
 