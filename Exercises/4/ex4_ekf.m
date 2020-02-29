%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ex-04: (1). Part (b). Implement EKF for the given non-linear
%                       system
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data
  % Lock random seed
  rng(123,'twister');

  gauss_rnd = @(m,S) m + chol(S)'*randn(size(m));
  rmse = @(x,y) sqrt(mean((x(:)-y(:)).^2));

  % Define parameters
  steps = 100;  % Number of time steps

  A = @(x) x - (0.01*sin(x));  % State function
  Ax = @(x) 1 - (0.01*cos(x)); % State Jacobian

  H = @(x) (0.5*sin(2*x));   % Measurement function
  Hx = @(x) cos(2*x);        % Measurement Jacobian

  Q = 0.01^2; % Process noise covariance
  R = 0.02^2;   % Measurement noise covariance

 % This is the true initial value
  x0 = [0;0.1]; 

  % Simulate data
  X = zeros(2,steps);  % The true signal
  Y = zeros(1,steps);  % Measurements
  T = 1:steps;         % Time
  x = x0;
  for k=1:steps
    x = gauss_rnd(A(x),Q);
    y = gauss_rnd(x(1),R);
    X(:,k) = x;
    Y(:,k) = y;
  end

  % Visualize
  figure; clf;
    plot(T,X(1,:),'--',T,Y,'o');
    legend('True signal','Measurements');
    xlabel('Time step'); title('\bf Simulated data')
    
  % Report and pause
  fprintf('This is the simulated data. Press enter.\n');
  pause;

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
  plot(T,X(1,:),'--',T,EST1(1,:),'-',T,Y,'o');
  legend('True signal','Estimated signal','Measurements');
  xlabel('Time step'); title('\bf Baseline solution')

  % Compute error
  err1 = rmse(X,EST1)
  
  % Report and pause
  fprintf('This is the base line estimate. Press enter.\n');
  pause

%% Extended Kalman Filter (EKF)

  m2 = 0;  % Initialize first step
  P2 = eye(1); % Some uncertanty in covariance  
  EST2 = zeros(1,steps); % Allocate space for results

  % Run Kalman filter
  for k=1:steps
    % Replace these with the Kalman filter equations
    
    %%Prediction
    m2_pred = A(m2);
    P2_pred = Ax(m2)*P2*Ax(m2)' + Q;
   
    %Update
    vk = Y(k) - H(m2_pred);
    Sk = Hx(m2_pred)*P2_pred*Hx(m2_pred)' + R;
    Kk = P2_pred*Hx(m2_pred)'/Sk;
    
    m2 = m2_pred + Kk*vk;
    P2 = P2_pred - (Kk*Sk*Kk');
    
    % Store the results
    EST2(:,k) = m2;
  end

  % Visualize results
  figure; clf
  
  % Plot the signal and its estimate
    plot(T,X(1,:),'--',T,EST2(1,:),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf Extended Kalman filter')

  % Compute error
  err2 = rmse(X(1,:),EST2)

  % Report and pause
  fprintf('This will be the EKF estimate. Press enter.\n');
  pause;