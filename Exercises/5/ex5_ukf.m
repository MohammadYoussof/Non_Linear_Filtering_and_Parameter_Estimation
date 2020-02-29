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
  %pause;

  %%
% Baseline solution
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
  
  %Plot the signal and its estimate
  plot(T,X(1,:),'--',T,EST1(1,:),'-',T,Y,'o');
  legend('True signal','Estimated signal','Measurements');
  xlabel('Time step'); title('\bf Baseline solution')

  % Compute error
  err1 = rmse(X,EST1)
  
  % Report and pause
  fprintf('This is the base line estimate. Press enter.\n');
  %pause
%%
% Unscented Kalman Filter (EKF)

   m2 = 0;  % Initialize first step
   P2 = eye(1); % Some uncertanty in covariance  
   EST2 = zeros(1,steps); % Allocate space for results
  
  % UKF constants assumption
   n =1 ;

   alpha = 1;
   kappa = 0;
   beta = 0;
   
   lambda = ((alpha^2)*(n+kappa))- n;
   sigma_points = zeros(n,((2*n)+1),1);
   weights_mean = zeros(((2*n)+1),1);
   weights_cov = zeros(((2*n)+1),1);
   weights_mean(1,:) = lambda/(n+lambda);
   weights_cov(1,:) = (lambda/(n+lambda) )+ (1- (alpha^2) + beta);
   
   for i =2:((2*n)+1)
       weights_cov(i,:) = 1/(2*(n+lambda));
       weights_mean(i,:) = 1/(2*(n+lambda));
   end
   
 
  % Run Kalman filter
  for k=1:steps
    % Replace these with the Kalman filter equations
   
    % Sigma point calculation
   sigma_points(:,1)=m2;
   cholesky_factor = cholcov(P2);
   sigma_points(:,2)=m2 + (sqrt(n+lambda)) * cholesky_factor;
   sigma_points(:,3)=m2 - (sqrt(n+lambda)) * cholesky_factor;
%    for i=2:()
%        sigma_points(:,i)= m2 + (sqrt(n+lambda)) * cholesky_factor;
%        sigma_points(:,n+i)= m2 - (sqrt(n+lambda)) * cholesky_factor;
%    end
   sigma_points_hat = A(sigma_points);
   m2_pred =  (sigma_points_hat * weights_mean);
   P2_pred = 0;
   for i=1:((2*n)+1)
       P2_pred = P2_pred + (weights_cov(i,:) * (sigma_points_hat(:,i)-m2_pred)*(sigma_points_hat (:,i)- m2_pred)' + Q);
   end
   
   sigma_points_pred = zeros(n,((2*n)+1),1);
   sigma_points_pred(:,1)= m2_pred;
   cholesky_factor_pred = cholcov(P2_pred);
   sigma_points_pred(:,2)= m2_pred + (sqrt(n+lambda)) * cholesky_factor_pred;
   sigma_points_pred(:,3)= m2_pred - (sqrt(n+lambda)) * cholesky_factor_pred;

%    for i=1:n
%        sigma_points_pred(:,i)= m2_pred + (sqrt(n+lambda)) * cholesky_factor_pred(:,i);
%       sigma_points_pred(:,n+i)= m2_pred - (sqrt(n+lambda)) * cholesky_factor_pred(:,i);
%    end

   Y_hat = H(sigma_points_pred);
   
   
   uk =  Y_hat  * weights_mean ;
   
   
   Sk =0;
   Ck =0;
   
   for i=1:((2*n) +1)
       
       Sk = Sk +((weights_cov(i,:) *(Y_hat(:,i)- uk))*(Y_hat(:,i)-uk)') + R;
       Ck =  Ck+ (weights_cov(i,:) * (sigma_points_pred(:,i) - m2_pred ))* (Y_hat(:,i)- uk)';
   end
   
   
   Kk = Ck*inv(Sk);
   m2 = m2_pred + (Kk*(Y(k)-uk));
   P2 = P2_pred - Kk*Sk*Kk';
   
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
  %pause;