%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Supplemental Matlab code for Exercise 4.3
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Generate data

  % Lock seed
  randn('state',123);

  % Implement RMSE (true and estimate)
  rmse = @(X,EST) sqrt(mean(sum((X-EST).^2)));
  
  % Create a bit curved trajectory and angle
  % measurements from two sensors
  S1 = [-1.5;0.5]; % Position of sensor 1
  S2 = [1;1];      % Position of sensor 2
  sd = 0.05;       % Standard deviation of measurements
  dt = 0.01;       % Sampling period
  x0 = [0;0;1;0];  % Initial state

  a = zeros(1,500);
  a(1,50:100)  = pi/2/51/dt + 0.01*randn(1,51);
  a(1,200:250) = pi/2/51/dt + 0.01*randn(1,51);
  a(1,350:400) = pi/2/51/dt + 0.01*randn(1,51);
  x = x0;
  t = 0;
  X = [];
  Theta = [];
  T = [];
  for i=1:500
    F = [0 0  1    0;...
         0 0  0    1;...
         0 0  0   a(i);...
         0 0 -a(i) 0];
    x = expm(F*dt)*x;
    y1 = atan2(x(2)-S1(2), x(1)-S1(1)) + sd * randn;
    y2 = atan2(x(2)-S2(2), x(1)-S2(1)) + sd * randn;
    t  = t + dt;
    X = [X x];
    T = [T t];
    Theta = [Theta [y1;y2]];
  end
  steps = size(Theta,2);


%% Dynamic model

  % Parameters of the dynamic model
  qc = 0.1;
  
  % This is the transition matrix
  A  = [1 0 dt 0;
        0 1 0 dt;
        0 0 1 0;
        0 0 0 1];

  % This is the process noise covariance
  Q = [qc*dt^3/3 0 qc*dt^2/2 0;
       0 qc*dt^3/3 0 qc*dt^2/2;
       qc*dt^2/2 0 qc*dt 0;
       0 qc*dt^2/2 0 qc*dt];
  
   
%% Baseline solution
   
  % Baseline solution. The estimates
  % of x_k are stored as columns of
  % the matrix EST1.

  %if 0 %% <--- Uncomment to disable
  
  fprintf('Running base line solution.\n');
    
  % Initialize to true value
  m1 = x0;     
  EST1 = zeros(4,steps);
  
  % Set up figure
  %figure(1); clf
  
  % Loop through steps
  for k=1:steps
      
    % Compute crossing of the measurements
    dx1 = cos(Theta(1,k));
    dy1 = sin(Theta(1,k));
    dx2 = cos(Theta(2,k));
    dy2 = sin(Theta(2,k));
    d = [dx1 dx2; dy1 dy2]\[S2(1)-S1(1);S2(2)-S1(2)];
    
    % Crossing
    cross_xy = S1 + [dx1;dy1]*d(1);
    
    % Compute estimate
    m1(3:4) = [0;0];
    m1(1:2) = cross_xy;
    EST1(:,k) = m1;
    
    % Animate
    if rem(k,10) == 1
      len = 3;
      dx1 = len*cos(Theta(1,k));
      dy1 = len*sin(Theta(1,k));
      dx2 = len*cos(Theta(2,k));
      dy2 = len*sin(Theta(2,k));
      clf;
      plot(X(1,:),X(2,:),'r-',...
            m1(1),m1(2),'bo',...
            EST1(1,1:k),EST1(2,1:k),'b--',...
            [S1(1);S1(1)+dx1],[S1(2);S1(2)+dy1],'k--',...
            [S2(1);S2(1)+dx2],[S2(2);S2(2)+dy2],'k--');
       axis([-2 2 -2.5 1.5]);
      
      % Pause and draw
      drawnow;
      pause(.1)
    end
  end
    
  % Compute error
  err1 = rmse(X,EST1)

  % Plot baseline
   figure(1); clf
 
     plot(X(1,:),X(2,:),'--',...
          EST1(1,:),EST1(2,:),'-',...
          S1(1),S1(2),'kx',S2(1),S2(2),'ko')
     legend('True trajectory','Baseline estimate','Sensor 1','Sensor 2');
     xlabel('x'); ylabel('y'); title('\bf Baseline Solution')
     axis([-2 2 -2.5 1.5]);
   
     fprintf('This is the BL solution. Press enter.\n');
     %pause;
   
  %end %% <--- Uncomment to disable
%%
%Bootstrap solution

  m2 = x0;% Initialize to true value
  R  = sd^2*eye(2);   % The joint covariance
  EST2 = zeros(4,steps);
  
  % Bootstrap design
  N = 1000;
  w = zeros(1,N);
  
  x_pf = zeros(4,N)+m2;
  my = zeros(2,N);
  
  % measurement model function
  hi = @(p, s) atan2(p(2)-s(2),p(1)-s(1));
  % Loop through steps
  
  for k=1:steps
    % draw from dynamic model
   
    % update from sensors    
    for i=1:N
        x_pf(:,i) = mvnrnd((A*x_pf(:,i))',Q')';
    end
    %x_pf = gauss_rndN(A*x_pf,Q);
    % calculate new weights
    for i=1:N
        my(:,i) = [hi(x_pf(1:2,i),S1); hi(x_pf(1:2,i),S2)];
    end
    
    % multivariate normal distribution
    y_pf = [Theta(1,k); Theta(2,k)];
    
    for i=1:N
        w(i)  = exp((-0.5)*((y_pf - my(:,i))'/R*(y_pf - my(:,i)))); % Constant discarded   
    end
    w  = w ./ sum(w);
    
    % resample
    idx = resampstr(w);
    x_pf = x_pf(:,idx);
    
    m2 = mean(x_pf,2);
    EST2(:,k) = m2;
    
    % Animate
    if rem(k,10) == 1
      len = 3;
      dx1 = len*cos(Theta(1,k));
      dy1 = len*sin(Theta(1,k));
      dx2 = len*cos(Theta(2,k));
      dy2 = len*sin(Theta(2,k));
      clf;
      plot(X(1,:),X(2,:),'r-',...
           m2(1),m2(2),'bo',...
           EST2(1,1:k),EST2(2,1:k),'b--',...
           [S1(1);S1(1)+dx1],[S1(2);S1(2)+dy1],'k--',...
           [S2(1);S2(1)+dx2],[S2(2);S2(2)+dy2],'k--');
      axis([-2 2 -2.5 1.5]);
      
      % Pause and draw
      drawnow;
      pause(.1)
    end
  end

  % Compute error
  err2 = rmse(X,EST2)
  
  % Plot Bootstrap
  figure(1); clf

    plot(X(1,:),X(2,:),'--',...
         EST2(1,:),EST2(2,:),'-',...
         S1(1),S1(2),'kx',S2(1),S2(2),'ko')
    legend('True trajectory','Bootstrap estimate','Sensor 1','Sensor 2');
    xlabel('x'); ylabel('y'); title('\bf Bootstrap Solution')
    axis([-2 2 -2.5 1.5]);
  
  
  fprintf('This will be the Bootstrap solution. Press enter.\n');
  %pause;
%%
 %SIR with CKF
  fprintf('Running SIR with CKF solution.\n');

  %m6 = x0;            % Initialize to true value
  P3 = eye(4);        % Some uncertainty
  R  = sd^2*eye(2);   % The joint covariance
  EST3 = zeros(4,steps);
  
  % CKF design
    n = 4;
    
    % storage for sigma points
    nsp = 2*n;
    
    xsp = zeros(n,nsp);
    xsu = zeros(n,nsp);

    % storage for xhat and yhat
    xhat = zeros(n, nsp);
    yhat = zeros(1, nsp);
    
    % measurement model function
    hi = @(p, s) atan2(p(2)-s(2),p(1)-s(1));
    
  % particle filter design
    N = 1000;
    adaptive_resamp = N/2;
    
    my = zeros(2,N);
    sy = R;
    
    % draw N samples from the prior
    x_pf = mvnrnd(x0,P3,N)';
    P_pf = repmat(P3,1,N);
    
    % set initial weights
    logw = ones(1,N)*-log(N);
    
    % calculate initial mean
    m3 = mean(x_pf,2);
  
  % Set up figure
  figure(1); clf
  
  % Loop through steps
  for k=1:steps
     
    for i=1:N
        % create the importance distribution

        % prediction step

        % form the sigma points
        xsp = set_cubature_sigma_points(x_pf(:,i),P_pf(:,(n*i-n+1):n*i),n);

        % propagate through dynamic model
        for j=1:nsp
            xhat(:,j) = A*xsp(:,j);
        end

        % compute predicted mean and covariance
        mm = cubature_mean(xhat,nsp);
        Pm = cubature_covariance(xhat,mm,xhat,mm,nsp) + Q;

        % update step

        % form the sigma points
        xsu = set_cubature_sigma_points(mm,Pm,n);

        % propagate through measurement model
        for j=1:nsp
            yhat(:,j) = hi(xsu(1:2,j),S1);
        end

        mu = cubature_mean(yhat,nsp);

        S = cubature_covariance(yhat,mu,yhat,mu,nsp) + R(1,1);
        C = cubature_covariance(xsu,mm,yhat,mu,nsp);

        K = C/S;

        mm = mm + K*(Theta(1,k) - mu);
        Pm = Pm - K*S*K';

        % form the sigma points again
        xsu = set_cubature_sigma_points(mm,Pm,n);

        % propagate through measurement model
        for j=1:nsp
            yhat(:,j) = hi(xsu(1:2,j),S2);
        end

        mu = cubature_mean(yhat,nsp);
        S = cubature_covariance(yhat,mu,yhat,mu,nsp) + R(2,2);
        C = cubature_covariance(xsu,mm,yhat,mu,nsp);

        K = C/S;

        ID_m = mm + K*(Theta(2,k) - mu);
        P_pf(:,(n*i-n+1):n*i) = Pm - K*S*K';
        
        % draw new sample from the importance distribution
        x_pf(:,i) = mvnrnd(ID_m', P_pf(:,(n*i-n+1):n*i)')';
        
    end
    
    % evaluate likelihood
    for i=1:N
        my(:,i) = [hi(x_pf(1:2,i), S1); hi(x_pf(1:2,i), S2)];
    end
    
    y_pf = [Theta(1,k); Theta(2,k)];
    
    prev_logw = logw;
    
    % store likelihood to w in log scale for better resolution
    for i = 1:N
        logw(i) = -0.5*(y_pf - my(:,i))'/sy*(y_pf - my(:,i));
    end
    
    % update weights and normalise, log sum is multiplication
    logw = prev_logw + logw;
    w = exp(logw);
    w = w./sum(w);
    
    % check if we need resampling
    
    n_eff = 1/sum(w.^2);
    
    %logw = log(w);
    if (n_eff < adaptive_resamp)
        idx = resampstr(w);
        x_pf = x_pf(:,idx);
        logw = ones(1,N)*-log(N);
    end
    
    m3 = sum(w.*x_pf,2);
    
    EST3(:,k) = m3;
    
    % Animate
    if rem(k,10) == 1
      len = 3;
      dx1 = len*cos(Theta(1,k));
      dy1 = len*sin(Theta(1,k));
      dx2 = len*cos(Theta(2,k));
      dy2 = len*sin(Theta(2,k));
      clf;
      plot(X(1,:),X(2,:),'r-',...
           m3(1),m3(2),'bo',...
           EST3(1,1:k),EST3(2,1:k),'b--',...
           [S1(1);S1(1)+dx1],[S1(2);S1(2)+dy1],'k--',...
           [S2(1);S2(1)+dx2],[S2(2);S2(2)+dy2],'k--');
      axis([-2 2 -2.5 1.5]);
      
      % Pause and draw
      drawnow;
      pause(.1)
    end
  end

  % Compute error
  err3 = rmse(X,EST3)
  
  % Plot EKF
  figure(1); clf

    plot(X(1,:),X(2,:),'--',...
         EST3(1,:),EST3(2,:),'-',...
         S1(1),S1(2),'kx',S2(1),S2(2),'ko')
    legend('True trajectory','SIR with CKF estimate','Sensor 1','Sensor 2');
    xlabel('x'); ylabel('y'); title('\bf IR with CKF Solution')
    axis([-2 2 -2.5 1.5]);
  
  
  fprintf('This will be the SIR with CKF solution. Press enter.\n');