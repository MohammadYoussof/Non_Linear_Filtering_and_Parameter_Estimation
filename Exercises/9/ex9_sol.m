%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab code for Exercise 9
%
% This software is distributed under the GNU General Public 
% Licence (version 2 or later); please refer to the file 
% Licence.txt, included with the software, for details.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc;clear; close all;
BASELINE = 0;
EKF_KALMAN = 0;
EKF_KALMAN_SMOOTHER = 0;
SIRwEKF = 0;
PFwSIREKF = 0;  
BOOTSTRAP = 1;
BACKWARD_SMOOTHENING = 1;
%% Generate data

  % Lock seed
  rng(123,'twister');

  % Implement RMSE (true and estimate)
  rmse = @(X,EST) sqrt(mean(sum((X-EST).^2)));
  
  %Gaussian random value generator
  gauss_rnd_local = @(m,S) m + chol(S)'*randn(size(m));
  
  % Define parameters
  steps = 100;  % Number of time steps
  % This is the true initial value
  x0 = 0; 

%% Dynamic model
  % This is the transition model
  A = @(x) x - (0.01*sin(x));  % State function
 
  %Measurement Model
  H = @(x) (0.5*sin(2*x));   % Measurement function
  
  q = 0.01^2; % Process noise covariance
  r = 0.02^2;   % Measurement noise covariance
  
  % This is the process noise covariance
  Q = q*eye(size(A,1));
  
  % This is the measurement noise covariance
  R = r*eye(size(H,1));


  % Simulate data
  X = zeros(2,steps);  % The true signal
  Y = zeros(1,steps);  % Measurements
  T = 1:steps;         % Time
  x = x0;
  for k=1:steps
    x = gauss_rnd_local(A(x),Q);
    y = gauss_rnd_local(H(x(1)),R);
    X(:,k) = x;
    Y(:,k) = y;
  end

  % Visualize
  figure(1); clf;
    plot(T,X(1,:),'--',T,Y,'o');
    legend('True signal','Measurements');
    xlabel('Time step'); title('\bf Simulated data')
   
%% Baseline solution
if BASELINE   
  
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
  figure(2); clf;
  
  % Plot the signal and its estimate
  plot(T,X(1,:),'--',T,EST1(1,:),'-',T,Y,'o');
  legend('True signal','Estimated signal','Measurements');
  xlabel('Time step'); title('\bf Baseline solution')

  % Compute error
  err1 = rmse(X,EST1)

end

%% EKF solution
if EKF_KALMAN  
    % EKF solution. The estimates
    % of x_k are stored as columns of
    % the matrix EST2.
    
    m2 = x0;            % Initialize to true value
    P2 = eye(size(m2,1));        % Some uncertainty

    kf_m = zeros(size(m2,1),size(Y,2)); %Allocate space for mean 
    kf_P = zeros(size(P2,1),size(P2,2),size(Y,2)); %Allocate space for covariance
    %Jacobian 
    A_jcb = @(x) 1 - (0.01*cos(x)); % State Jacobian
    H_jcb = @(x) cos(2*x);        % Measurement Jacobian

    % Loop through steps
    for k=1:steps

        %%Prediction
        m2_pred = A(m2);
        P2_pred = A_jcb(m2)*P2*A_jcb(m2)' + Q;

        %Update
        vk = Y(:,k) - H(m2_pred);
        Sk = H_jcb(m2_pred)*P2_pred*H_jcb(m2_pred)' + R;
        Kk = P2_pred*H_jcb(m2_pred)'/Sk;

        m2 = m2_pred + Kk*vk;
        P2 = P2_pred - (Kk*Sk*Kk');

        kf_m(:,k) = m2;
        kf_P(:,:,k) = P2;

    end

    % Compute error
    err2 = rmse(X,kf_m)

    % Plot EKF
    figure(3); clf

    % Plot the signal and its estimate
    plot(T,X(1,:),'--',T,kf_m(1,:),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf Extended Kalman filter')

end 


%% EKF with RTS Smoothening solution
if EKF_KALMAN_SMOOTHER
    if EKF_KALMAN
      % EKF solution. The estimates
      % of x_k are stored as columns of
      % the matrix EST2.

        ms = kf_m(:,end);
        Ps = kf_P(:,:,end);
        rts_m = zeros(size(m2,1),size(Y,2));
        rts_P = zeros(size(P2,1),size(P2,2),size(Y,2));
        rts_m(:,end) = ms;
        rts_P(:,:,end) = Ps;
        for k=size(kf_m,2)-1:-1:1
            mp = A(kf_m(:,k));
            Pp = A_jcb(mp)*kf_P(:,:,k)*A_jcb(mp)'+Q;
            Gk = kf_P(:,:,k)*A_jcb(mp)'/Pp; 
            ms = kf_m(:,k) + Gk*(ms - mp);
            Ps = kf_P(:,:,k) + Gk*(Ps - Pp)*Gk';
            rts_m(:,k) = ms;
            rts_P(:,:,k) = Ps;
        end

        % Compute error
        err3 = rmse(X,rts_m)
        % Plot EKF
        figure(4); clf

        % Plot the signal and its estimate
        plot(T,X(1,:),'--',T,rts_m(1,:),'-',T,Y,'o');
        legend('True signal','Estimated signal smoothed','Measurements');
        xlabel('Time step'); title('\bf Extended Kalman filter w RTS Smoother')

    end    
    
end
  

%% SIR with EKF with Backward Particle Smoothening solution
if SIRwEKF  
%   EKF solution. The estimates
%   of x_k are stored as columns of
%   the matrix EST2.
  
    m3 = x0;            % Initialize to true value
    P3 = eye(size(m3,1));        % Some uncertainty
   
%     Jacobian 
    A_jcb = @(x) 1 - (0.01*cos(x)); % State Jacobian
    H_jcb = @(x) cos(2*x);        % Measurement Jacobian
    
%     Particle Filter
    
%     Store final result
    BS6_m = zeros(size(m3,1),steps);
    
%     Number of particles
    N = 1000;
    adaptive_resamp = N/2;
     
%     draw N samples from the prior
    x_pf = mvnrnd(m3,P3,N)';
    P_pf = repmat(P3,1,N);
    
%     For measurment gaussian
    sy = R;

%     set initial weights
    logWeights = ones(1,N)*-log(N);
    
%     calculate initial mean
    m6 = mean(x_pf,2);

%     Loop through steps
    for k=1:steps
        
%         Create importance distribution for particles
        for i=1:N
            
        %Prediction
        m3_pred = A(x_pf(:,i));
        P3_pred = A_jcb(x_pf(:,i))*P_pf(:,i)*A_jcb(x_pf(:,i))' + Q;

%         Update
        vk = Y(:,k) - H(m3_pred);
        Sk = H_jcb(m3_pred)*P3_pred*H_jcb(m3_pred)' + R;
        Kk = P3_pred*H_jcb(m3_pred)'/Sk;

        m3 = m3_pred + Kk*vk;
        P_pf(:,i) = P3_pred - (Kk*Sk*Kk');
        
%         draw new sample from the importance distribution
        x_pf(:,i) = mvnrnd(m3', P_pf(:,i)')';
           
        end
        
%         evaluate likelihood
        for i=1:N
            my(:,i) = H(x_pf(1,i));
        end
    
        prev_logWeights = logWeights;
    
%         store likelihood to w in log scale for better resolution
        for i = 1:N
            logWeights(i) = -0.5*(Y(k) - my(:,i))'/sy*(Y(k) - my(:,i));
        end
    
%         update weights and normalise, log sum is multiplication
        logWeights = prev_logWeights + logWeights;
        weights = exp(logWeights);
        weights = weights./sum(weights);
    
%         check if we need resampling
    
        n_eff = 1/sum(weights.^2);
    
        logWeights = log(weights);
        if (n_eff < adaptive_resamp)
            idx = resampstr(weights);
            x_pf = x_pf(:,idx);
            logWeights = ones(1,N)*-log(N);
        end
    
        m6 = sum(weights.*x_pf,2);
    
        BS6_m(:,k) = m6;
        
    end
  
%     Compute error
    err4 = rmse(X,BS6_m)
    
%     Plot SIR
    figure(5); clf
    
%     Plot the signal and its estimate
    plot(T,X(1,:),'--',T,BS6_m(1,:),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf SIR Filter with EKF')


end

% SIR Backward Particle Smoothening solution
if PFwSIREKF
    
    
    
    
end

%% Bootstrap Filter
if BOOTSTRAP
    
    m4 = x0;              % Initialize to true value
    P4 = eye(size(m4,1)); % Some uncertainty 
    N = 10000;
    
    BPF_m = zeros(size(m4,1),steps);
  
    % Initial sample set 
    x_pf  = zeros(size(m4,1),N) + m4;
    %For storing the histories
    X_PF_ALL = zeros(size(m4,1),N,length(Y)); % filter history
    
    for k=1:length(Y)

        % Propagate through the dynamic model
        
        for i=1:N
            x_pf(:,i) = mvnrnd((A(x_pf(:,i)))',Q')';
        end
    
        % Compute the weights
        my = H(x_pf(1,:));
        W  = exp(-1/(2*R)*(Y(k) - my).^2); % Constant discarded
        W  = W ./ sum(W);
    
        % Do resampling
        ind = resampstr(W);
        x_pf   = x_pf(:,ind);      
    
        X_PF_ALL(:,:,k) = x_pf;
        % Mean estimate
        m4 = mean(x_pf,2);
    
        BPF_m(:,k) = m4;
        
    end
    
    %Compute error
    err5 = rmse(X,BPF_m)
    
    %Plot SIR
    figure(6); clf;
    
    %Plot the signal and its estimate
    plot(T,X(1,:),'--',T,BPF_m(1,:),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf Bootstrap Filter')

  
    
end

%%
if BACKWARD_SMOOTHENING
    if BOOTSTRAP
%Backward Smoothening

    NS = 100;
    SM_SSX = zeros(size(m4,1),NS,length(Y));
  
    for i=1:NS
        ind = floor(rand * N + 1);
        xn = X_PF_ALL(:,ind,end);
        SM_SSX(:,i,end) = xn;
        for k=length(Y)-1:-1:1
            SX = X_PF_ALL(:,:,k);
            mu = A(SX);
            W  = gauss_pdf(xn,mu,Q);
            W  = W ./ sum(W);
            

            ind = categ_rnd(W);
            xn = X_PF_ALL(:,ind,k);
            SM_SSX(:,i,k) = xn;
        end
%         if rem(i,10)==0
%             plot(T,X(1,:),'r-',T,squeeze(SM_SSX(1,i,:)),'b--');
%             title(sprintf('Doing backward simulation %d/%d\n',i,NS));
%             drawnow;
%         end
    end

    MMS = squeeze(mean(SM_SSX,2));
    
    %Compute error
    err5 = rmse(X,MMS')
    
    %Plot SIR
    figure(6); clf;
    
    %Plot the signal and its estimate
    plot(T,X(1,:),'--',T,MMS(:,1),'-',T,Y,'o');
    legend('True signal','Estimated signal','Measurements');
    xlabel('Time step'); title('\bf Backward Smoothening')
    end
end    
  
  
  
  
