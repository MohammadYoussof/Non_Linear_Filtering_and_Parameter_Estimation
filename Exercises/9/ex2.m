clc;
clear all;

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

%% Bootstrap Solution
 m2 = x0;              % Initialize to true value
 P2 = eye(size(m2,1)); % Some uncertainty 
 N = 10000;
    
 BPF_m = zeros(size(m2,1),steps);
  
 % Initial sample set 
 x_pf  = m2 + zeros(size(m2,1),N) ;
 %For storing the histories
 x_pfh = zeros(size(m2,1),N,length(Y)); % filter history
    
 for k=1:length(Y)
        
  for i=1:N
    x_pf(:,i) = mvnrnd((A(x_pf(:,i)))',Q')';
  end
    
  my = H(x_pf(1,:));
  W  = exp(-1/(2*R)*(Y(k) - my).^2); % Constant discarded
  W  = W ./ sum(W);
    
  % Do resampling
  ind = resampstr(W);
  x_pf   = x_pf(:,ind);      
    
  x_pfh(:,:,k) = x_pf;
  % Mean estimate
  m2 = mean(x_pf,2);
    
  BPF_m(:,k) = m2;
        
end
    
    %Compute error
err2 = rmse(X,BPF_m)
    
    %Plot SIR
figure(3); clf;
    
%Plot the signal and its estimate
plot(T,X(1,:),'--',T,BPF_m(1,:),'-',T,Y,'o');
legend('True signal','Estimated signal','Measurements');
xlabel('Time step'); title('\bf Bootstrap Filter')

%% Backward Smoothing
NS = 100;
sm_ss = zeros(size(m2,1),NS,length(Y));
  
for i=1:NS
    ind = floor(rand * NS + 1);
    xn = x_pfh(:,ind,end);
    sm_ss(:,i,end) = xn;
    for k=length(Y)-1:-1:1
        SX = x_pfh(:,:,k);
        mu = A(SX);
        W  = gauss_pdf(xn,mu,Q);
        W  = W ./ sum(W);
        ind = categ_rnd(W);
        xn = x_pfh(:,ind,k);
        sm_ss(:,i,k) = xn;
    end

end
MMS = squeeze(mean(sm_ss,2));
    
%Compute error
err3 = rmse(X,MMS')
    
%Plot Smoothing output
figure(4); clf;

%Plot the signal and its estimate
plot(T,X(1,:),'--',T,MMS(:,1),'-',T,Y,'o');
legend('True signal','Estimated signal','Measurements');
xlabel('Time step'); title('\bf Backward Smoothening');