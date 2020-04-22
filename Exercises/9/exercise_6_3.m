% ELEC-E8105: Exercise 6.3
%
% 2016-03-14 -- Roland Hostettler <roland.hostettler@aalto.fi>

% Housekeeping
clear variables;

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
    F = [
        0 0     1    0;...
        0 0     0    1;
        0 0     0 a(i);
        0 0 -a(i)    0;
    ];
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
A  = [
    1 0 dt  0;
    0 1  0 dt;
    0 0  1  0;
    0 0  0  1;
];

% This is the process noise covariance
Q = [
    qc*dt^3/3         0 qc*dt^2/2         0;
            0 qc*dt^3/3         0 qc*dt^2/2;
    qc*dt^2/2         0     qc*dt         0;
            0 qc*dt^2/2         0     qc*dt;
];

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
figure(1); clf

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
pause;

%end %% <--- Uncomment to disable

%% EKF solution  
% Jacobian %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hx = @(x, y, sx, sy) -(1 + ((y-sy)/(x-sx)).^2).^(-1).*((y-sy)./(x-sx).^2);
Hy = @(x, y, sx, sy) (1 + ((y-sy)/(x-sx)).^2).^(-1).*1./(x-sx);

% Uncomment these lines to verify the Jacobian
% x = 1;
% delta = 0.01;
% y = 2;
% atan2(y-S1(2), x+delta*x - S1(1)) - atan2(y-S1(2), x-S1(1));
% Hx(x, y, S1(1), S1(2))*delta*x;
% 
% atan2(y+delta*y-S1(2), x-S1(1)) - atan2(y-S1(2), x-S1(1));
% Hy(x, y, S1(1), S1(2))*delta*y;

H = @(x) [
    Hx(x(1), x(2), S1(1), S1(2)), Hy(x(1), x(2), S1(1), S1(2)), zeros(1, 2);
    Hx(x(1), x(2), S2(1), S2(2)), Hy(x(1), x(2), S2(1), S2(2)), zeros(1, 2);
];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% EKF solution. The estimates
% of x_k are stored as columns of
% the matrix EST2.

%if 0 %% <--- Uncomment to disable
fprintf('Running EKF solution.\n');

m2 = x0;            % Initialize to true value
P2 = eye(4);        % Some uncertainty
R  = sd^2*eye(2);   % The joint covariance
EST2 = zeros(4,steps);

% Set up figure
figure(2); clf

% Loop through steps
for k=1:steps
    %%%% EKF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Prediction (NOTE: No linearization here, this is accurate!)
    mp2 = A*m2;
    Pp2 = A*P2*A' + Q;

    % Update
    v = [
        Theta(1, k) - atan2(mp2(2)-S1(2), mp2(1)-S1(1));
        Theta(2, k) - atan2(mp2(2)-S2(2), mp2(1)-S2(1));
    ];
    S = H(mp2)*Pp2*H(mp2)' + R;
    K = (Pp2*H(mp2)')/S;
    m2 = mp2 + K*v;
    P2 = Pp2 - K*S*K';    
    %%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Store the estimate
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

% Plot EKF
figure(2); clf
plot(X(1,:),X(2,:),'--',...
     EST2(1,:),EST2(2,:),'-',...
     S1(1),S1(2),'kx',S2(1),S2(2),'ko')
legend('True trajectory','EKF estimate','Sensor 1','Sensor 2');
xlabel('x'); ylabel('y'); title('\bf EKF Solution')
axis([-2 2 -2.5 1.5]);

fprintf('This is the EKF solution. Press enter.\n');
pause;

%end %% <--- Uncomment to disable

%% UKF solution    
% UKF solution. The estimates
% of x_k are stored as columns of
% the matrix EST3.

%if 0 %% <--- Uncomment to disable
fprintf('Running UKF solution.\n');

%%%% UKF Parameters & Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 1e-3;
beta = 2;
kappa = 0;
L = 4;

% Weights
lambda = alpha^2*(L+kappa)-L;
Wm = 1/(2*(L+lambda))*ones(1, 2*L+1);
Wm(1) = lambda/(L+lambda);
Wc = Wm;
Wc(1) = Wc(1) + 1-alpha^2+beta;

h = @(x) [
    atan2(x(2)-S1(2), x(1)-S1(1));
    atan2(x(2)-S2(2), x(1)-S2(1));
];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m2 = x0;            % Initialize to true value
P2 = eye(4);        % Some uncertainty
R  = sd^2*eye(2);   % The joint covariance
EST3 = zeros(4,steps);

% Set up figure
figure(3); clf

% Loop through steps
for k=1:steps
    %%%% UKF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Prediction
    % NOTE: No sigma-points needed, this is the true prediction as the
    % process dynamics are linear
    mp2 = A*m2;
    Pp2 = A*P2*A' + Q;

    %% Update
    % Sigma-points
    Xs = [ ...
        mp2, ...
        mp2*ones(1, L) + sqrt(L + lambda)*chol(Pp2, 'lower'), ...
        mp2*ones(1, L) - sqrt(L + lambda)*chol(Pp2, 'lower'), ...
    ];

    % Predict
    Yp = zeros(2, 2*L+1);
    for l = 1:2*L+1
        Yp(:, l) = h(Xs(:, l));
    end
    yp = (Wm*Yp.').';

    % Covariances
    Sk = R;
    Ck = 0;
    for l = 1:2*L+1
        Sk = Sk + Wc(l)*((Yp(:, l)-yp)*(Yp(:, l)-yp)');
        Ck = Ck + Wc(l)*((Xs(:, l)-mp2)*(Yp(:, l)-yp)');
    end

    % Update
    K = Ck/Sk;
    m2 = mp2 + K*(Theta(:, k)-yp);
    P2 = Pp2 - K*Sk*K';
    %%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Store the estimate
    EST3(:,k) = m2;

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
figure(3); clf
plot(X(1,:),X(2,:),'--',...
EST3(1,:),EST3(2,:),'-',...
S1(1),S1(2),'kx',S2(1),S2(2),'ko')
legend('True trajectory','UKF estimate','Sensor 1','Sensor 2');
xlabel('x'); ylabel('y'); title('\bf UKF Solution')
axis([-2 2 -2.5 1.5]);


fprintf('This is the UKF solution. Press enter.\n');
pause;

%end %% <--- Uncomment to disable

%% CKF solution    
% UKF solution. The estimates
% of x_k are stored as columns of
% the matrix EST4.

%if 0 %% <--- Uncomment to disable
fprintf('Running CKF solution.\n');

%%%% UKF Parameters & Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 1;
beta = 0;
kappa = 0;
L = 4;

% Weights
lambda = alpha^2*(L+kappa)-L;
Wm = 1/(2*(L+lambda))*ones(1, 2*L+1);
Wm(1) = lambda/(L+lambda);
Wc = Wm;
Wc(1) = Wc(1) + 1-alpha^2+beta;

h = @(x) [
    atan2(x(2, :)-S1(2), x(1, :)-S1(1));
    atan2(x(2, :)-S2(2), x(1, :)-S2(1));
];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m2 = x0;            % Initialize to true value
P2 = eye(4);        % Some uncertainty
R  = sd^2*eye(2);   % The joint covariance
EST4 = zeros(4,steps);

% Set up figure
figure(4); clf

% Loop through steps
for k=1:steps

    %%%% CKF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Prediction
    % NOTE: No sigma-points needed, this is the true prediction as the
    % process dynamics are linear
    mp2 = A*m2;
    Pp2 = A*P2*A' + Q;

    %% Update
    % Sigma-points
    Xs = [...
        mp2, ...
        mp2*ones(1, L) + sqrt(L + lambda)*chol(Pp2, 'lower'), ...
        mp2*ones(1, L) - sqrt(L + lambda)*chol(Pp2, 'lower'), ...
    ];

    % Predict
    Yp = zeros(2, 2*L+1);
    for l = 1:2*L+1
        Yp(:, l) = h(Xs(:, l));
    end
    yp = (Wm*Yp.').';

    % Covariances
    Sk = R;
    Ck = 0;
    for l = 1:2*L+1
        Sk = Sk + Wc(l)*((Yp(:, l)-yp)*(Yp(:, l)-yp)');
        Ck = Ck + Wc(l)*((Xs(:, l)-mp2)*(Yp(:, l)-yp)');
    end

    % Update
    K = Ck/Sk;
    m2 = mp2 + K*(Theta(:, k)-yp);
    P2 = Pp2 - K*Sk*K';
    %%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Store the estimate
    EST4(:,k) = m2;

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
             EST4(1,1:k),EST4(2,1:k),'b--',...
             [S1(1);S1(1)+dx1],[S1(2);S1(2)+dy1],'k--',...
             [S2(1);S2(1)+dx2],[S2(2);S2(2)+dy2],'k--');
        axis([-2 2 -2.5 1.5]);

        % Pause and draw
        drawnow;
        pause(.1)
    end
end

% Compute error
err4 = rmse(X,EST4)

% Plot EKF
figure(4); clf
plot(X(1,:),X(2,:),'--',...
     EST4(1,:),EST4(2,:),'-',...
     S1(1),S1(2),'kx',S2(1),S2(2),'ko')
legend('True trajectory','CKF estimate','Sensor 1','Sensor 2');
xlabel('x'); ylabel('y'); title('\bf CKF Solution')
axis([-2 2 -2.5 1.5]);

fprintf('This is the CKF solution. Press enter.\n');
pause;

%end %% <--- Uncomment to disable

%% Bootstrap PF solution    
fprintf('Running bootstrap PF solution.\n');

% Parameters
M = 10000;
M_T = M/3;
P0 = eye(4);
m2 = x0*ones(1, M) + chol(P0, 'lower')*randn(4, M);
w = 1/M*ones(1, M);

%P2 = eye(4);        % Some uncertainty
R  = sd^2*eye(2);   % The joint covariance
EST5 = zeros(4,steps);

% Set up figure
figure(5); clf

% Loop through steps
for k=1:steps
    %%%% PF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Propagation
    m2 = A*m2 + chol(Q, 'lower')*randn(4, M);

    % Weigths
    w = w.*mvnpdf(Theta(:, k).', h(m2).', R).';
    w = w/sum(w);

    % Estimate
    m = m2*(w.');

    % Resample
    ess = 1/sum(w.^2);
    if ess < M_T
        ri = resample(w);
        m2 = m2(:, ri);
        w = 1/M*ones(1, M);
    end

    %%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Store the estimate
    EST5(:,k) = m;

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
             EST5(1,1:k),EST5(2,1:k),'b--',...
             [S1(1);S1(1)+dx1],[S1(2);S1(2)+dy1],'k--',...
             [S2(1);S2(1)+dx2],[S2(2);S2(2)+dy2],'k--');
        axis([-2 2 -2.5 1.5]);

        % Pause and draw
        drawnow;
        pause(.1)
    end
end

% Compute error
err5 = rmse(X,EST5)

% Plot EKF
figure(5); clf
plot(X(1,:),X(2,:),'--',...
     EST5(1,:),EST5(2,:),'-',...
     S1(1),S1(2),'kx',S2(1),S2(2),'ko')
legend('True trajectory','CKF estimate','Sensor 1','Sensor 2');
xlabel('x'); ylabel('y'); title('\bf Bootstrap PF Solution')
axis([-2 2 -2.5 1.5]);

fprintf('This is the bootstrap PF solution. Press enter.\n');
pause;

%end %% <--- Uncomment to disable

%% PF w/ CKF importance distribution solution    
fprintf('Running PF w/ CKF solution.\n');

% Parameters
% M = 1000;
M_T = M/3;
n = 4;
Xi = sqrt(n)*eye(4);

P0 = eye(4);
m2 = x0*ones(1, M) + chol(P0, 'lower')*randn(4, M);
w = 1/M*ones(1, M);

mu_k = zeros(n, M);
P_k = zeros(n, n, M);

%P2 = eye(4);        % Some uncertainty
R  = sd^2*eye(2);   % The joint covariance
EST6 = zeros(4,steps);

% Set up figure
figure(6); clf

% Loop through steps
for k=1:steps
    %%%% PF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Propagation    
    % Predict
    mp = A*m2;
    for i = 1:M
        % Generate sigma-points
        Mp = [mp(:, i)*ones(1, n) + chol(Q, 'lower')*Xi, mp(:, i)*ones(1, n) - chol(Q, 'lower')*Xi];

        % Calculate covariances
        Yp = h(Mp);
        yp = 1/(2*n)*sum(Yp, 2);
        Sk = R;
        Ck = zeros(n, 2);
        for j = 1:2*n
            Sk = Sk + 1/(2*n)*(Yp(:, j) - yp)*(Yp(:, j) - yp)';
            Ck = Ck + 1/(2*n)*(Mp(:, j) - mp(:, i))*(Yp(:, j) - yp)';
        end

        % Calculate the proposal parameters
        Kk = Ck/Sk;
        mu_k(:, i) = mp(:, i) + Kk*(Theta(:, k) - yp);
        P_k(:, :, i) = Q - Kk*Sk*Kk';

        % Propagate
        m2(:, i) = mu_k(:, i) + chol(P_k(:, :, i), 'lower')*randn(4, 1);
    end

    % Normalize the weights
    w = (w.'.*mvnpdf(Theta(:, k).', h(m2).', R).*mvnpdf(m2.', mp.', Q)./mvnpdf(m2.', mu_k.', P_k)).';
    w = w/sum(w);

    % Estimate the mean
    m = m2*(w.');

    % Resample
    ess = 1/sum(w.^2);
    if ess < M_T
        ri = resample(w);
        m2 = m2(:, ri);
        w = 1/M*ones(1, M);
    end
    %%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Store the estimate
    EST6(:,k) = m;

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
             EST6(1,1:k),EST6(2,1:k),'b--',...
             [S1(1);S1(1)+dx1],[S1(2);S1(2)+dy1],'k--',...
             [S2(1);S2(1)+dx2],[S2(2);S2(2)+dy2],'k--');
        axis([-2 2 -2.5 1.5]);

        % Pause and draw
        drawnow;
        pause(.1)
    end
end

% Compute error
err5 = rmse(X,EST6)

% Plot EKF
figure(6); clf
plot(X(1,:),X(2,:),'--',...
     EST6(1,:),EST6(2,:),'-',...
     S1(1),S1(2),'kx',S2(1),S2(2),'ko')
legend('True trajectory','CKF estimate','Sensor 1','Sensor 2');
xlabel('x'); ylabel('y'); title('\bf PF w/ CKF Solution')
axis([-2 2 -2.5 1.5]);

fprintf('This is the PF w/ CKF solution. Press enter.\n');
pause;

%end %% <--- Uncomment to disable
