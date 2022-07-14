clear all, close all

% Simulate Lorenz system rho=10
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=10;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];
for j=1:80  % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    rho=((y(2:end,2)-y(1:end-1,2))/dt+y(1:end-1,1).*y(1:end-1,3)+y(1:end-1,2))./y(1:end-1,1);
    input=[input; y(1:end-1,:) rho.*y(1:end-1,1)];
    output=[output; y(2:end,:)];
end

% Simulate Lorenz system rho=28
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=28;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

for j=1:80  % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    rho=((y(2:end,2)-y(1:end-1,2))/dt+y(1:end-1,1).*y(1:end-1,3)+y(1:end-1,2))./y(1:end-1,1);
    input=[input; y(1:end-1,:) rho.*y(1:end-1,1)];
    output=[output; y(2:end,:)];
end

% Simulate Lorenz system rho=35
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=35;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

for j=1:80  % training trajectories
    x0=30*(rand(3,1)-0.5);
    [t,y] = ode45(Lorenz,t,x0);
    rho=((y(2:end,2)-y(1:end-1,2))/dt+y(1:end-1,1).*y(1:end-1,3)+y(1:end-1,2))./y(1:end-1,1);
    input=[input; y(1:end-1,:) rho.*y(1:end-1,1)];
    output=[output; y(2:end,:)];
end

% Training the network
net = feedforwardnet([10 10 10 10]);
% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'radbas';
% net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');


% Simulate Lorenz system rho=17
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=17;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);

Rho(1)=((y(2,2)-y(1,2))/dt+y(1,1).*y(1,3)+y(1,2))./y(1,1);

for jj=2:length(t)
    y0=net([x0(1:3); Rho(jj-1)*x0(1)]);
    ynn(jj,1:3)=y0(1:3).'; 
    Rho(jj)=((y(jj,2)-y(jj-1,2))/dt+y(jj-1,1).*y(jj-1,3)+y(jj-1,2))./y(jj-1,1);
    x0=y0;
end

figure(1)
subplot(4,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('X coordinate','Interpreter','latex')
title('X @ $\rho=17$','Interpreter','latex')
legend('Data-X','NN-X')
subplot(4,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('Y coordinate','Interpreter','latex')
title('Y @ $\rho=17$','Interpreter','latex')
legend('Data-Y','NN-Y')
subplot(4,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('Z coordinate','Interpreter','latex')
title('Z @ $\rho=17$','Interpreter','latex')
legend('Data-Z','NN-Z')
subplot(4,2,7), plot(t,Rho,'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('$\hat{\rho}$','Interpreter','latex')
title('$\rho=17$','Interpreter','latex')
legend('Rho estimated')

% Simulate Lorenz system rho=40
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; r=40;

Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

input=[]; output=[];

x0=30*(rand(3,1)-0.5);
[t,y] = ode45(Lorenz,t,x0);

Rho(1)=((y(2,2)-y(1,2))/dt+y(1,1).*y(1,3)+y(1,2))./y(1,1);

for jj=2:length(t)
    y0=net([x0(1:3); Rho(jj-1)*x0(1)]);
    ynn(jj,1:3)=y0(1:3).'; 
    Rho(jj)=((y(jj,2)-y(jj-1,2))/dt+y(jj-1,1).*y(jj-1,3)+y(jj-1,2))./y(jj-1,1);
    x0=y0;
end

figure(1)
subplot(4,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('X coordinate','Interpreter','latex')
title('X @ $\rho=40$','Interpreter','latex')
legend('Data-X','NN-X')
subplot(4,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('Y coordinate','Interpreter','latex')
title('Y @ $\rho=40$','Interpreter','latex')
legend('Data-Y','NN-Y')
subplot(4,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('Z coordinate','Interpreter','latex')
title('Z @ $\rho=40$','Interpreter','latex')
legend('Data-Z','NN-Z')
subplot(4,2,8), plot(t,Rho,'Linewidth',[2])
xlabel('time[s]','Interpreter','latex')
ylabel('$\hat{\rho}$','Interpreter','latex')
title('$\rho=40$','Interpreter','latex')
legend('Rho estimated')