%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1.1 Setup Dati time series [N animali]

clear all
clc
load('WorkspaceDMD.mat')
global xdata

% Target rank
r=2;

% OPTDMD
[w_opt,e_opt,b_opt,atilde]=optdmd(xdata,t,r,1);

% w - each column is a DMD mode
% e - each entry e(i) is an eigenvalue corresponding to w(:,i)
% b - the best fit coefficient of each DMD mode

T1=0:Dt:60;

% Reconstruct
xk=w_opt*diag(b_opt)*exp(e_opt.*T1);

figure(3)
subplot(1,3,1)
plot(t,Hare,'ob')
title('Snowhare and Lynx - r=2 OPTDMD','Interpreter','latex')
hold on
grid on
xlabel('time from 0 [years]')
ylabel('Hare and Lynx')
plot(t,Lynx,'or')
plot(T1,xk(1,:),'r')
hold on
plot(T1,xk(2,:),'b')
legend('Hare','Lynx','DMD Lynx', 'DMD Hare')

%% Bagging 
clear all
clc
load('WorkspaceDMD.mat')

% Allocate some parameters as number of data to ensemble
imax=30;
N=400;
imode=1;
r=2;
NdataBagging=18;
e=e_opt;
w_ensemble=[];
b_ensemble=[];
lambda_ensemble=[];

% Bagging algorithm and matrix ensembling
for i=1:N
    X(i,:)=randperm(imax,NdataBagging);
    X(i,:)=sort(X(i,:));
    Ttest(i,:)=t(X(i,:));
    [w,e,b] = optdmd(xdata(:,X(i,:)),Ttest(i,:),r,imode,[],e);
    if  abs(real(e(1)))<=0.01 & abs(real(e(2)))<=0.01 
%        plot(real(e),imag(e),'o'); % uncomment to see the resulting eigenvalues
%        grid on
%        hold on
        lambda_ensemble = [lambda_ensemble e];
        w_ensemble=cat(3,w_ensemble,w);
        b_ensemble=[b_ensemble b];
    end
end


% Means of eigenvalues, eigenvectors and weights

mean_lambda1 = mean(lambda_ensemble(1,:));
mean_lambda2 = mean(lambda_ensemble(2,:));

mean_b1=mean(b_ensemble(1,:));
mean_b2=mean(b_ensemble(2,:));

mean_w11=mean(w_ensemble(1,1,:));
mean_w12=mean(w_ensemble(1,2,:));
mean_w21=mean(w_ensemble(2,1,:));
mean_w22=mean(w_ensemble(2,2,:));

B=[mean_b1;mean_b2];
L=[mean_lambda1;mean_lambda2];
W=[mean_w11 mean_w12; mean_w21 mean_w22];

T1=0:0.01:60;

% Mean and variances of the forecasted quantities (by all the quantities ensembled)
xk=[];
for i=1:length(b_ensemble(1,:))
    xk(:,i)=W*diag(b_ensemble(:,i))*exp(lambda_ensemble(:,i).*T1(end));
end

MEAN_FORECAST=[mean(xk(1,:));mean(xk(2,:))];
VAR_FORECAST=[var(xk(1,:));var(xk(2,:))];

% Reconstruct with higher time resolution
xk=W*diag(B)*exp(L.*T1);

subplot(1,3,2)
plot(t,Hare,'ob')
title('DMD Bagging - r=2','Interpreter','latex')
hold on
grid on
xlabel('time from 0 [years]')
ylabel('Hare and Lynx')
plot(t,Lynx,'or')
plot(T1,xk(1,:),'r')
hold on
plot(T1,xk(2,:),'b')
legend('Hare','Lynx', 'DMD Lynx', 'DMD Hare')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Time delay
clear all
clc
load('WorkspaceDMD.mat')

% Allocate variables
imode=1;
r=3;
NdataTD=18;
N=30-NdataTD;
X=[];
Ttest=[];

% Henkel matrix
for i=1:N+1
    X=[X; xdata(:,i:i+NdataTD-1)];
    Ttest(i,:)=t(:,i:i+NdataTD-1);
end

% DMD

[wH,eH,bH,atildeH] = optdmd(X,t(1:NdataTD),r,imode,[]);

% Reconstruct with higher time resolution
Th=0:0.1:60;
hk=wH*diag(bH)*exp(eH.*Th);

subplot(1,3,3)
plot(t,Hare,'ob')
title('DMD Time Embedding - r=3','Interpreter','latex')
hold on
grid on
xlabel('time from 0 [years]')
ylabel('Hare and Lynx')
plot(t,Lynx,'or')
plot(Th,hk(1,:),'r');
plot(Th,hk(2,:),'b');
legend('Hare','Lynx','TD-DMD Lynx', 'TD-DMD Hare')

%% Time delay bagging
clear all
clc
load('WorkspaceDMD.mat')

% Allocate
rank=3;
imode=1;
NdataTD=18;
N=30-NdataTD;
H=[];
Ttest=[];

% Henkel matrix
for i=1:N+1
    H=[H; xdata(:,i:i+NdataTD-1)];
end

% DMD init
[wH,eH,bH,atildeH] = optdmd(H,t(1:NdataTD),rank,imode,[]);

imax=NdataTD;
N=400;
imode=1;
NdataBagging=9;
e=eH;
w_ensemble=[];
b_ensemble=[];
lambda_ensemble=[];

% Bagging

for i=1:N
    X(i,:)=randperm(imax,NdataBagging);
    X(i,:)=sort(X(i,:));
    Ttest(i,:)=t(X(i,:));
    [w,e,b,atildeH] = optdmd(H(:,X(i,:)),Ttest(i,:),rank,imode,[],e);
    if  abs(real(e))<0.01
%plot(real(e),imag(e),'o');
%grid on
%hold on
        lambda_ensemble = [lambda_ensemble e];
        w_ensemble=cat(3,w_ensemble,w);
        b_ensemble=[b_ensemble b];
    end
end

B=[b_ensemble(:,1)];
L=[lambda_ensemble(:,1)];
W=[w_ensemble(:,:,1)];
for j=2:length(b_ensemble(1,:))
    B=b_ensemble(:,j)+B;
    L=lambda_ensemble(:,j)+L;
    W=w_ensemble(:,:,j)+W;
end
B=B./length(b_ensemble(1,:));
L=L./length(lambda_ensemble(1,:));
W=W./length(w_ensemble(1,1,:));

T1=0:0.01:60;

% Mean and variances
clear i

xiter=[];
for j=1:length(b_ensemble(1,:))
    xiter(:,j)=W*diag(b_ensemble(:,j))*exp(imag(lambda_ensemble(:,j))*i.*T1(end));
end

MEAN_FORECAST=[mean(xiter(1,:));mean(xiter(2,:))];
VAR_FORECAST=[var(xiter(1,:));var(xiter(2,:))];

% PLOT  
xk=W*diag(B)*exp(L.*T1);

figure
plot(t,Hare,'ob')
title('DMD Time Embedding with bagging - r=%d',rank-2)
hold on
grid on
xlabel('time from 0 [years]')
ylabel('Hare and Lynx')
plot(t,Lynx,'or')
plot(T1,xk(1,:),'r')
hold on
plot(T1,xk(2,:),'b')
legend('Hare','Lynx', 'DMD Lynx', 'DMD Hare')


%% BEST FIT OPTIMIZATION

clear all
close all
clc
load('WorkspaceDMD.mat')

% Constraints definition
lb = [0 0 0 0 27000 15000];
ub = [2 0.001 0.001 2 37000 25000];

A1 = [];
b1 = [];
Aeq = [];
beq = [];
fval=[];

% Iterate local optimization problem for randomic initial conditions
for i=1:500
    x0=ub.*rand(size(ub));
    [x(:,i),fval(i)] = fmincon(@LVopt,x0,A1,b1,Aeq,beq,lb,ub,[]); %,options

    b=x(1);
    p=x(2);
    r=x(3);
    d=x(4);

    y0=[];
    y0(1)=x(5);
    y0(2)=x(6);
    
end

% get the k minimum values of cost function (as index)
k=3;
ValMin=mink(fval,k);

for i=1:k

index(i)=find(fval==ValMin(i));

% optimized values of the best k minima found

b=x(1,index(i));
p=x(2,index(i));
r=x(3,index(i));
d=x(4,index(i));

y0(1)=x(5,index(i));
y0(2)=x(6,index(i));

% reproduce ode
tspan=0:0.01:60;
[tode,y] = ode45(@(t,y) LVsys(t,y,b,p,r,d), tspan, y0);

% plotting
figure
plot(t,Hare,'ob')
hold on
grid on
xlabel('time from 0 [years]')
ylabel('Hare and Lynx')
plot(t,Lynx,'or')
plot(tode,y(:,1),'r');
plot(tode,y(:,2),'b');
legend('Hare','Lynx', 'Est. Lynx', 'Est. Hare')

end
