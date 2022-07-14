%% KS-NN TRAINING FULL STATES

clear all
close all
clc

load('kuramoto_sivashinsky.mat')

%Define subset
Nsubset=9;
Num=1+floor(Nsubset/2);

usave1=[usave usave(:,1:Nsubset-1)]; %Lets apply periodic boundary conditions

input=[];
output=[];

%Take each subset of 9 values
for i=1:length(xsave')
    input=[input; usave1(1:70,i:Nsubset-1+i)];
    output=[output; usave1(2:71,i+Num)];
end

%rescaling for normalization
max_v = max(max(usave));
min_v = min(min(usave));

scale_down = @(x) (x - min_v) / (max_v - min_v);
scale_up = @(x) x * (max_v - min_v) +  min_v;

input = scale_down(input);
output = scale_down(output);

%Train NNs

trainFcn = 'trainlm';
net = fitnet([12 9 3], trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';  % Mean Squared Error
net = train(net,input.',output.');

%Reconstruct the output t+dt
uReconstructed=net(input');

%Rescale
uReconstructed=scale_up(uReconstructed);
uReconstructed=reshape(uReconstructed, [70 1024]);

%plot
figure(1)
subplot(1,3,1)
contour(xsave,tsave,usave,[-10 -5 0 5 10]),shading interp, colormap(gray)
xlabel('x axis')
ylabel('time')
title('KS-Data')

subplot(1,3,3)
contour(xsave,tsave(2:end),uReconstructed,[-10 -5 0 5 10]),shading interp, colormap(gray)
xlabel('x axis')
ylabel('time')
title('KS-NN-Subset')

figure(2)
plot(xsave,uReconstructed(30,:))
hold on
grid on
plot(xsave,usave(31,:))
title('data vs NN results @ t=30')
xlabel('x axis')
ylabel('Values')

%% 2.1 NN-Features with derivatives

clear all
clc

load('kuramoto_sivashinsky.mat')

%Lets apply periodic boundary conditions to calculate the derivatives
usave1=[usave(:,1021:1024) usave usave(:,1:4)]; 
N=4;
xsave1=[-3*dx:dx:0 xsave' xsave(end)+dx:dx:xsave(end)+4*dx];

%Derivatives calculation
for i=1:length(tsave)-1
    u_x(i,:)=diff(usave1(i,:))./diff(xsave1);
    u_xx(i,:)=diff(u_x(i,:))./diff(xsave1(2:end));
    u_xxx(i,:)=diff(u_xx(i,:))./diff(xsave1(3:end));
    u_xxxx(i,:)=diff(u_xxx(i,:))./diff(xsave1(4:end));
end

%Define input and output
input=[];
output=[];
for i=1:length(xsave')
    input=cat(1,input,[usave1(1:70,i+4) u_x(1:70,i+4) u_xx(1:70,i+4) u_xxx(1:70,i+4) u_xxxx(1:70,i+4)]); %.*usave1(1:70,i+4)
    output=cat(1,output,usave1(2:71,i));
end

%Rescaling
max_v = max(max(usave));
min_v = min(min(usave));

scale_down = @(x) (x - min_v) / (max_v - min_v);
scale_up = @(x) x * (max_v - min_v) +  min_v;

input = scale_down(input);
output = scale_down(output);

%Train NN
trainFcn = 'trainlm';
net = fitnet([12 9 3], trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';  % Mean Squared Error
net = train(net,input.',output.');

%Reconstruct Output and reshape
OutEst = (net(input'))';
uReconstructed=reshape(OutEst,[length(tsave)-1 length(xsave')]);

%Rescale up
uReconstructed=scale_up(uReconstructed);

%Plot
figure
subplot(1,2,1)
contour(xsave,tsave,usave,[-10 -5 0 5 10]),shading interp, colormap(gray)
xlabel('x axis')
ylabel('time')
title('KS-Data')

subplot(1,2,2)
contour(xsave,tsave(2:end),uReconstructed,[-10 -5 0 5 10]),shading interp, colormap(gray)
xlabel('x axis')
ylabel('time')
title('KS-NN-derivatives')

%Recursive calculation to se how NN trained behaves for t+2dt
% Start=input(1:length(tsave)-1:end,:);
% Out=net(Start');
% Out=scale_up(Out);
% for i=1:length(tsave)-1
%     %SAVE
%     OutRecursive(i,:)=Out;
%     %Lets apply periodic boundary conditions
%     Out=[Out(1021:1024) Out Out(1:4)]; 
%     %Find derivatives
%     u_x=diff(Out)./diff(xsave1);
%     u_xx=diff(u_x)./diff(xsave1(2:end));
%     u_xxx=diff(u_xx)./diff(xsave1(3:end));
%     u_xxxx=diff(u_xxx)./diff(xsave1(4:end));
%     inputRecursive=[];
%     for j=1:length(xsave')
%         inputRecursive=[inputRecursive; Out(j+4) u_x(j+4) u_xx(j+4) u_xxx(j+4) u_xxxx(j+4)];
%     end
%     Out=net(inputRecursive');
%     Out=scale_up(Out);
% end
% 
% subplot(1,3,3)
% contour(xsave,tsave(2:end),OutRecursive,[-10 -5 0 5 10]),shading interp, colormap(gray)
% xlabel('x axis')
% ylabel('time')
% title('KS-NN-Recursive')

figure
plot(xsave,uReconstructed(30,:))
hold on
grid on
plot(xsave,usave(31,:))
plot(xsave,OutRecursive(30,:))

%% 2.3 Reaction Diffusion Equation SVD-NN
clear all
close all
clc

load('reaction_diffusion_big')

%Reshape data
Dim=size(u,1);
flat_u=[];
for i = 1:length(t)
    ur = reshape(u(:,:,i), [1, Dim^2]);
    flat_u = [flat_u; ur];
end
flat_u = flat_u';

%apply SVD on reshaped data
[uSVD,sSVD,vSVD]=svd(flat_u,'econ');

r=4; 
Ur=uSVD(:,1:r);
Sr=sSVD(1:r,1:r);
Vr=vSVD(:,1:r);

UReconstructed=Ur*Sr*Vr';
DataR=zeros(size(u));

%Reconstruct data to see if SVD reduction worked
for i=1:length(t)
    Temp=UReconstructed(:,i);
    Temp=reshape(Temp,[Dim Dim]);
    DataR(:,:,i)=Temp;
    Temp=[];
end

nframe = 100;

figure(5)
subplot(1,2,1)
pcolor(x,y,u(:,:,nframe)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('R-D Data @ t=100')
subplot(1,2,2)
pcolor(x,y,DataR(:,:,nframe)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('R-D SVD reconstruction @ t=100')

%Create reduced states
for i = 1:length(t)
    state = flat_u(:,i);
    reduced_state = Ur\state;
    reduced_states(:,i) = reduced_state;
end

%Train NN
input=reduced_states(:,1:end-1);
output=reduced_states(:,2:end);
hiddenLayerSize=[7,4];
trainFcn = 'trainlm';
net = fitnet(hiddenLayerSize,trainFcn);
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';
net.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression', 'plotfit'};
[net,tr] = train(net,input,output);

Out=net(input);
Rec=Ur*Out;
Rec=reshape(Rec,[Dim Dim length(t)-1]);

figure(6)
subplot(1,2,1)
pcolor(x,y,u(:,:,nframe)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('R-D Data @ t=100')
subplot(1,2,2)
pcolor(x,y,Rec(:,:,nframe)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('R-D NN @ t=100')

% Ten istances recursive NN test
Out=net(input);
for i=1:10
    Out=net(Out);
end
Rec=Ur*Out;
Rec=reshape(Rec,[Dim Dim length(t)-1]);

figure(7)
subplot(1,2,1)
pcolor(x,y,u(:,:,nframe)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('R-D Data @ t=100')
subplot(1,2,2)
pcolor(x,y,Rec(:,:,nframe+11)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('R-D NN recursive @ t=100')



%% NN on SVD eigenvalues

clear all
close all
clc

load('reaction_diffusion_big')

[uSVD,sSVD,vSVD]=pagesvd(u,'econ');

r=8;
uR=[];
input=[];
output=[];
Dim=size(u);

for i=1:length(t')
    uR=cat(3,uR,uSVD(:,1:r,i)*sSVD(1:r,1:r,i)*vSVD(:,1:r,i)');
end

for k=1:length(t')-1
   input=[input; reshape(diag(sSVD(1:r,1:r,k)),[1 r])];
   output=[output; reshape(diag(sSVD(1:r,1:r,k+1)),[1 r])];
end

net = feedforwardnet([10 10 10]);
net = train(net,input.',output.');

figure
subplot(1,3,1)
pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('Reaction-Diffusion data at tf')

subplot(1,3,2)
pcolor(x,y,uR(:,:,end)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('y axis')
title('SVD reduction at tf')

sReconstructed=diag(net(input(end,:)').');

Forecast=uSVD(:,1:r,end)*sReconstructed*vSVD(:,1:r,end)';

subplot(1,3,3)
pcolor(x,y,Forecast(:,:)); shading interp; colormap(hot)
xlabel('x axis')
ylabel('time')
title('NN-Prediction')

save('NN-RD_final')