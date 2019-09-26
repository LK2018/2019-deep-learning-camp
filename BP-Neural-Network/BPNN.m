clear;clc;
X=double(imread('DATA1.tif'));


load X11.mat;
Y=X11;
X_size=size(X);
%准备数据集
for n=1:X_size(3)
    X(:,:,n)=mapminmax(X(:,:,n),-1,1);%归一化
end
Train=[];%138*2500
Label=[];%1*2500
n=1;         
for i=1:X_size(1)
    for j=1:X_size(2)
            Train(:,n)=reshape(X(j,i,:),1,X_size(3));
            Label(1,n)=Y(j,i);
            n=n+1;
    end
end
Label=mapminmax(Label,0,1);%sigmoid函数的取值范围为[0,1]
%随机初始化参数
W1=normrnd(0,1,5,138);
W2=normrnd(0,1,5,5);
W3=normrnd(0,1,1,5);
B1=normrnd(0,1,5,1);
B2=normrnd(0,1,5,1);
B3=normrnd(0,1,1,1);
%定义激活函数
active=inline('1./(1+exp(-x))','x');
active_diff=inline('(1./(1+exp(-x))).*(1-(1./(1+exp(-x))))','x');
a=1;
batch_size = 32;
for n=1:100
    %向前传播
    B1=repmat(B1,1,batch_size);
    B2=repmat(B2,1,batch_size);
    B3=repmat(B3,1,batch_size);
    idx=randperm(X_size(1)*X_size(2));%随机排列
    batch=Train(:,idx(1:batch_size));
    Z1=W1*batch+B1; 
    A1=active(Z1);
    Z2=W2*A1+B2; 
    A2=active(Z2);
    Z3=W3*A2+B3;
    Output=active(Z3);

    Residual=Label(:,idx(1:batch_size))-Output;

    W3_old = W3;
    W2_old = W2;
    %向后传播
    cost=0;
    for j=1:batch_size
        S3=diag(active_diff(Z3(:,j)))*Residual(:,j);
        S2=diag(active_diff(Z2(:,j)))*W3_old'*S3;
        S1=diag(active_diff(Z1(:,j)))*W2_old'*S2;  
        W3=W3+a*S3*A2(:,j)'/batch_size;    
        W2=W2+a*S2*A1(:,j)'/batch_size;    
        W1=W1+a*S1*batch(:,j)'/batch_size;
        B3=B3(:,1)+a*S3/batch_size;    
        B2=B2(:,1)+a*S2/batch_size;    
        B1=B1(:,1)+a*S1/batch_size;
        cost=cost+Residual(:,j)^2;
    end
    fprintf('更新次数:%d，cost=%f\n',n,cost);
    loss(n)=cost;
end
%对整张影像进行预测
Z1=W1*Train+repmat(B1,1,2500); 
A1=active(Z1);
Z2=W2*A1+repmat(B2,1,2500); 
A2=active(Z2);
Z3=W3*A2+repmat(B3,1,2500);
Output=active(Z3);
save('Output','Output');
figure; 
subplot(121); imagesc(reshape(Output,X_size(1),X_size(2)));
subplot(122); imagesc(reshape(Label,X_size(1),X_size(2)));
figure;plot(loss);