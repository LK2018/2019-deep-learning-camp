clear;
im = double(imread('DATA1.tif'));
im_size = size(im);
for n=1:im_size(3)
    im(:,:,n) = mapminmax(im(:,:,n),0,225);
end

load X11.mat;
Y=X11;
X_size = [50,50,138]
n = 10;
for i=1:X_size(1)
    for j=1:X_size(2)
            Label(1,n)=Y(j,i);
            n=n+1;
    end
end
Label=mapminmax(Label,0,255);
%im2=im2double(im1); %������ת��Ϊdouble����
im3=mat2gray(im1); %��im��һ����[0 1]�����ڣ���im�е����ֵ����Сֵ��ֵΪ1��0
im4=im2uint8(im3); %��im���䵽[0 255]
figure(),imshow(im4(:,:,100:102)); %��ʾ����1������3��Χ�ڵ�ͼ��