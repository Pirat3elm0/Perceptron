
% Two Class Uniform Data
ClassUA = rand(250,2);
ClassUB = rand(250,2);

ClassUA = ClassUA + 0.5;
ClassUB = ClassUB + 1.5;

vect_0 = zeros(250,1);
vect_1 = ones(250,1);

ClassUA_res = [ClassUA vect_0];
ClassUB_res = [ClassUB vect_1];
Class_res=[ClassUA_res; ClassUB_res];
data=Class_res(:,1:2).';
classification=Class_res(:,3).';

%Splitting into training and test data sets 80/20
p=0.8;
N=size(Class_res,1);
idx=false(N,1);
idx(1:round(p*N))=true;
idx=idx(randperm(N));

Class_train=Class_res(idx,:);
Class_train=Class_train(randperm(size(Class_train, 1)),:);
Class_test=Class_res(~idx,:);

%Split by Class for graphing visual differences
idxA=Class_test(:,3)==0;
Class_testA=Class_test(idxA,:);
Class_testB=Class_test(~idxA,:);

idxATrain=Class_train(:,3)==0;
Class_trainA=Class_train(idxATrain,:);
Class_trainB=Class_train(~idxATrain,:);


%initializing learning rate, number of iterations
l_rate=0.5;
n_epoch=20;

%Training and predicting and trainging the weights
weights=train_weights(Class_train, l_rate, n_epoch);
disp(weights)

%Create the classification decision line
x=-5:0.1:5;
intercept=-weights(1)/weights(3);
slope=-weights(2)/weights(3);
y=slope*x+intercept;

%Graph the training data and decision line
figure(2);
plot(Class_trainA(:,1),Class_trainA(:,2),'k+');

axis([0.25 2.5 0.25 2.5]);
xlabel('Input One');
ylabel('Input Two');
title('Training set');
hold on;
plot(Class_trainB(:,1),Class_trainB(:,2),'r+');
plot(x,y);

%Graph the test data and decision line
figure(3);
axis([0.25 2.5 0.25 2.5]);
xlabel('Input One');
ylabel('Input Two');
title('Test results');
hold on;
plot(Class_testA(:,1), Class_testA(:,2), 'k+');
plot(Class_testB(:,1), Class_testB(:,2), 'r+');
plot(x,y);

%Function to predict the classification based on the input values and
%weights
function y = predict(row, weights)

activation = weights(1);
for i = 1:(length(row)-1)
    activation = activation + weights(i+1)*row(i);
end
if activation >=0
    y=1;
else
    y=0;
end
end

%Function to adjust the weights if the predicted classification was wrong
function weights=train_weights(train, l_rate, n_epoch)

weights=[1,0,0];

for epoch=1:n_epoch
    sum_error=0;
    for i=1:length(train)
        row=train(i,:);
        prediction = predict(row,weights);
        error=row(end) - prediction;
        sum_error=sum_error+error^2;
        %adjust the bias based on the learning rate and error
        weights(1) = weights(1)+l_rate*error;
        %adjust each weight (for X1 and X2) based on the input value,
        %error, and learning rate
        for j=1:(length(row)-1)
            weights(j+1)=weights(j+1)+l_rate*error*row(j);
        end
    end
    %When sum_error is zero, the weights are trained to that dataset with
    %100% accuracy
    text=['Epoch: ', num2str(epoch), ' l_rate: ', num2str(l_rate), ' error: ', num2str(sum_error)];
    disp(text);
    disp(weights);

end
end
