# QestOptPOVM

Readme
 % ZHANG Jianchao 2023/4/25 
% Revised by Jun Suzuki 2023/4/28

This is the file to explain the functions and give some examples.



## Introduction: 

In quantum estimation theory, the classical Fisher information matrix (CFIM) is of vital importance in the problem of finding the best measurement which gives the most information about parameter encoded in a family of quantum states. The main objective of this code, QestOpt, is to obtain an optimal measurement and the minimum value of the trace of the inverse of the CFIM for a given quantum state numerically. This problem can be formulated as a convex optimization problem, and we implement a code based the standard steepest decent method to find the best gradient direction. Hence, optimization is effcient and numerically accurate. 



## Problem setting: 

Suppose we have a family of quantum states which is parametrized by n parameters as  with . We aim at finding the true parameter value  by performing a suitable measurement. Here the measurement is mathematically described by a positive operator-valued measure (POVM), which is denoted by  (outcomes). By performing  on , we have a probability distribution . This defines the CFIM  whose ij component is defined by 
$$
J_{\theta,ij}(\Pi)=
\sum_{k=1}^K \frac{Tr(\partial_i\rho_\theta \Pi_k)Tr(\partial_j\rho_\theta \Pi_k)}{Tr(\rho_\theta \Pi_k)}
$$
From this definition, we see that it is sufficient to specify a state $\rho$  and its derivarives on each direction in order to calculate the CFIM, where $\rho_{\theta_0,i} = \left.\frac{\partial \rho_
\theta}{\partial \theta_i}\right|_{\theta=\theta_0}$ . This is called the local settings, which will be explained later in this part. 
We are interested in minizing the inverse of the CFIM, since it gives the minimum variance of an unbiased estimator. This is accomplished by calculating the minimal value of trace of inverse of the CFIM $ {\rm Tr} [ (J_\theta(\Pi))^{-1}]$ . (This is called the A-optimal design. ref1).

To find the optimal POVM among all possible POVMs, we need first solve a restricted class of optimization, where the number of POVM elements is given as . The  POVM as the a set of matrices where each matrix is semi-definite positive and the summation of matrices is identity. The optimization problem is written in following. 

$$\min_{\Pi} {\rm Tr} [ (J_\theta(\Pi))^{-1}]\\
s.t. \forall k,\ \Pi_k\geq0,\ and \sum_{k=1}^K \Pi_k=I$$

$J_\theta$: Fisher information matrix, a function of the state and measurement.
$\Pi$ :POVM (positive operator-valued measurement)
$\rho_\theta$ : quantum state, a function of parameters $\theta$ .

This means, for a given K, we have a space of POVMs. After finding an optimal POVM for each , we compare different s to obtain the best choice for . Therefore, we can find the best POVM in the space of all possible POVMs. 
In the following, we will drop the parameter  when we fix it at some point. In this case, we need to specify a state  and its derivatives $\rho_i=\partial_i\rho\ (i=1,2,...,n)$.

Here is additional comments on the local setting. 

The local settings is a special case of estimation problem. In the real problem, the parameter theta is unknown. The POVM relates to the design of experiment. What we want is to find the relatively best POVM that could extract the most information about theta. However, this is extremely complicated if we have no prior infromation about theta. 

The local settings means for each given true parameter theta, we find the optimal POVM that maximize the Classical Fisher Information. Also, this means the locally optimal POVM depends on theta.

Under local setting, our algorithm has the following three guiding functions

1. In some special case, this locally optimal POVM is global.
2. In design of experiment, this method gives the help in two stage estimation or sequential design estimation.
3. This method suggests how we shrink the sample size.

## Quick start with  a minimum necessary input.


Firstly, let us see a quick start. With a given state rho and derivatives of state drho{j} (j=1,2,...n), we find the optimal POVM. As a simple example (two-parameter qubit with three POVM outcomes), this is executed as follows. 

```matlab
%rho and drho is given 
rho=0.5*eye(2);

drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];

[povm,ob]=Qest([],rho,drho,'k',3);  %Find the optical POVM set with 3 elements.
```

From the output, we can see the algorithm stops at 42 iterations when variation is small. And the corresponding objective function value is 4. 

## Options

There are many options in the input. In order to describe the function of this algorithm clearly, the explanation of each parameter in stated in a table.

### Table of options

| Parameter Name | Explanation                                                  | Input Type     |
| -------------- | ------------------------------------------------------------ | -------------- |
| k              | The number of POVM elements                                  | int            |
| klim           | get all results among a set of k                             | list of int    |
| tensor         | Control tensor product, if use, algorithm will find the result in tensor product | int            |
| povm           | Use specified initial POVM or not                            | int            |
| seed           | random seed, if empty, means random each time                | cell of matrix |
| Conjugate      | Conjuagte gradient, default 0. If set 1, means use conjugate gradient | Boolean        |
| check          | Show the validation check                                    | Boolean        |

To specifically explain the parameters, we use several examples below. They are designed to present the way to use different inputs. This table is easy to see a certain parameter is describled by which example. 

| Parameter Name | Example Serial |
| -------------- | -------------- |
| k              | 1              |
| klim           | 2              |
| tensor         | 3              |
| povm           | 4              |
| seed           | 5              |
| Conjugate      | 6              |
| check          | 7              |

## Example 1: qubit case

State and derivatives;$\rho=\frac{1}{2}
\pmatrix{1 & 0 \cr 0 &1}$ ,$\partial_1\rho=\frac{1}{2}\pmatrix{0 & 1 \cr 1 &0}$ ，$\partial_2\rho=\frac{1}{2}\pmatrix{0 & -i \cr i &0}$ .
This is a two-parameter qubit model and the parameter is set so that the state is the completely mixed-state. 
number of parameter(N)=2,  N equals to the dimension of derivative of rho
number of POVM elements(K)=3.
Find the POVM set gives the minimal of the objective function.
Qest(rho,drho,'k',3)

```matlab
%Set rho and drho 
rho=0.5*eye(2);

drho=cell(1,2); %the size for derivative of rho
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];
d=2;    %dimension of hilbert space, e.g. qubit means d=2;
n=2;    % number of parameters 
setting=[d,n];
%we recommend to use setting, but if you are not sure what is
%setting, use setting=[];
%setting=[];


[povm,ob]=Qest(setting,rho,drho,'k',3);  %Find the optical POVM set with 3 elements.
celldisp(povm);
```

## Example 2: Qutrit case

The next example is a qutrit model, which means the dimension of a hilbert space is 3. The state is normailized identity (the completely mixed state) and the derivative is the same as the previous one. Addtionally, the seed is used and set as 1. It is easy to check that this example will provides the same POVM on each repetition.
Besides that, the input argument 'check' is set as 1. This means the validation check of parameters is showed in consoles. Otherwise it is hided and only be visible when warning or wrong occurs.

```matlab
% qutrit case with d=3, n=2
rho=eye(3)/3;

drho=cell(1,2);
drho{1}=[0   1/2 0;
         1/2 0   0;
         0   0   0];
drho{2}=[0      -1/2*1i  0;
         1/2*1i 0        0;
         0      0        0];
d=3;    %dimension of hilbert space, e.g. qubit means d=2;
n=2;    % number of parameters 
setting=[d,n];
%we recommend to use setting, but if you are not sure what is
%setting, use setting=[];
%setting=[];

%the seed means we set the seed 1, or we have no fixed seed

[povm,ob]=Qest(setting,rho,drho,'k',3,'seed',1,'check',1);  %Find the optical POVM set with 3 elements.
celldisp(povm);
```

Explanation of each checkment.
check dimension of Hilbert space 

- the dimension is the same as setting.
check number of parameters 
- the number is the same as setting.
check n<=d^2-1
- less than the greatest number of parameters.
check rho is positive definite
- rho should be positive definite
check trace of rho is 1
- trace of rho should be 1.
check independency of drho
- drho should be linear independent
check K number of POVM elements is valid
- number of POVM elements has constraints.

## Example 3: Klim

Find the best POVM within a set of possible POVM outcomes ${\cal K}=\{K_1,K_2,\cdots,K\}$ .
This example is provided for the situation that we do not know the appropriate number of POVM elements. It is sufficient to test in a set of numbers and discover the least objective function. If all the numbers in the set produce approximately same result, we can select the POVM with the minimum number of elements we need.
The following example is when we look for the best POVM with 3, 5, and 7 outcomes. 

```matlab
%Set rho and drho 
rho=0.5*eye(2);

drho=cell(1,2);
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];

setting=[];

[povm,ob]=Qest(setting,rho,drho,'klim',[3 5 7]); %Find the optimal POVM set for K=3:10.
celldisp(povm{1}); % here povm{1} is the povm set for K=3. 
```

## Example 4: two qubit model

The i.i.d. extension of a multi-parameter model is of great importance for quantum metrology. This is accomplished by using the option called 'tensor'. 
As an example, the same qubit model (Example 1) is considered. This model is defined by the M tensor product of the single copy model. The state in this example is $\rho^{\otimes M}$ . This M-copies model is considered to simultaneously estimate the parameters in multiple qubit. For simplicity we set M=2 (two copies). 

```matlab
%Set rho and drho 
rho=0.5*eye(2);

drho=cell(1,2);
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];

setting=[2,2];
M=2; % M is the number of tensor product

[povm,ob]=Qest(setting,rho,drho,'k',3,'tensor',M); 
celldisp(povm);
```

## Example 5: Use known povm as initialization.

In QestOpt, the default setting of an initial POVM is randomly chosen. This is the reason that if we do not set the seed, the result of each run will be different. We can set a specific value of the seed (See the next example.) for reproducing the same result. Alternatively, we can start with a given initial POVM. This option is useful to test some special POVM is optimal or not. Also we can start the algorithm with a specific POVM we already got to save the computing time. 
Here is an example with three outcomes povm{k}. Note that this POVM is not normalized (the summation is not equal to the identity matrix.), however the code will automatically normalize it as long as each element is semi-definite posive.  

```matlab
%Set rho and drho 
rho=0.5*eye(2);

drho=cell(1,2);
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];
d=2;    %dimension of hilbert space, e.g. qubit means d=2;
n=2;    % number of parameters 
setting=[d,n];
%setting=[]; it's altnative arguments.
povm=cell(1,3);
povm{1}=eye(2)+0.5*[0   1; % known initialization povm
                1  0];
povm{2}=eye(2)+0.5*[0   -1i;
                1i  0];
povm{3}=eye(2)+0.5*[1  0;
                0  -1];

[povm,ob]=Qest(setting,rho,drho,'povm',povm,'seed',1);  %Find the optical POVM set with 3 elements.
celldisp(povm);
```

## Example 6 Multiple seeds

The main issue upon finding the optimal POVM is that the POVM space is large, and the optimization algorithm will stop at local minima. This is a well-known and common problem in a practical optimization problem. One way to avoid local minima is to run the algorithm from different initial POVMs. We may try different seeds many times to see the best result. 
Here we provide two methods to implement this. The first one is that we want to find the best one over many seeds.

```matlab
%Multiple seeds case 1
rho=0.5*eye(2);

drho=cell(1,2);
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];

setting=[2,2];

seedset=[1 3 5]; % we try to get the best one from this set of seeds

minob=1e+10;
for i=seedset
    [povm,ob]=Qest(setting,rho,drho,'k',3,'seed',i);  
    if ob<minob % find better one
        minob=ob;
        currentpovm=povm;
    end
end
povm=currentpovm;
disp(minob);
celldisp(povm);

```

The second one is the case we want to figure out how this algorithm influenced by different seeds. In order to do this, we set a method to output a set of objective function. If these values have small varience means no much impact by the setting of seeds.

```matlab
%Multiple seeds case 2
rho=0.5*eye(2);

drho=cell(1,2);
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];

setting=[2,2];

seedset=1:20; % we try to get the set of all objective function
oblist=zeros(1,20);
k=3;
povmlist=cell(3,20); %povmlist is a cell evolving all povm, 
% e.g. povmlist(:,1) is a cell containing the povm when seed is 1

for i=seedset
    [povm,ob]=Qest(setting,rho,drho,'k',3,'seed',i);  
    oblist(i)=ob;
    povmlist(:,i)=povm;
end
disp(oblist);
celldisp(povmlist(:,5)); %have a look at 5th povm
disp(var(oblist));
```

We can see in this case variance of oblist is extremely small. This means the setting different seed is totally unneccesary in this case. 

## Example 7 Conjugate gradient

There is a known drawback of gradient decsent is that in some cases, the gradient decsent will be significantly small due to the dimension of space. And it can be fluctuating because of the shape of contour. 
Conjugate gradient (CG) is an extended version of gradient descent. It could handle the above mentioned problem when the graident is tiny. There is no such thing that CG is strictly better than gradient descent but it worth to try when we can not find the answer by gradient descent.
The original CG is presented in scalar, here is the version designed for a matrix.

```matlab
%Conjugate gradient
rho=0.5*eye(2);

drho=cell(1,2);
drho{1}=[0   1/2;
         1/2 0];
drho{2}=[0 -1/2*1i;
         1/2*1i 0];
d=2;    % dimension of hilbert space, e.g. qubit means d=2;
n=2;    % number of parameters 
setting=[d,n];
useCG=1 % 1 means use, 0 means do not use. default does not use

[povm,ob]=Qest(setting,rho,drho,'k',4,'seed',2,'Conjugate',useCG);  
celldisp(povm);

```

## Other manual options and supplemental comments:

Finally, we list other issues to improve the perfomance of the code.

### Stopping rule:

The stopping rule used in this program has two criteria. If one of them is satisfied, the iteration will be stopped. 

1. The significantly small variance. 

   ​     At each step, the code detects the difference between objective functions. If the gap is smaller than a the endurance setted in advance, the iteration step will stop. The default value is set to . If this value needs to be modified (Maybe too small), please check line 154 in Qest.m. 

2. The maximum iterations.
        It is a possibility that we can not find numerically good answer after many iterations. The maximual number of iterations is setted as we do not want to compute the result endlessly. The default amount is 1000. This number is a little bit small for typical problems. You may check line 153 in Qest.m to adjust it.

### Line search:

The scheme of handling line search, which is the mean to find the relatively better step size, is the naive enumerate method. I set a list from 1e-6 to 1e+6 ($\{10^{-6}, 10^{-4}, 10^{-2}, 10^{-1}, 1, 10^{2}, 10^{4},10^{6}\}  $ ). And at each step, all the candidates in this list will be tested. The one which provides the least objective value will be selected. This method is time inefficient. Based on our emperical test, we can remove some number to accelerate the speed of algorithm. Either we could add the number which we guess would be the best step size to increase the accuracy. This can be done by changing 'alpha' (line 3)  in iteration.m. 

### Precision:

The accuracy of this algorithm is high in the examples presented. For the given special derivative of state, the qubit, qutrit and up to four i.i.d. model the precision is at least 8 digits. Since many different cases has not been tested, there is not too much to guarantee for other complicated examples.  

### Time cost:

The cost of time increase linearly with the number of POVM elements (K) but it is exponentially surge with the number of tensor product (M). It is suggested not to test very large number M first. The actual execution time, of course, depends on the spec of a hardware. 



The version of matlab 2021 has been tested, so please use at least this version. If older version is compatible, we will update this.