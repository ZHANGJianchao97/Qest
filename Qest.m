function [povm,ob] = Qest(setting,rho,drho,varargin)
%QEST 

    %Check the input variables
    p = inputParser;            % function input analyzer
    addParameter(p,'k',0);      % set argument name and default value
    addParameter(p,'klim',NaN);      % 设置变量名和默认参数
    addParameter(p,'tensor',0); 
    addParameter(p,'seed',NaN); 
    addParameter(p,'povm',NaN); 
    addParameter(p,'seedset',[]); 
    addParameter(p,'Conjugate',0);
    addParameter(p,'check',0); % show the check information or not

    parse(p,varargin{:});       % analyze the arguments, if exist, update对输入变量进行解析，如果检测到前面的变量被赋值，则更新变量取值



    %%check the validation of input
    if ~isempty(setting)
        d=setting(1); %dimension of hilbert space, e.g. qubit means d=2;
        n=setting(2); % number of parameters 
        % check d
        checkdisp('check dimension of hilbert space',p.Results.check)
        if d==length(rho)
            checkdisp('correct',p.Results.check)
        else
            error('Input rho has different dimension with d in setting ')
        end
        % check n
        checkdisp('check number of parameters',p.Results.check)
        if n==length(drho)
            checkdisp('correct',p.Results.check)
        else
            error('Input drho has different dimension with n in setting ')
        end
        checkdisp('check n<=d^2-1',p.Results.check)
        if n<d^2
            checkdisp('correct',p.Results.check)
        else
            error('Input drho has more parameters than this space can estimate')
        end       
    else
        setting=[length(rho),length(drho)];
    end
    

    %check rho positive
    checkdisp('check rho is positive definite',p.Results.check)
    if all(eig(rho)>-1e-10)
        checkdisp('correct',p.Results.check)
    else
        error('rho is not positive definite')
    end
    %check tr(rho)=1
    checkdisp('check trace of rho is 1',p.Results.check)
    if trace(rho)==1
        checkdisp('correct',p.Results.check)
    else
        error('trace of rho is not 1')
    end
    %check drho independent

    checkdisp('check independency of drho',p.Results.check)
    A=[];
    for i=1:length(drho)
        A=[A;reshape(drho{i},1,[])]; %vectorize matrix
    end
    if rank(A)==length(drho)
        checkdisp('correct',p.Results.check)
    else
        error('drho is not an linearly independent set')
    end
%check finished


    if ~isnan(p.Results.seed)
        rng(p.Results.seed,'twister');                         % set the seed 
        s = rng;                                  % save generator settings as s
    end
    
    %check the algorithm to use 
    algorithm=1; % 1 means use gradient decent; default setting
    
    if p.Results.Conjugate==1
        algorithm=2; % 2 means use conjuagte gradient decent
    end

    if p.Results.k~=0 % means there is k in the input
        K=p.Results.k;
        check_K(K,setting,p.Results.check); %check if K is valid

        if p.Results.tensor~=0 %the tensor case
            M=p.Results.tensor; %M is the number of tensor product
            [povm,ob]=Qest1(rho,drho,algorithm,'k',K,'tensor',M);
        else
            [povm,ob]=Qest1(rho,drho,algorithm,'k',K);
        end
    elseif ~isnan(p.Results.klim) % means there is klim in the input
        Klim=p.Results.klim;
        Klen=length(Klim); %length of klim
        check_K(min(Klim),setting,p.Results.check);
        povm=cell(1,Klen); %the space to store povms
        oblist=zeros(1,Klen);
        for i=1:Klen
            K=Klim(i);
            [povmtemp,obtemp]=Qest1(rho,drho,algorithm,'k',K);
            povm{i}=povmtemp;
            oblist(i)=obtemp;
            plot(Klim,oblist);
        end
        ob=oblist;
    elseif iscell(p.Results.povm) % means there is given povm as initialization
        ini_povm=p.Results.povm;
        k=length(ini_povm);
        [povm,ob]=Qest1(rho,drho,algorithm,'k',3,'povm',ini_povm);
    end
end

function [povm,ob]=Qest1(rho,drho,algorithm,varargin)

    
    N=length(drho); % number of parameters equals to the number of derivatives.
    d=length(rho); % dimension of hilbert space  

    p = inputParser;            % function input analyzer
    addParameter(p,'k',0); 
    addParameter(p,'tensor',0); 
    addParameter(p,'povm',NaN); 
    parse(p,varargin{:});       % if exist, update value, if not, use default.
    
    if p.Results.k~=0
        K=p.Results.k;
    end

    if ~iscell(p.Results.povm) % povm is not given 
        if p.Results.tensor==0 % no tensor case
            meop=Qestinitial(K,d); % get the random initial measurement
        else
            M=p.Results.tensor;
            [meop,rho,drho]=Qestinitial_tensor(rho,drho,K,d,M); % get the random initial measurement
        end
    else % povm is given
        meop=Povm_meop(p.Results.povm);
    end
    % In this program, meop means measurement operator(Karus operator),
    % povm means positive operator-valued measurement

    [ob,FIinv,dis,prob]=meop_ob(rho,drho,meop); % calculate the current objective function,
    currentob=ob;                             % also find the embedded
                                              % inverse of FI, d and prob.

    itermax=1000;
    tol=1e-8; % Stopping rule 1, the tolerance
    if algorithm==1
        for t=1:itermax
            [meop,ob,FIinv,dis,prob]=iteration(rho,drho,meop,FIinv,dis,prob);
            %fprintf('Trace of inverse of FI : %10.7f \n',ob);
            if abs(currentob-ob)<tol
                fprintf('Stop at iteration :%3.d when delta<%g \n',t,tol);
                break
            end
            currentob=ob;
        end
    elseif algorithm==2
        oldH=cell(1,K);
        oldP=cell(1,K);
        for k=1:K
            oldH{k}=zeros(d,d);
            oldP{k}=zeros(d,d);
        end
        for t=1:itermax
            [meop,ob,FIinv,dis,prob,newH,newP]=iteration(rho,drho,meop,FIinv,dis,prob,oldH,oldP);
            %fprintf('Trace of inverse of FI : %10.7f \n',ob);
            if abs(currentob-ob)<tol
                fprintf('Stop at iteration :%3.d when delta<%g \n',t,tol);
                break
            end
            currentob=ob;
            oldH=newH;
            oldP=newP;
        end
    end


    fprintf('Objective function: Trace of inverse of FI : %10.7f \n',ob)
    povm=cell(1,K);
    for k=1:K
        povm{k}=meop{k}'*meop{k};
    end  
end


function check_K(K,setting,check)
    checkdisp('check K number of POVM elements is valid',check)
    d=setting(1);n=setting(2); 
    
    if K>n
        checkdisp('correct',check)
    else
        error('number of POVM elements is too small')
    end  
    if K>d^2+n*(n+1)/2-1 %d=2,n=2 this=6 K>6
        warning('number of POVM elements is too large.')
    end
end

function meop=Povm_meop(povm) %from povm to get karus operator
    dim=length(povm{1});
    numPo=length(povm);
    G=zeros(dim,dim);
    for i=1:numPo
        G=G+povm{i};
    end
    if det(G)<1e-10
        warning('The determinant of summation of povm less than e-10');
    end
    Gis=sqrtm(inv(G));
    meop=cell(1,numPo);
    for k=1:numPo
        povm{k}=Gis*povm{k}*Gis;
        meop{k}=chol(realdiag(povm{k}));  %get A_k
    end   
end


%   truncate the imaginary part of diagonal entries.
function outpovm=realdiag(povm)
    outpovm=povm-diag(imag(diag(povm))*1j);
end

function checkdisp(string,check)
    if check==1
        disp(string);
    end
end
