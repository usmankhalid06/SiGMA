function [D,X,Err]= my_ACSD(Y,Di,spa,nIter)
    D = Di;
    X = zeros(size(D,2),size(Y,2)); 
    fprintf('Iteration:     ');
    for iter=1:nIter
        fprintf('\b\b\b\b\b%5i',iter);
        Dold = D;
        for j =1:size(D,2)
            X(j,:) = 0;
            E = Y-D*X;
            xk = D(:,j)'*E; 
            thr = spa./abs(xk);
            X(j,:) = sign(xk).*max(0, bsxfun(@minus,abs(xk),thr/2));
            rInd = find(X(j,:));
            if (length(rInd)<1)
%                 D(:,j)= randn(size(D(:,j),1), 1);
                [~,ind]= max(sum(Y-D*X.^2)); 
                D(:,j)= Y(:,ind)/norm(Y(:,ind));
            else
                D(:,j) = E(:,rInd)*X(j,rInd)'./norm(E(:,rInd)*X(j,rInd)');
            end                 
                   
        end
        Err(iter) = sqrt(trace((D-Dold)'*(D-Dold)))/sqrt(trace(Dold'*Dold));
        % D = I_CD(D,X,Y);  
    end
end

function D = I_CD(D,X,Y)
    T2 = 0.99;
    T1 = 3;
    K=size(D,2);
    Er=sum((Y-D*X).^2,1); 
    G=D'*D; G = G-diag(diag(G));
    for jj=1:1:K
        if max(G(jj,:))>T2 || length(find(abs(X(jj,:))>1e-7))<=T1 
            [~,pos]=max(Er);
            Er(pos(1))=0;
            D(:,jj)=Y(:,pos(1))/norm(Y(:,pos(1)));
            G=D'*D; 
            G = G-diag(diag(G));
        end
    end
end
