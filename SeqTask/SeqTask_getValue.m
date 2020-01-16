function val=SeqTask_getValue(range)
% Get either a single or random value
if (length(range)==1)
    val = range;
elseif (range(2)-range(1)<=1) 
    val = unifrnd(range(1),range(2));
else 
    val = unidrnd(range(2)-range(1)+1)+range(1)-1;
end