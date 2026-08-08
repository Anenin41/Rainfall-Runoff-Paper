function w_edge = roe_avg_non_periodic_BC(w) % Roe-averaging

wL = w(:,[1 1:end]); wR = w(:,[1:end end]); % extend at boundaries
for i = 1:length(wL)-1
    w_edge(:,i)= (wL(:,i) + wR(:,i))./2;
end
end

