function w_edge = roe_avg(w,x) % Roe-averaging

shiftm1 = circshift((1:length(x))',-1);
shiftp1 = circshift((1:length(x))',1);
w_avg   = 0.5*(w(:,:,1) + w(:, shiftm1, 1));
w_edge  = w_avg(:,shiftp1,1);
end

