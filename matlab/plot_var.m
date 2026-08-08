function plot_var(x, W,t,dt_save,dt, variable,n)
% Plot height velocity and alpha for any n
  if mod(t,dt_save)<dt
  fprintf("Saving data at time %6.4e\n", t);
        if (variable == "h")
            plot(x,W(1,:),'b','LineWidth',2)
            title(sprintf('Water height at t = %0.2f',t))
            xlabel('x');
            ylabel('h');
        elseif (variable == "u")
            plot(x,W(2,:)./W(1,:),'r','LineWidth',2)
            title(sprintf('Water velocity at t = %0.2f',t))
            xlabel('x');
            ylabel('u');
        elseif (variable == "alpha")
            for ii = 1:n
                 figure(ii)
                 plot(x,W(ii+2,:)./W(1,:),'k','LineWidth',2)
                 xlabel('x');
                 ylabel(sprintf("$\\alpha_{%3d}$",ii),'Interpreter','latex')
            end

        else
            return
        end
   
  end
end