function h=plot_parameters(Ep,Cp,gP,xlabels)
    % Plots the estimated parameters of a model, optionally in a grouped 
    % bar plot with the parameters used to generate the data.
    %
    % Ep      - vector of expected values of estimated parameters
    % Cp      - covariance matrix of estimated parameters
    % gP      - (optional) parameters used to generate the data
    % xlabels - (optional) labels for the x-axis
    %
    % E.g.
    % figure;
    % plot_parameters(P,Cp,gP);
    %
    % Zeidman, Friston, Parr
    % _____________________________________________________________________

    h = []; % handles to return

    % Validate inputs
    P = Ep(:);    
    if isvector(Cp), Cp = diag(Cp); end
    n  = length(P);
    
    % Default legend text and labels
    if nargin < 4
        xlabels = 1:n;
    end
    
    % Concatenate generative parameters if provided
    has_gP = nargin > 2 && ~isempty(gP);
    if has_gP
        gP = gP(:);
    end
    
    % Compute 90% confidence interval
    ci = spm_invNcdf(1 - 0.05);               % confidence interval
    C  = diag(Cp);    
    c  = ci*sqrt(C);
    c  = c(:)';           
    
    % Plot generative parameters if provided        
    w       = 0.1;    
    xoffset = w;
    gap     = 0.2;
    totalw  = w + gap; % Total width per bar
    x = xoffset : totalw : xoffset+((n-1)*totalw);
    if has_gP
        h{1} = plot_bars(x,gP,w,[0.5 0.5 0.5]);
        hold on;
    end
    
    % Plot estimated parameters
    if has_gP
        xp = x + w;
    else
        xp = x;
    end
    h{2} = plot_bars(xp,P,w,[51 153 255]./255);    
    
    % Restore zero line
    xlims = [0 xp(end)+w+xoffset];
    line([xlims(1) xlims(2)],[0 0],'Color','k');
    
    % Add error bars    
    col   = [1 3/4 3/4];
    errx  = xp+(w/2);
    for k = 1:n
        h{2+k} = line([errx(k) errx(k)],[-1 1]*c(k) + P(k),...
            'LineWidth',4,'Color',col);
    end
    
    % Xlabel
    set(gca,'XTick',xp,'XTickLabel',xlabels,'XLim',xlims);
    
    % Ylimits
    alldata = [gP; Ep-c(:); Ep+c(:)];
    grandmin = min(alldata);
    grandmax = max(alldata);
    ylim([grandmin-0.05*abs(grandmin) grandmax+0.05*abs(grandmax)]);
    
    hold off;
        
    function H = plot_bars(x,y,w,c)
        H = [];
        for i = 1:length(x)
            if y(i) == 0
                % Do nothing
            elseif y(i) > 0
                % Positive bar
                H(i) = rectangle('Position',[x(i),0,w,y(i)],'FaceColor',c);
            else
                % Negative bar
                H(i) = rectangle('Position',[x(i),y(i),w,abs(y(i))],'FaceColor',c);
            end
            hold on
        end
    end
end