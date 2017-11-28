% Plot polling behavior of pastream for writing/reading files
% Depends on cbrewer: https://www.mathworks.com/matlabcentral/fileexchange/34087-cbrewer---colorbrewer-schemes-for-matlab
% 
% buffersize test command:
% for i in {11..16}; do echo -n "$(( 1 << i )) " && pastream tone.wav out.wav -D3 --buffersize=$(( 1 << i )) -q > timing$i.csv; done
%
% blocksize test command:
% for i in {1..16..2}; do echo -n "$(( i * 512 )) " && pastream tone.wav out.wav -D3 --blocksize=$(( i * 512 )) -q > timing$i.csv; done
%
% where 'tone.wav' is just a 10 second sinewave generated with sox
% and '3' is the device index
fig1 = figure();
hold on;
title('player')

fig2 = figure();
hold on;
title('recorder')

colors = cbrewer('div', 'PiYG', 10);
%for i=1:6 % buffersize
for i=1:8
    j = i*2 - 1;
%    j = i + 10; % buffersize
    
    m = dlmread(sprintf('timing%d.csv', j), ' ');
    
    figure(fig1)
    pidx = m(:, 1) == 0;
    t = m(pidx, 2);
    v = m(pidx, 3);

    plot(t, v / 512, 'o', 'color', colors(i, :))    
    stairs(t, v / 512, 'color', colors(i, :), ...
        'DisplayName', sprintf('Blocksize=%d', j * 512))
%    stairs(t, v / 512, 'color', colors(i, :), ...
%        'DisplayName', sprintf('Blocksize=%d', 2^j)) % buffersize
    
    figure(fig2)
    ridx = m(:, 1) == 1;
    t = m(ridx, 2);
    v = m(ridx, 3);
    plot(t, v / 512, 'o', 'color', colors(i, :))
    stairs(t, v / 512, 'color', colors(i, :), 'DisplayName', sprintf('Blocksize=%d', j * 512))
%    stairs(t, v / 512, 'color', colors(i, :), ...
%        'DisplayName', sprintf('Blocksize=%d', 2^j)) % buffersize
    
end

for fig={fig1, fig2}
    grid on;
    figure(fig{1});
    xlabel('time (s)'); ylabel('frames\_available / 512')
    legend('show')
end