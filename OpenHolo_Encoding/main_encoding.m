%% load sample complex field data
load ('./sampleHologram/teapot_sample_hologram')
figure(); imagesc(abs(hologram)); colorbar(); axis equal; title('amplitude of complex field')
figure(); imagesc(angle(hologram)); colorbar(); axis equal; title('phase of complex field')

%% example 1 - single side band encoding
encoded = fn_encoding(hologram, 'ssb', 'passband', 'left');
figure(); imagesc(encoded); colorbar(); axis equal; title('single side band with left passband')

%% example 2 - numerical interference
encoded = fn_encoding(hologram, 'numerical interference');
figure(); imagesc(encoded); colorbar(); axis equal; title('numerical interference')

%% example 3 - phase to amplitude
encoded = fn_encoding(hologram, 'phase to amplitude');
figure(); imagesc(encoded); colorbar(); axis equal; title('phase to amplitude')

%% example 4 - real
encoded = fn_encoding(hologram, 'real');
figure(); imagesc(encoded); colorbar(); axis equal; title('real')

%% exmaple 5 - off-axis + single side band encoding
hologram2 = fn_freqShift(hologram,[0,100]);
encoded = fn_encoding(hologram2, 'ssb', 'passband', 'right');
figure(); imagesc(encoded); colorbar(); axis equal; title('offaxis + single side band')

%% exmplae 6 - Burckhardt encoding
encoded = fn_encoding(hologram, 'burckhardt');
figure(); imagesc(encoded); colorbar(); axis equal; title('burckhardt')

%% exmplae 7 - Two phase encoding
encoded = fn_encoding(hologram, 'two phase');
figure(); imagesc(encoded); colorbar(); axis equal; title('two phase')