clc
close all
clear all

addpath('/home/user/code/caffe/matlab'); % Directory Where /caffe/matlab is located 

caffe.set_mode_cpu(); % CPU or GPU 

net_model = 'LR_deploy_AR.prototxt';
net_weights = ['R_angular.caffemodel'; 'G_angular.caffemodel'; 'B_angular.caffemodel'];
phase = 'test'; 

folder = '/home/user/code/caffe/examples/LR_prof/Train';
filepaths = dir(fullfile('*.png'));

    
im_label = double( LFXReadStanfordIllum( fullfile(filepaths(1).name)))./ 65536;

for dim = 1 : 3
    GT = squeeze( im_label(:,:,:,:,dim) ); % Low Angular and Low Spatial light Field Input

    im_input = GT(1:2:end,1:2:end,1:2:end,1:2:end); 

    net = caffe.Net(net_model, net_weights(dim,:), phase);


        for x = 1 : size(im_input,3)
            for y = 1 : size(im_input,4)

                fprintf('x = %i and y = %i\n', x,y)
                subim_input= im_input(:,:,x,y);      
                scores = net.forward({subim_input});
                ans =double(cell2mat(scores(1)));
                output(:,:,x,y,dim)= ans;
            end
        end
end

LF_LowSR_HighAR = output; % High Angular Resolution and Low Spatial Resolution output of AR network

clear im_input
clear subim_input
clear output

net_model = 'LR_deploy_SR.prototxt';
net_weights = ['R_spatial.caffemodel'; 'G_spatial.caffemodel'; 'B_spatial.caffemodel'];

for dim = 1 : 3
    im_input = squeeze( LF_LowSR_HighAR(:,:,:,:,dim) );

    bicubic = interp2(squeeze(im_input(7,7,:,:)),'cubic');
    output = interp2(squeeze(im_input(7,7,:,:)),'cubic');

    label =squeeze(im_label(7,7,:,:,dim));


    net = caffe.Net(net_model, net_weights(dim,:), phase);

    for x = 1 : size(im_input,3)-1
        for y = 1 : size(im_input,4)-1

            fprintf('x = %i and y = %i\n', x,y);
            subim_input(:,:,1)= im_input(:,:,x,y);
            subim_input(:,:,2)= im_input(:,:,x,y+1);
            subim_input(:,:,3)= im_input(:,:,x+1,y);
            subim_input(:,:,4)= im_input(:,:,x+1,y+1);
            scores = net.forward({subim_input});
            ans =double(cell2mat(scores(1)));
            output((2*x)-1,2*y)= ans(1);
            output(2*x,(2*y)-1)= ans(2);
            output(2*x,2*y)= ans(3);

        end
    end
    output(output<0) = 0; output(output>1) = 1;
    Proposed(:,:,dim) = output; 
    Bicubic(:,:,dim) = bicubic;
    Label(:,:,dim) = label;

end


psnr_Proposed = psnr(Proposed,Label(1:end-1,:,:))
psnr_Bicubic  = psnr(Bicubic,Label(1:end-1,:,:))
ssim_Proposed = ssim(Proposed,Label(1:end-1,:,:))
ssim_Bicubic  = ssim(Bicubic,Label(1:end-1,:,:))
