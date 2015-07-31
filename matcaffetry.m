model = './models/bvlc_reference_caffenet/deploy.prototxt';
weights = './models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';
caffe.set_mode_cpu();
net = caffe.Net(model, 'test'); % create net but not load weights
net.copy_from(weights); % load weights