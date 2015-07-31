import caffe
from caffe import layers as L
from caffe import params as P

def irisnet(imagefile, batch_size):
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    n.data_pair, n.label_pair = L.ImageData(batch_size=batch_size, source=imagefile,
                             transform_param=dict(scale=1./255), ntop=2)
    n.slice_pair = L.Slice(n.data_pair, n.data, ndata_p)#,slice_param=dict(slice_dim=1, slice_point=1))
    
    n.conv1 = L.Convolution(n.data, kernel_h=11, kernel_w=23, num_output=2, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    #n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    #n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.ip1 = L.InnerProduct(n.pool1, num_output=500, weight_filler=dict(type='xavier'))
    n.sigmoid = L.Sigmoid(n.ip1)#, in_place=True)
    #n.feat = L.InnerProduct(n.Sigmoid)
    #n.ip2 = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.conv1_p = L.Convolution(n.data_p, kernel_h=11, kernel_w=23,num_output=2, weight_filler=dict(type='xavier'))
    n.pool1_p = L.Pooling(n.conv1_p, kernel_size=2, stride=2, pool=P.Pooling.MAX)  
    n.ip1_p = L.InnerProduct(n.pool1_p, num_output=500, weight_filler=dict(type='xavier'))
    n.sigmoid_p = L.Sigmoid(n.ip1_p)#, in_place=True)
   # n.feat_p = L.InnerProduct(n.sigmoid_p, num_output=500, weight_filler=dict(type='xavier'))
   # n.ip2_p = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))    
    n.loss = L.ConstrastiveLoss(n.sigmoid, n.sigmoid_p)#,constrastive_para=dict(margin=1))
    return n.to_proto()

with open('examples/deep-iris/deep_iris_train.prototxt', 'w') as f:
    f.write(str(irisnet('examples/deep-iris/train.txt', 64)))
    
with open('examples/deep-iris/deep_iris_test.prototxt', 'w') as f:
    f.write(str(irisnet('examples/deep-iris/test.txt', 100)))
