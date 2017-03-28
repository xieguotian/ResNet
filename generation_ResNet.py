head = """
opt_memory: true
opt_test_shared_memory: true
name: "ResNet281_opt"
layer {
	name: "data"
	type: "Data"
	top: "data"
	top: "label"
	include {
		phase: TRAIN
	}
	transform_param {
		mirror: true
		crop_size: 224
		mean_value: 104.007
		mean_value: 116.669
		mean_value: 122.679
		force_color: true
	}
	data_param {
		source: "classification/data/imagenet_train_lmdb_s"
		batch_size: 32
		backend: LMDB
	}
}

#### conv1 downsample to 4 times smaller. ########
layer {
	name: "conv1"
	type: "Convolution"
	bottom: "data"
	top: "conv1"
	param {
		lr_mult: 1
		decay_mult: 1
	}
	param {
    	lr_mult: 1
    	decay_mult: 0
  	}
	convolution_param {
		num_output: 64
		kernel_size: 7
		pad: 3
		stride: 2
		
		weight_filler {
			type: "msra"
			variance_norm: FAN_OUT
		}
		bias_filler {
      		type: "constant"
      		value: 0
    	}
	}
}

layer {
	name: "bn_conv1"
	type: "BatchNormTorch"
	bottom: "conv1"
	top: "bn_conv1"
	param {
		lr_mult: 0
	}
	param {
		lr_mult: 0
	}
	param {
		lr_mult: 0
	}
	param {
		lr_mult: 1
		decay_mult: 0
	}
	param {
		lr_mult: 1
		decay_mult: 0
	}
	scale_param {
		bias_term: true
	}
}

layer {
	name: "conv1_relu"
	type: "ReLU"
	bottom: "bn_conv1"
	top: "bn_conv1"
}

layer {
	name: "pool1"
	type: "Pooling"
	bottom: "bn_conv1"
	top: "pool1"
	pooling_param {
		pool: MAX
		kernel_size: 3
		stride: 2
		pad: 1
		ceil_mode:false
	}
}


"""

prj_bottleneck = """
### p1 ###
layer {{
	name: "res{0}_branch2a"
	type: "Convolution"
	bottom: "{1}"
	top: "res{0}_branch2a"
	param {{
		lr_mult: 1
		decay_mult: 1
	}}
	param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
	convolution_param {{
		
		num_output: {2}
		kernel_size: 1
		weight_filler {{
			type: "msra"
			variance_norm: FAN_OUT
		}}
		bias_filler {{
      		type: "constant"
      		value: 0
    	}}
	}}

}}
layer {{
	name: "bn{0}_branch2a"
	type: "BatchNormTorch"
	bottom: "res{0}_branch2a"
	top: "bn{0}_branch2a"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}

layer {{
	name: "res{0}_branch2a_relu"
	type: "ReLU"
	bottom: "bn{0}_branch2a"
	top: "bn{0}_branch2a"
}}
### p2 ###
layer {{
	name: "res{0}_branch2b"
	type: "Convolution"
	bottom: "bn{0}_branch2a"
	top: "res{0}_branch2b"
	param {{
		lr_mult: 1
		decay_mult: 1
	}}
	param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
	convolution_param {{
		
		num_output: {2}
		kernel_size: 3
		pad: 1
		stride: {3}
		weight_filler {{
		type: "msra"
		variance_norm: FAN_OUT
		}}
		bias_filler {{
      		type: "constant"
      		value: 0
    	}}
	}}
}}
layer {{
	name: "bn{0}_branch2b"
	type: "BatchNormTorch"
	bottom: "res{0}_branch2b"
	top: "bn{0}_branch2b"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}
layer {{
	name: "res{0}_branch2b_relu"
	type: "ReLU"
	bottom: "bn{0}_branch2b"
	top: "bn{0}_branch2b"
}}

### p3 ###
layer {{
	name: "res{0}_branch2c"
	type: "Convolution"
	bottom: "bn{0}_branch2b"
	top: "res{0}_branch2c"
	param {{
		lr_mult: 1
		decay_mult: 1
	}}
	param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
	convolution_param {{
		
		num_output: {4}
		kernel_size: 1
		weight_filler {{
			type: "msra"
			variance_norm: FAN_OUT
		}}
		bias_filler {{
      		type: "constant"
      		value: 0
    	}}
	}}
}}
layer {{
	name: "bn{0}_branch2c"
	type: "BatchNormTorch"
	bottom: "res{0}_branch2c"
	top: "bn{0}_branch2c"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}

### projection skip connect ###
layer{{
	name: "res{0}_branch1"
	type: "Convolution"
	bottom: "{1}"
	top: "res{0}_branch1"
    param {{
      lr_mult: 1
      decay_mult: 1
    }}
    param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
    convolution_param {{
       
       num_output: {4}
       kernel_size: 1
	   stride: {3}
       weight_filler {{
         type: "msra"
        variance_norm: FAN_OUT
    }}
    bias_filler {{
      	type: "constant"
      	value: 0
    }}
  }}
}}
layer {{
	name: "bn{0}_branch1"
	type: "BatchNormTorch"
	bottom: "res{0}_branch1"
	top: "bn{0}_branch1"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}

layer {{
	name: "res{0}"
	type: "Eltwise"
	bottom: "bn{0}_branch2c"
	bottom: "bn{0}_branch1"
	top: "bn{0}_branch2c"

	eltwise_param{{
		operation: SUM
	}}
}}

layer {{
	name: "res{0}_relu"
	type: "ReLU"
	bottom: "bn{0}_branch2c"
	top: "bn{0}_branch2c"
}}

"""

identity_bottleneck = """
### p1 ###
layer {{
	name: "res{0}_branch2a"
	type: "Convolution"
	bottom: "{1}"
	top: "res{0}_branch2a"
	param {{
		lr_mult: 1
		decay_mult: 1
	}}
	param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
	convolution_param {{
		
		num_output: {2}
		kernel_size: 1
		weight_filler {{
			type: "msra"
			variance_norm: FAN_OUT
		}}
		bias_filler {{
			type: "constant"
			value: 0
		}}
	}}
}}
layer {{
	name: "bn{0}_branch2a"
	type: "BatchNormTorch"
	bottom: "res{0}_branch2a"
	top: "bn{0}_branch2a"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}
layer {{
	name: "res{0}_branch2a_relu"
	type: "ReLU"
	bottom: "bn{0}_branch2a"
	top: "bn{0}_branch2a"
}}
### p2 ###
layer {{
	name: "res{0}_branch2b"
	type: "Convolution"
	bottom: "bn{0}_branch2a"
	top: "res{0}_branch2b"
	param {{
		lr_mult: 1
		decay_mult: 1
	}}
	param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
	convolution_param {{
		
		num_output: {2}
		kernel_size: 3
		stride: {3}
		pad: 1
		weight_filler {{
		type: "msra"
		variance_norm: FAN_OUT
		}}
		bias_filler {{
			type: "constant"
			value: 0
		}}
	}}
}}
layer {{
	name: "bn{0}_branch2b"
	type: "BatchNormTorch"
	bottom: "res{0}_branch2b"
	top: "bn{0}_branch2b"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}
layer {{
	name: "res{0}_branch2b_relu"
	type: "ReLU"
	bottom: "bn{0}_branch2b"
	top: "bn{0}_branch2b"
}}

### p3 ###
layer {{
	name: "res{0}_branch2c"
	type: "Convolution"
	bottom: "bn{0}_branch2b"
	top: "res{0}_branch2c"
	param {{
		lr_mult: 1
		decay_mult: 1
	}}
	param {{
    	lr_mult: 1
    	decay_mult: 0
  	}}
	convolution_param {{
		
		num_output: {4}
		kernel_size: 1
		weight_filler {{
			type: "msra"
			variance_norm: FAN_OUT
		}}
		bias_filler {{
      		type: "constant"
      		value: 0
    	}}
	}}
}}
layer {{
	name: "bn{0}_branch2c"
	type: "BatchNormTorch"
	bottom: "res{0}_branch2c"
	top: "bn{0}_branch2c"
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	param {{
		lr_mult: 1
		decay_mult: 0
	}}
	scale_param {{
		bias_term: true
	}}
}}
### identity skip connect ###
layer {{
	name: "res{0}"
	type: "Eltwise"
	bottom: "bn{0}_branch2c"
	bottom: "{1}"
	top: "bn{0}_branch2c"

	eltwise_param{{
		operation: SUM
	}}
}}

layer {{
	name: "res{0}_relu"
	type: "ReLU"
	bottom: "bn{0}_branch2c"
	top: "bn{0}_branch2c"
}}


"""

classifier_block = """
#### classifier ####

layer {
  name: "pool5"
  type: "Pooling"
  bottom: "bn5c_branch2c"
  top: "pool5"
  pooling_param {
    pool: AVE
	kernel_size: 7
	stride: 1
	ceil_mode: false
  }
}

layer {
  name: "fc1000"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc1000"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 1
    decay_mult: 0
  }
  inner_product_param {
    num_output: 1000
    weight_filler {
      type: "msra"
	variance_norm: FAN_OUT
    }

    bias_filler {
      	type: "constant"
      	value: 0
    }
  }
}

layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "fc1000"
  bottom: "label"
  top: "loss"
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "fc1000"
  bottom: "label"
  top: "accuracy"
}
"""
#num = [3,8,36,3] # 152 layer residual
#num = [3,24,36,3] # 200 layer residual
#num = [3,8,52,3] # old 200 layer residual
#num = [3,18,52,3] # 230 layer residual
#num = [7,30,52,11] # 302 layer residual
#num = [5,25,52,11] # 281 layer residual
num = [3,4,6,3]    # 50 layer residual
#num = [3,4,23,3] #101 layer
#in_dim = [64,256,512,1024]
#out_dim = [256,512,1024,2048]
in_dim = [64,128,256,512]
out_dim = [256,512,1024,2048]
key_w = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
with open('ResNet50.prototxt','w') as fid:
	print >>fid,head
	for idx,num_block in enumerate(num):
		if idx<len(num)-1:
			if idx==0:
				name = '%d%s'%(idx+2,key_w[0])
				print >>fid,prj_bottleneck.format(name,'pool1',in_dim[idx],1,out_dim[idx],idx)
				for i in range(2,num_block+1):
					name = '%d%s'%(idx+2, key_w[i-1])
					name_last = 'bn%d%s_branch2c'%(idx+2,key_w[i-2])
					print >>fid,identity_bottleneck.format(name,name_last,in_dim[idx],1,out_dim[idx],idx)
				name_last = 'bn%d%s_branch2c' % (idx + 2, key_w[num[idx] - 1])
			else:
				name = '%d%s'%(idx+2,key_w[0])
#				name_last = 'bn%d%s_branch2c'%(idx+1,key_w[num[idx-1]-1])
				print >>fid,prj_bottleneck.format(name,name_last,in_dim[idx],2,out_dim[idx],idx)
				name_last = 'bn%d%s_branch2c' % (idx + 2, key_w[0])
				for i in range(2,num_block+1):
					name = '%db%d'%(idx+2,i-1)#key_w[i-1])
					#name_last = 'res%db%d_branch2c_s'%(idx+2,i-2)#key_w[i-2])
					print >>fid,identity_bottleneck.format(name,name_last,in_dim[idx],1,out_dim[idx],idx)
					name_last = 'bn%db%d_branch2c' % (idx + 2, i - 1)  # key_w[i-2])
				name_last = 'bn%db%d_branch2c' % (idx + 2, num_block - 1)  # key_w[i-2])
			#name = '%d%s'%(idx+1,key_w[num_block-1])
			#name_last = 'res%d%s_branch2c'%(idx+1,key_w[num_block-2])
			#print >>fid,subsample_bottleneck.format(name,name_last,in_dim[idx],2,out_dim[idx],idx)

		else:
			name = '%d%s'%(idx+2,key_w[0])
			#name_last = 'res%d%s_branch2c_s'%(idx+1,key_w[num[idx-1]-1])
			print >>fid,prj_bottleneck.format(name,name_last,in_dim[idx],2,out_dim[idx],idx)
			name_last = 'bn%d%s_branch2c'%(idx+2,key_w[0])
			for i in range(2,num_block+1):
				name = '%d%s'%(idx+2,key_w[i-1])
				#name_last = 'res%d%s_branch2c_s'%(idx+2,key_w[i-2])
				print >>fid,identity_bottleneck.format(name,name_last,in_dim[idx],1,out_dim[idx],idx)
				name_last = 'bn%d%s_branch2c' % (idx + 2, key_w[i - 1])
		'''
		if idx==0:
			name = '%d_%d'%(idx+1,1)
			print >>fid,prj_bottleneck.format(name,'pool1',in_dim[idx],1,out_dim[idx])
			for i in range(2,num_block+1):
				name = '%d_%d'%(idx+1,i)
				name_last = 'res_out%d_%d'%(idx+1,i-1)
				print >>fid,identity_bottleneck.format(name,name_last,in_dim[idx],1,out_dim[idx],idx)
		else:
			name = '%d_%d'%(idx+1,1)
			name_last = 'res_out%d_%d'%(idx,num[idx-1])
			print >>fid,prj_bottleneck.format(name,name_last,in_dim[idx],2,out_dim[idx])
			for i in range(2,num_block+1):
				name = '%d_%d'%(idx+1,i)
				name_last = 'res_out%d_%d'%(idx+1,i-1)
				print >>fid,identity_bottleneck.format(name,name_last,in_dim[idx],1,out_dim[idx],idx)
		'''
	print >>fid,classifier_block

