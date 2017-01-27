#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

sys.path.append('./caffe/python')
sys.path.append('./caffe')

from caffe.proto import caffe_pb2
from google.protobuf import text_format
import caffe
import numpy as np
import os


########## basic functions ##########
def getBlob(net, blob_name):
	blob_idx = list(net._blob_names).index(blob_name)
	return net._blobs[blob_idx].data

def getParam(net, layer_name, index):
	layer_idx = list(net._layer_names).index(layer_name)
	return net.layers[layer_idx].blobs[index].data

def setBlob(net, blob_name, data):
	blob_idx = list(net._blob_names).index(blob_name)
	np.copyto(net._blobs[blob_idx].data, data)

def setParam(net, layer_name, index, data):
	layer_idx = list(net._layer_names).index(layer_name)
	np.copyto(net.layers[layer_idx].blobs[index].data, data)
	
base_root = "./caffe/models"

default_path = "/bvlc_alexnet"
proto_prifix = "/train_val.prototxt"
solver_prifix = "/solver.prototxt"
weight_prefix = "/bvlc_alexnet.caffemodel"
snapshot_prefix = "/wq_finetune"


"""works = [(None, 2),(None, 3),(None, 4),(None, 5),(None, 6),
	(2,None),(3,None),(4,None),(5,None),
	(2,2),(2,3),(2,4),(2,5),(2,6),
	(3,2),(3,3),(3,4),(3,5),(3,6),
	(4,2),(4,3),(4,4),(4,5),(4,6),
	(5,2),(5,3),(5,4),(5,5),(5,6)]
"""
works = [(2,2)]

for weight_level, act_level in works:
	if weight_level is not None:
		weight_level = int(pow(2, weight_level))
	if act_level is not None:
		act_level = int(pow(2,act_level))
		
	def model_name():
		name = "/"
		if weight_level is None:
			name += "__"
		else:
			name += ("%02d" % weight_level)
		
		if act_level is None:
			name += "__"
		else:
			name += ("%02d" % act_level)
		return name
		

	########## prototxt update ##########
	net_proto = caffe_pb2.NetParameter()

	with open(base_root + default_path + proto_prifix, "r") as f:
		text_format.Merge(str(f.read()), net_proto)

	for l in net_proto.layer:
		if weight_level is not None:
			if l.type == "Convolution":	  
				l.type = "WeightQuantConvolution"
				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.

				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.

				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.
			if l.type == "InnerProduct":
				l.type = "WeightQuantInnerProduct"
				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.

				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.

				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.
		if act_level is not None:
			if l.type == "ReLU":
				l.type = "WeightLogQuantReLU";
				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.

				param =	 l.param.add()
				param.lr_mult = 0.
				param.decay_mult = 0.
			
	def ensure_dir(f):
		d = os.path.dirname(f)
		print(d)
		if not os.path.exists(d):
			os.makedirs(d)		
	ensure_dir(base_root + model_name() + "/")

	with open(base_root + model_name() + proto_prifix, "w") as f:
		f.write(text_format.MessageToString(net_proto))

	## solver update
	solver_proto = caffe_pb2.SolverParameter()

	with open(base_root + default_path + solver_prifix, "r") as f:
		text_format.Merge(str(f.read()), solver_proto)

	solver_proto.net = base_root + model_name() + proto_prifix
	solver_proto.snapshot_prefix = base_root + model_name() + snapshot_prefix

	with open(base_root + model_name() + solver_prifix, "w") as f:
		f.write(text_format.MessageToString(solver_proto))

	########## load parameters ##########
	def getDefaultParam(proto, weight):
		net = caffe.Net(proto,weight,caffe_pb2.TEST)
		
		layers = list(net.layers)
		layer_names = list(net._layer_names)
		
		rtn_map = {}
		for idx in xrange(len(layers)):
			if len(net.layers[idx].blobs) > 0:
				rtn_map[layer_names[idx]] = (layers[idx].type,	[blob.data for blob in net.layers[idx].blobs])
		return rtn_map

	param = getDefaultParam(base_root + default_path + proto_prifix
		, base_root + default_path + weight_prefix)

	########## update parameters ##########
	def updateParam(proto, param, target_param):
		net = caffe.Net(proto, caffe_pb2.TEST)

		for k, v in param.items():
			type, blobs = v
			
			if weight_level is not None and (type == "InnerProduct" or type == "Convolution"):			
				param = blobs[0].flatten()
				blobs.append([0, weight_level, 1])
				quant_info = [0 for idx in xrange(64)]
				half_level = (weight_level+1)//2
				slice = [ int(len(param)*(idx+1)/half_level) for idx in xrange(half_level)]

				quant_info[0:half_level] = slice
				blobs.append(quant_info)
				blobs.append(blobs[0].copy())
				
				for idx, b in enumerate(blobs):
					setParam(net, k, idx, b)
			else:
				for idx, b in enumerate(blobs):
					setParam(net, k, idx, b)
					
					
		if act_level is not None:				
			layers = list(net.layers)
			layer_names = list(net._layer_names)	
					
			for idx, layer in enumerate(layers):
				if layer.type == "WeightLogQuantReLU":
					setParam(net, layer_names[idx], 0 , [act_level, 1] )
					setParam(net, layer_names[idx], 1 , [0 for idx in xrange(256)] )

		net.save(target_param)

	updateParam(base_root + model_name() + proto_prifix, param, 
		base_root + model_name() + weight_prefix)



