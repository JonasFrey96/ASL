#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   09 January 2019

import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

class DenseCRF(object):
	def __init__(self, iter_max=5, pos_w=50, pos_xy_std=1, bi_w=50, bi_xy_std=100, bi_rgb_std=3):
		self.iter_max = iter_max
		self.pos_w = pos_w
		self.pos_xy_std = pos_xy_std
		self.bi_w = bi_w
		self.bi_xy_std = bi_xy_std
		self.bi_rgb_std = bi_rgb_std

	def __call__(self, image, probmap):
		"""
		image: np.array H,W,3 uint8
		probmap: NR_CLASSES, H,W float32 NR_CLASSES = 41 for NYU!
		"""

		C, H, W = probmap.shape
		U = utils.unary_from_softmax(probmap)
		U = U.reshape((C,-1))
		U = np.ascontiguousarray(U)

		image = np.ascontiguousarray(image)

		d = dcrf.DenseCRF2D(W, H, C)
		d.setUnaryEnergy(U)
		d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
		d.addPairwiseBilateral(
				sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
		)

		Q = d.inference(self.iter_max)
		Q = np.array(Q).reshape((C, H, W))
		label = np.argmax(Q, axis=0)
		return label, Q