"""library for unit testing the neural network architectures"""
import unittest

from kernel_eval.models import vgg11, vgg13, vgg16, vgg19

class TestVGG(unittest.TestCase):
    """class for unit testing the VGG neural network architectures"""

    def test_vgg11(self):
        """test the VGG11 neural network architecture"""
        model = vgg11()
        self.assertEqual(21, len(model.features))

    def test_vgg13(self):
        """test the VGG13 neural network architecture"""
        model = vgg13()
        self.assertEqual(25, len(model.features))

    def test_vgg16(self):
        """test the VGG16 neural network architecture"""
        model = vgg16()
        self.assertEqual(31, len(model.features))

    def test_vgg19(self):
        """test the VGG19 neural network architecture"""
        model = vgg19()
        self.assertEqual(37, len(model.features))
