from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--input_path', type=str, default='/fs1/home/tjuvis_2022/lxx/GANimation-master/sample_dataset/imgs/N_0000000356_00190.jpg',help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self.is_train = False
