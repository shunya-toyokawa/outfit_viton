import torch
from .base_model import BaseModel
from . import networks
from util.visualizer import tensor2label
from util.util import mix_map


class OvtonInferenceModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='ov_resnet_app9blocks', dataset_mode='ov')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_VGG', type=float, default=1.0, help='weight for VGG loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.label_nc = 20
        self.visual_names = ['vis_seg_A', 'vis_seg_B', 'vis_seg_mixed', 'vis_fake_seg', 'img_A', 'img_B', 'fake_img']
        self.model_names = ['Shape', 'App']
        # define networks (both generator and discriminator)
        self.netShape = networks.define_G(opt.input_nc, 20, opt.ngf, 'ov_resnet_shape6blocks', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netApp = networks.define_G(opt.input_nc, 3, opt.ngf, 'ov_resnet_app9blocks', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if isinstance(self.netShape, torch.nn.DataParallel):
            self.netShape = self.netShape.module
            self.netShape.eval()
        if isinstance(self.netApp, torch.nn.DataParallel):
            self.netApp = self.netApp.module
            self.netApp.eval()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.seg_A = input['seg_map_A'].to(self.device)
        self.parse_A = input['parse_A'].to(self.device)
        self.dense_map = input['dense_map'].to(self.device)
        self.seg_B = input['seg_map_B'].to(self.device)

        self.input_seg_mixed = mix_map(self.seg_A, self.seg_B)
        self.img_A = input['img_A'].to(self.device)
        self.img_B = input['img_B'].to(self.device)
        self.image_paths = input['A_path']

        self.vis_seg_A = tensor2label(self.seg_A[0].cpu(), self.label_nc)
        self.vis_seg_B = tensor2label(self.seg_B[0].cpu(), self.label_nc)
        self.vis_seg_mixed = tensor2label(self.input_seg_mixed[0].cpu(), self.label_nc)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_seg = self.netShape(self.input_seg_mixed, self.dense_map)
        self.vis_fake_seg = tensor2label(self.fake_seg[0].detach().cpu(), self.label_nc)

        self.fake_img = self.netApp.inference(self.fake_seg, self.seg_A, self.img_A, self.seg_B, self.img_B)

    def optimize_parameters(self):
        pass


