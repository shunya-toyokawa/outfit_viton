import torch
from .base_model import BaseModel
from . import networks
from util.visualizer import tensor2label
from util.util import mix_map


class OvtonInferenceOnlineModel(BaseModel):
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
            parser.add_argument('--lambda_VGG', type=float, default=10.0, help='weight for VGG loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.label_nc = 20
        self.visual_names = ['img_A', 'img_B', 'fake_img']
        self.model_names = ['Shape', 'App', 'D']
        # define networks (both generator and discriminator)
        self.netShape = networks.define_G(opt.input_nc, 20, opt.ngf, 'ov_resnet_shape6blocks', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netApp = networks.define_G(opt.input_nc, 3, opt.ngf, 'ov_resnet_app9blocks', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                      opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        # vgg19 perceptual loss
        self.vggnet_fix = networks.VGG19_feature_color_torchversion()
        self.vggnet_fix.load_state_dict(torch.load('models/vgg19_conv.pth'))
        self.vggnet_fix.eval()
        for param in self.vggnet_fix.parameters():
            param.requires_grad = False
        self.vggnet_fix.to(self.device)
        self.criterionMSE = torch.nn.MSELoss()

        if isinstance(self.netShape, torch.nn.DataParallel):
            self.netShape = self.netShape.module
            self.netShape.eval()
        if isinstance(self.netApp, torch.nn.DataParallel):
            self.netApp = self.netApp.module
            self.netApp.eval()

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netApp.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_G)

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
        self.parse_B = input['parse_B'].to(self.device)

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
        self.ref_img = self.netApp(self.seg_B, self.parse_B, self.img_B)
        self.fake_img = self.netApp.inference(self.fake_seg, self.seg_A, self.img_A, self.seg_B, self.img_B)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        localize_mask = torch.sum(self.seg_B[:, 5:8, :, :], dim=1)
        roi_ref = localize_mask * self.img_B - (1 - localize_mask)
        roi_fake = localize_mask * self.fake_img - (1 - localize_mask)
        localize_mask_A = torch.sum(self.seg_A[:, 5:8, :, :], dim=1)
        bg_ref = (1 - localize_mask_A) * self.img_A - localize_mask_A
        bg_fake = (1 - localize_mask) * self.fake_img - localize_mask
        roi_seg_B = localize_mask * self.seg_B
        pred_rec_B = self.netD(torch.cat((roi_seg_B, roi_ref), 1))
        self.loss_GAN_ref = self.criterionGAN(pred_rec_B, True)

        pred_fake_B = self.netD(torch.cat((self.fake_seg, roi_fake), 1))
        self.loss_GAN_que = self.criterionGAN(pred_fake_B, True)

        # VGG Loss
        real_features = self.vggnet_fix(torch.cat((roi_ref, bg_ref), dim=0), ['r12', 'r22', 'r32', 'r42', 'r52'])
        fake_features = self.vggnet_fix(torch.cat((roi_fake, bg_fake), dim=0), ['r12', 'r22', 'r32', 'r42', 'r52'])
        self.loss_G_VGG = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for i, (real_feature, fake_feature) in enumerate(zip(real_features, fake_features)):
            self.loss_G_VGG += self.criterionMSE(real_feature, fake_feature) * weights[i]
        self.loss_G_VGG  = self.loss_G_VGG * 5  # self.opt.lambda_VGG
        # combine loss and calculate gradients
        self.loss_G = self.loss_GAN_ref + self.loss_G_VGG + self.loss_GAN_que
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights


