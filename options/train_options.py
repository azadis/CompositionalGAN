# ========================================================
# Compositional GAN
# Options for the training parameters
# By Samaneh Azadi
# ========================================================


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--n_latest', type=int, default=5, help='save the last n images with frequency update_html_freq')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--niterSTN', type=int, default=0, help='# of iter for pre-training STN')
        self.parser.add_argument('--niterCompletion', type=int, default=0, help='# of iter for pre-training Completion network')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_L2', type=float, default=100.0, help='weight for L2 loss')
        self.parser.add_argument('--lambda_gp', type=float, default=10.0, help='weight for gradient penalty loss')
        self.parser.add_argument('--lambda_gan', type=float, default=1.0, help='weight for gan loss')
        self.parser.add_argument('--lambda_mask', type=float, default=50.0, help='weight for mask prediction')
        self.parser.add_argument('--lambda_STN', type=float, default=1.0, help='weight for STN l1 loss')
        self.parser.add_argument('--lambda_AFN', type=float, default=100.0, help='weight for AFN loss in the end-to-end framework')
        self.parser.add_argument('--lambda_identity', type=float, default=1.0, help='weight for identity loss in baseline cycle-gan framework')
        self.parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for l1 loss in in baseline cycle-gan framework')
        self.parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for l1 loss in baseline cycle-gan framework')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--eval', action='store_true', help='run in eval mode')

        self.isTrain = True
