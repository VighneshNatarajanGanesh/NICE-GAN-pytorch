import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
# import torch.utils.tensorboard as tensorboardX
from thop import profile
from thop import clever_format
import h5py

class NICE(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'NICE_light'
        else :
            self.model_name = 'NICE'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.recon_weight = args.recon_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        # add support for different channel input and output images:
        self.img_ch_a = args.img_ch_a
        self.img_ch_b = args.img_ch_b

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.start_iter = 1

        self.fid = 1000
        self.fid_A = 1000
        self.fid_B = 1000
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel a: ", self.img_ch_a)
        print("# the size of image channel b: ", self.img_ch_b)
        print("# base channel number per layer : ", self.ch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# recon_weight : ", self.recon_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """

        # we have fifferent transformations as we have different number of channels which require
        # different arguments for Normalize function
        train_transform_a = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, )*self.img_ch_a, std=(0.5, )*self.img_ch_a)
        ]) # removed horizontal flip, resize and random_crop because they use pil which doesn't suppert multichannel images above 3

        train_transform_b = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,)*self.img_ch_b, std=(0.5,)*self.img_ch_b)
        ])


        test_transform_a = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, )*self.img_ch_a, std=(0.5, )*self.img_ch_a)
        ]) # removed Resize as it uses PIL

        test_transform_b = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,)*self.img_ch_b, std=(0.5,)*self.img_ch_b)
        ])

        # thereform note: the images must be preprocessed properly!! with all the augmentation and cropping!


        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform_a)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform_b)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform_a)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform_b)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False,pin_memory=True)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False,pin_memory=True)

        """ Define Generator, Discriminator """
        self.gen2B = ResnetGenerator(input_nc=self.img_ch_a, output_nc=self.img_ch_b, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch_b, output_nc=self.img_ch_a, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch_a, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch_b, ndf=self.ch, n_layers=self.n_dis).to(self.device)
        
        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch_a, self.img_size, self.img_size]).to(self.device)
        macs, params = profile(self.disA, inputs=(input, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        _,_, _,  _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(real_A_ae, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """ 
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)


    def train(self):
        # writer = tensorboardX.SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))
        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

        self.start_iter = 1
        if self.resume:
            params = torch.load(os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
            self.gen2B.load_state_dict(params['gen2B'])
            self.gen2A.load_state_dict(params['gen2A'])
            self.disA.load_state_dict(params['disA'])
            self.disB.load_state_dict(params['disB'])
            self.D_optim.load_state_dict(params['D_optimizer'])
            self.G_optim.load_state_dict(params['G_optimizer'])
            self.start_iter = params['start_iter']+1
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
            print("ok")
          

        # training loop
        testnum = 4
        for step in range(1, self.start_iter):
            if step % self.print_freq == 0:
                for _ in range(testnum):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = testB_iter.next()

        print("self.start_iter",self.start_iter)
        print('training start !')
        start_time = time.time()
        for step in range(self.start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            real_LA_logit,real_GA_logit, real_A_cam_logit, _, real_A_z = self.disA(real_A)
            real_LB_logit,real_GB_logit, real_B_cam_logit, _, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, _ = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, _ = self.disB(fake_A2B)


            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))            
            D_ad_cam_loss_A = self.MSE_loss(real_A_cam_logit, torch.ones_like(real_A_cam_logit).to(self.device)) + self.MSE_loss(fake_A_cam_logit, torch.zeros_like(fake_A_cam_logit).to(self.device))
            D_ad_cam_loss_B = self.MSE_loss(real_B_cam_logit, torch.ones_like(real_B_cam_logit).to(self.device)) + self.MSE_loss(fake_B_cam_logit, torch.zeros_like(fake_B_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_A + D_ad_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_B + D_ad_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()
            # writer.add_scalar('D/%s' % 'loss_A', D_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('D/%s' % 'loss_B', D_loss_B.data.cpu().numpy(), global_step=step)  

            # Update G
            self.G_optim.zero_grad()

            _,  _,  _, _, real_A_z = self.disA(real_A)
            _,  _,  _, _, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A_z)
            fake_B2A = self.gen2A(real_B_z)

            fake_LA_logit, fake_GA_logit, fake_A_cam_logit, _, fake_A_z = self.disA(fake_B2A)
            fake_LB_logit, fake_GB_logit, fake_B_cam_logit, _, fake_B_z = self.disB(fake_A2B)
            
            fake_B2A2B = self.gen2B(fake_A_z)
            fake_A2B2A = self.gen2A(fake_B_z)


            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_ad_cam_loss_A = self.MSE_loss(fake_A_cam_logit, torch.ones_like(fake_A_cam_logit).to(self.device))
            G_ad_cam_loss_B = self.MSE_loss(fake_B_cam_logit, torch.ones_like(fake_B_cam_logit).to(self.device))

            G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

            fake_A2A = self.gen2A(real_A_z)
            fake_B2B = self.gen2B(real_B_z)

            G_recon_loss_A = self.L1_loss(fake_A2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2B, real_B)


            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_A + G_ad_loss_LA ) + self.cycle_weight * G_cycle_loss_A + self.recon_weight * G_recon_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_B + G_ad_loss_LB ) + self.cycle_weight * G_cycle_loss_B + self.recon_weight * G_recon_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()
            # writer.add_scalar('G/%s' % 'loss_A', G_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('G/%s' % 'loss_B', G_loss_B.data.cpu().numpy(), global_step=step)  

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

            # for name, param in self.gen2B.named_parameters():
            #     writer.add_histogram(name + "_gen2B", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.gen2A.named_parameters():
            #     writer.add_histogram(name + "_gen2A", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.disA.named_parameters():
            #     writer.add_histogram(name + "_disA", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.disB.named_parameters():
            #     writer.add_histogram(name + "_disB", param.data.cpu().numpy(), global_step=step)

            
            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % self.print_freq == 0:
                print('current D_learning rate:{}'.format(self.D_optim.param_groups[0]['lr']))
                print('current G_learning rate:{}'.format(self.G_optim.param_groups[0]['lr']))
                self.save_path("_params_latest.pt",step)

            if step % self.print_freq == 0:
                train_sample_num = testnum
                test_sample_num = testnum
                A2B = np.zeros((self.img_size * 5, 0, 3))
                B2A = np.zeros((self.img_size * 5, 0, 3))

                # the next 4 will be used only if both domains are not rgb
                rgbDomain = np.zeros((self.img_size * 4, 0, self.img_ch_a))
                genNonRgbDomain = np.zeros((self.img_size, 0, self.img_ch_b))
                nonRgbDomain = np.zeros((self.img_size * 4, 0, self.img_ch_b))
                genRgbDomain = np.zeros((self.img_size, 0, self.img_ch_a))

                self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

                self.gen2B,self.gen2A = self.gen2B.to('cpu'), self.gen2A.to('cpu')
                self.disA,self.disB = self.disA.to('cpu'), self.disB.to('cpu')
                for sample_num_i in range(train_sample_num):
                    try:
                        real_A, _ = trainA_iter.next()
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, _ = trainA_iter.next()

                    try:
                        real_B, _ = trainB_iter.next()
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B, _ = trainB_iter.next()
                    real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
                    # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    
                    _, _,  _, A_heatmap, real_A_z= self.disA(real_A)
                    _, _,  _, B_heatmap, real_B_z= self.disB(real_B)

                    fake_A2B = self.gen2B(real_A_z)
                    fake_B2A = self.gen2A(real_B_z)

                    _, _,  _,  _,  fake_A_z = self.disA(fake_B2A)
                    _, _,  _,  _,  fake_B_z = self.disB(fake_A2B)

                    fake_B2A2B = self.gen2B(fake_A_z)
                    fake_A2B2A = self.gen2A(fake_B_z)

                    fake_A2A = self.gen2A(real_A_z)
                    fake_B2B = self.gen2B(real_B_z)

                    flag = 0 # flag will be one if a try fails and except is reached

                    try: # both domains are rgb image
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(B_heatmap[0]), self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                    except: # domain a is rgb domain b is hsi
                        flag = 1

                        rgbDomain = np.concatenate((rgbDomain, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                               cam(tensor2numpy(A_heatmap[0]), self.img_size),
                                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)
                        genNonRgbDomain = np.concat((nonRgbDomain, tensor2numpy(denorm(fake_A2B[0]))), 1)

                        nonRgbDomain = np.concatenate((nonRgbDomain, np.concatenate((tensor2numpy(denorm(real_B[0])),
                                                                                     cam(tensor2numpy(B_heatmap[0]), self.img_size),
                                                                                     tensor2numpy(denorm(fake_B2B[0])),
                                                                                     tensor2numpy(denorm(fake_B2A2B[0]))), 0)), 1)
                        genRgbDomain = np.concat((nonRgbDomain, RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = testB_iter.next()
                    real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
                    # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    _, _,  _, A_heatmap, real_A_z= self.disA(real_A)
                    _, _,  _, B_heatmap, real_B_z= self.disB(real_B)

                    fake_A2B = self.gen2B(real_A_z)
                    fake_B2A = self.gen2A(real_B_z)

                    _, _,  _,  _,  fake_A_z = self.disA(fake_B2A)
                    _, _,  _,  _,  fake_B_z = self.disB(fake_A2B)

                    fake_B2A2B = self.gen2B(fake_A_z)
                    fake_A2B2A = self.gen2A(fake_B_z)

                    fake_A2A = self.gen2A(real_A_z)
                    fake_B2B = self.gen2B(real_B_z)

                    try:
                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)
                    except: # it is not, then write it as a hdf5
                        flag = 1
                        rgbDomain = np.concatenate((rgbDomain, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                               cam(tensor2numpy(A_heatmap[0]), self.img_size),
                                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)
                        genNonRgbDomain = np.concat((nonRgbDomain, tensor2numpy(denorm(fake_A2B[0]))), 1)

                        nonRgbDomain = np.concatenate((nonRgbDomain, np.concatenate((tensor2numpy(denorm(real_B[0])),
                                                           cam(tensor2numpy(B_heatmap[0]), self.img_size),
                                                           tensor2numpy(denorm(fake_B2B[0])),
                                                           tensor2numpy(denorm(fake_B2A2B[0]))), 0)), 1)
                        genRgbDomain = np.concat((nonRgbDomain, RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))), 1)


                # write the images:
                if flag == 0:
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                else:
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'rgbSource_%07d.png' % step), rgbDomain * 255.0)
                    with h5py.File(os.path.join(self.result_dir, self.dataset, 'img', 'genNonRgb_%07d.png' % step), 'w') as f:
                        dset = f.create_dataset('hs_data', data=genNonRgbDomain)
                    with h5py.File(os.path.join(self.result_dir, self.dataset, 'img', 'nonRgbSource_%07d.png' % step), 'w') as f:
                        dset = f.create_dataset('hs_data', data=nonRgbDomain)
                    cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'genRgb_%07d.png' % step), genRgbDomain * 255.0)

                self.gen2B,self.gen2A = self.gen2B.to(self.device), self.gen2A.to(self.device)
                self.disA,self.disB = self.disA.to(self.device), self.disB.to(self.device)
                
                self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

    def save(self, dir, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))


    def save_path(self, path_g,step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.result_dir, self.dataset + path_g))

    def load(self):
        params = torch.load(os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.start_iter = params['start_iter']

    def test(self):
        self.load()
        print(self.start_iter)

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(),self.disB.eval()
        for n, (real_A, real_A_path) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            _, _,  _, _, real_A_z= self.disA(real_A)
            fake_A2B = self.gen2B(real_A_z)

            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
            print(real_A_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeB', real_A_path[0].split('/')[-1]), A2B * 255.0)

        for n, (real_B, real_B_path) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)
            _, _,  _, _, real_B_z= self.disB(real_B)
            fake_B2A = self.gen2A(real_B_z)

            B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
            print(real_B_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeA', real_B_path[0].split('/')[-1]), B2A * 255.0)
