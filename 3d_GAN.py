from ops import *
from utlis import *
import argparse,sys

class VAE_GAN:  
    def __init__(self, logdir, epoch, batch, len_vector, len_x, save_iner, save_path, len_data, save_im_inter, print_log_inter, record_dir):
        self.logdir = logdir
#         self.record_dir = 'E:/zdelete/train.tfrecords'
        self.record_dir = record_dir
                                       
        self.epoch = epoch
        self.batch_size = batch
        self.save_iner = save_iner
        self.len_vector = len_vector
        self.save_path = save_path
        self.len_data = len_data
        self.save_im_inter = save_im_inter
        self.print_log_inter = print_log_inter
        self.zeros = np.zeros([batch,len_x,len_x,len_x,1],np.float32)
        
        self.x = tf.placeholder(tf.float32,[None,len_x,len_x,len_x,1],'X')
        self.x_test = tf.placeholder(tf.float32,[None,len_x,len_x,len_x,1],'X_test')
        #  calculate: scale * vector + offset for explore the latent space
        self.create_save_file()
    
    def create_save_file(self):
        for i in range(self.epoch):
            for j in range(self.len_data//self.batch_size):
                if (i*self.len_data+j*self.batch_size)%self.save_im_inter==0:
                    dir_ = self.save_path+'output_ims'+'/'+'epoch-'+str(i)+'_sample-'+str(j)
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
             
    def build_VAE(self, name, x, in_train, len_vector, reuse):  # input：[b,64,64,64,1] ----> 出：[b,len_z] --> m
        with tf.variable_scope(name,reuse=reuse):               #                       |-->     [b,len_z] --> m_2
            conv1 = conv3d('conv1', x, 4, 4*4, 64, 2, True)     #                       |-->     [b,len_z] --> fai_2
            bn1 = BN('bn1', conv1, in_train)                    #                       |-->     [b,len_z] --> log_fai_2
            relu1 = lrelu('relu1', bn1)
                                                                                                                                          
            conv2 = conv3d('conv2', relu1, 4, 4*4, 128, 2, True)
            bn2 = BN('bn2', conv2, in_train)
            relu2 = lrelu('relu2', bn2)
            
            conv3 = conv3d('conv3', relu2, 4, 4*4, 256, 2, True)
            bn3 = BN('bn3', conv3, in_train)
            relu3 = lrelu('relu3', bn3)
            
            conv4 = conv3d('conv4', relu3, 4, 4*4, 512, 2, True)
            bn4 = BN('bn4', conv4, in_train)
            relu4 = lrelu('relu4', bn4)
            
            conv5 = conv3d('conv5', relu4, 4, 4*4, self.len_vector, 1, False)
            bn5 = BN('bn5', conv5, in_train)
            m_abs = tf.reshape(tf.abs(bn5),[-1,self.len_vector])
            m_2 = tf.square(bn5)
            mean_2 = tf.reshape(m_2,[-1,self.len_vector])
            
            conv6 = conv3d('conv6', relu4, 4, 4*4, self.len_vector, 1, False)
            bn6 = BN('bn6', conv6, in_train)
            
            fai_2 = tf.reshape(tf.square(bn6),[-1,self.len_vector])
            
            log_fai_2 = tf.reshape(tf.log(fai_2),[-1,self.len_vector])
            
            return m_abs,mean_2,fai_2,log_fai_2
    
    def build_G(self, name, x, reuse, in_train):  #  输入：[b, vector] -->  输出：[b, 64, 64, 64, 1]
        with tf.variable_scope(name,reuse=reuse):
            z = tf.reshape(x, [self.batch_size, 1, 1, 1, self.len_vector])
            
            conv1 = conv3d_trans('conv1', z, 4, 4*4, 512, 1, False, [self.batch_size,4,4,4,512])
            bn1 = BN('bn1', conv1, in_train)
            relu1 = relu('relu1', bn1)

            conv2 = conv3d_trans('conv2', relu1, 4, 4*4, 256, 2, True, [self.batch_size,8,8,8,256])
            bn2 = BN('bn2', conv2, in_train)
            relu2 = relu('relu2', bn2)

            conv3 = conv3d_trans('conv3', relu2, 4, 4*4, 128, 2, True, [self.batch_size,16,16,16,128])
            bn3 = BN('bn3', conv3, in_train)
            relu3 = relu('relu3', bn3)

            conv4 = conv3d_trans('conv4', relu3, 4, 4*4, 64, 2, True, [self.batch_size,32,32,32,64])
            bn4 = BN('bn4', conv4, in_train)
            relu4 = relu('relu4', bn4)

            conv5 = conv3d_trans('conv5', relu4, 2, 4*4, 1, 2, True, [self.batch_size,64,64,64,1])
            sigmoid = tf.nn.sigmoid(conv5)
            return sigmoid
        
    def build_D(self, name, x, reuse, in_train):      #  输入：[b, 64, 64, 64, 1] -->  输出：[b, 1, 1, 1, 1]
        with tf.variable_scope(name,reuse=reuse):
            conv1 = conv3d('conv1', x, 4, 4*4, 64, 2, True)
            bn1 = BN('bn1', conv1, in_train)
            relu1 = lrelu('relu1', bn1)
            
            conv2 = conv3d('conv2', relu1, 4, 4*4, 128, 2, True)
            bn2 = BN('bn2', conv2, in_train)
            relu2 = lrelu('relu2', bn2)
            
            conv3 = conv3d('conv3', relu2, 4, 4*4, 256, 2, True)
            bn3 = BN('bn3', conv3, in_train)
            relu3 = lrelu('relu3', bn3)
            
            conv4 = conv3d('conv4', relu3, 4, 4*4, 512, 2, True)
            bn4 = BN('bn4', conv4, in_train)
            relu4 = lrelu('relu4', bn4)
            
            conv5 = conv3d('conv5', relu4, 4, 4*4, 1, 1, False) 
            bn5 = BN('bn5', conv5, in_train)
            sigmoid = tf.nn.sigmoid(bn5)
            return sigmoid
        
    def train(self):
        path = self.save_path+'model.ckpt'
        fed = np.random.randn(self.batch_size,64,64,64,1)
        fed_test = np.zeros_like(fed)
        
        m,m_squ,fai_squ,log_fai_squ = self.build_VAE('VAE', self.x, True, self.len_vector, False) 
        
        noise = np.random.randn(self.batch_size,self.len_vector)
        inter_v = (-1*(m) + noise)/tf.sqrt(fai_squ)
        
        X_fake = self.build_G('G', inter_v, False, True)
        real_D_out = self.build_D('D', self.x, False, True)
        fake_D_out = self.build_D('D', X_fake, True, True)
        
        D_accu = accuracy(real_D_out,fake_D_out,self.batch_size)
        tf.summary.scalar('accuracy_of_D',D_accu)
#          collect vars
        D_vars = [var for var in tf.all_variables() if 'D' in var.name]
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]
        vae_vars = [var for var in tf.all_variables() if 'VAE' in var.name]
#       calculate loss:
#      vae_loss = 0.5 * (  ∑i=1:d  [μ(i)^2 + σ(i)^2 − logσ(i)^2 − 1]  )
        vae_loss = (tf.reduce_mean(tf.squared_difference(X_fake,self.x)) + tf.reduce_mean(0.5*(tf.reduce_sum(m_squ+fai_squ-log_fai_squ,1)-self.len_vector)))/self.len_vector                    
        G_loss = tf.reduce_mean((-1)*tf.log(fake_D_out)) + tf.reduce_mean(tf.squared_difference(X_fake,self.x))
        D_loss = tf.reduce_mean((-1)*tf.log(real_D_out) + (-1)*tf.log((-1)*fake_D_out+1))
        
        tf.summary.scalar('vae_loss',vae_loss)
        tf.summary.scalar('G_loss',G_loss)
        tf.summary.scalar('D_loss',D_loss)
        
        optim_vae = tf.train.AdadeltaOptimizer().minimize(vae_loss,var_list=vae_vars)
        optim_G = tf.train.AdadeltaOptimizer().minimize(G_loss,var_list=G_vars)
        optim_D = tf.train.AdadeltaOptimizer().minimize(D_loss,var_list=D_vars)
        
        Saver = tf.train.Saver(max_to_keep=15)
        merge = tf.summary.merge_all()
        
        Im = read_and_decode(self.record_dir)
        image_batch = tf.train.batch([Im], batch_size=self.batch_size, capacity=50000)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.summary.FileWriter(self.logdir,sess.graph)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
            for i in range(self.epoch):
                for j in range(self.len_data//self.batch_size):  
                    b_image = sess.run(image_batch)
                    fed_dict={self.x:b_image,self.x_test:self.zeros}
                    
                    accuracy_D,merge_run = sess.run([D_accu,merge],feed_dict=fed_dict)
                    
                    g_s = i*self.len_data+j*self.batch_size
                    graph.add_summary(merge_run, g_s)
                    if accuracy_D<0.8:
                        sess.run(optim_D,feed_dict=fed_dict)
                    _,Vae_loss = sess.run([optim_vae,vae_loss],feed_dict=fed_dict)
                    x_fake,g_loss,_ = sess.run([X_fake,G_loss,optim_G],feed_dict=fed_dict)
                    print(g_s)
                    if g_s%self.print_log_inter==0:
                        print('accuracy_D at epoch %d,sample %d,is: %f'%(i,j,accuracy_D))
                        print('Vae_loss at epoch %d,sample %d is: %f'%(i,j,Vae_loss))
                        print('g_loss at epoch %d,sample %d is: %f'%(i,j,g_loss))
                    if i%self.save_iner == 0:
                        Saver.save(sess,path,i)
                    if g_s%self.save_im_inter==0:
                        Save(x_fake,self.save_path,i,j)
                        print('save fake data at: ',i*self.len_data+j*self.batch_size)               
            coord.request_stop()
            coord.join(threads)

    def test(self,load_path,save_dir,scale,im1_dir,im2_dir):  #  build_VAE(self, name, x, in_train, len_vector, reuse):
        #  get ims for test:
        cube_len = 64
        fed1 = get_test_im(im1_dir, cube_len)   #  [1,64,64,64,1] 
        fed2 = get_test_im(im2_dir, cube_len)
        
        m,_,fai_squ,_ = self.build_VAE('VAE', self.x, False, self.len_vector, False)
        m1,_,fai_squ1,_ = self.build_VAE('VAE', self.x_test, False, self.len_vector, True)
        
        noise = np.random.randn(1,self.len_vector)
        inter_v0 = (-1*(m) + noise)/tf.sqrt(fai_squ)
        
        noise1 = np.random.randn(1,self.len_vector)
        inter_v1 = (-1*(m1) + noise1)/tf.sqrt(fai_squ1)
        
        inter_v = scale * inter_v0 + (1-scale) * inter_v1
        
        X_fake = self.build_G('G', inter_v, False, True)
        
        vars_restore = [var for var in tf.all_variables() if ('G' in var.name) or ('VAE' in var.name)]
        
        Saver = tf.train.Saver(var_list=vars_restore)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            Saver.restore(sess,load_path)
            dict_1 = {self.x:fed1,self.x_test:fed2}
            x_fake = sess.run(X_fake,feed_dict=dict_1)
            Save_test(x_fake,save_dir,1,1)
            print('test success')





def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, help='the number of epoches', default=800)
    parser.add_argument('--batch', type=int, help='batch size', default=50)
    parser.add_argument('--len_vector', type=int, help='lenth of latent vector', default=200)
    parser.add_argument('--len_x', type=int, help='cube size', default=64)
    parser.add_argument('--save_iner', type=int, help='Number of cycles to save the model', default=5)
    parser.add_argument('--len_data', type=int, help='Number of image data', default=10668)
    parser.add_argument('--save_im_inter', type=int, help='Number of cycles to save the image', default=50)
    parser.add_argument('--print_log_inter', type=int, help='Number of cycles of print loss', default=5)
    parser.add_argument('--save_path', type=str, help='Save model dir', default="E:/threeD/")
    parser.add_argument('--logdir', type=str, help='Path to the net_graph file', default="E:/threeD/")
    parser.add_argument('--record_dir', type=str, help='Path to the tf_record file', default="E:/zdelete/train.tfrecords")
    parser.add_argument('--phase', type=str, help='Train or test', default="train")
    
    parser.add_argument('--load_path', type=str, help='ckpt_dir', default="E:/threeD/model.ckpt-400")
    parser.add_argument('--save_dir', type=str, help='path to save test results', default="E:/testout/")
    parser.add_argument('--scale', type=float, help='the scaling factor of the latent variable smoothing, less than 1', default=0.2)
    parser.add_argument('--im1_dir', type=str, help='one mat file path to test', default="D:/3DShapeNets/volumetric_data/chair/30/train/chair_000000182_1.mat")
    parser.add_argument('--im2_dir', type=str, help='another mat file path to test', default="D:/3DShapeNets/volumetric_data/chair/30/train/chair_000000182_2.mat")
    
    return parser.parse_args(argv)

def main(args):
    model_vae = VAE_GAN(args.logdir, args.epoch, args.batch, args.len_vector, args.len_x, args.save_iner, args.save_path, args.len_data, args.save_im_inter, args.print_log_inter, args.record_dir)  
    if args.phase=='train':
        model_vae.train()
    else:
        model_vae.test(args.load_path,args.save_dir,args.scale,args.im1_dir,args.im2_dir)
    return True  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
