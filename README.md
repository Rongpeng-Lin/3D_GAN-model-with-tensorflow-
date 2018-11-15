# 3D_GAN-model-with-tensorflow-
A tensorflow implementation of VAE_3D_GAN
## Instructions<br>
&#8195;1.Implement the tensorflow on the VAE_GAN structure of the original text. The training picture is actually the coordinate value. In order to speed up the training process, the picture is first made into tfrecord format.<br>
&#8195;2.About VAE:<br>&#8195;&#8195;In the original text, the vector encoded by the encoder is used directly. The potential vector in the VAE theory obeys the normal distribution, but the VAE outputs the predicted value of the mean and the variance instead of the latent vector. In VAE, P(Z|X) is a normal distribution. If the latent variable Z is directly output, when the batch number is small, the mean and variance calculated by Z cannot be used to estimate the similarity between the normal distribution and the normal distribution. Even if the sampling is performed according to the normal distribution, the mean and variance of the sampling result do not satisfy 0 and 1, so the predicted mean and variance are used here, and Z is not directly predicted.<br>
&#8195;3.Interpolation calculation during testing:<br>&#8195;&#8195;When testing the smoothness of the model relative to the latent variable, given a smoothing coefficient, similar to finding a random ratio between the two points, the interpolation between the two vectors is obtained: v = alpha*v0+(1-alpha) *v1.<br>
## Use it<br>
&#8195;1.Generate tfrecord data:<br>
&#8195;&#8195;python&#8195;E:\zdelete\create_tf_record.py&#8195;--name="chair"&#8195;--if_train=True&#8195;--cube_len=64&#8195;--max_epoch=500&#8195;--LOCAL_PATH="D:/3DShapeNets/volumetric_data/"&#8195;--TFRECORD_DIR="E:/zdelete"&#8195;--split_name="train"<br>
&#8195;2.Train:<br>
&#8195;&#8195;python&#8195;E:\zdelete\3d_GAN.py&#8195;--epoch=500&#8195;--batch=50&#8195;--len_vector=200&#8195;--len_x=64&#8195;--save_iner=10&#8195;--len_data=10668&#8195;--save_im_inter=10&#8195;--print_log_inter=15&#8195;--save_path="E:/threeD/"&#8195;--logdir="E:/threeD/"&#8195;--record_dir="E:/zdelete/train.tfrecords"&#8195;--phase="train"<br>
&#8195;3.Test:<br>
&#8195;&#8195;python&#8195;E:\zdelete\3d_GAN.py&#8195;--epoch=500&#8195;--batch=50&#8195;--len_vector=200&#8195;--len_x=64&#8195;--save_iner=10&#8195;--len_data=10668&#8195;--save_im_inter=10&#8195;--print_log_inter=15&#8195;--save_path="E:/threeD/"&#8195;--logdir="E:/threeD/"&#8195;--record_dir="E:/zdelete/train.tfrecords"&#8195;--phase="test"&#8195;--load_path="E:/threeD/model.ckpt-400"&#8195;--save_dir="E:/testout/"&#8195;--scale=0.3&#8195;--im1_dir="D:/3DShapeNets/volumetric_data/chair/30/train/chair_000000182_1.mat"&#8195;--im2_dir="D:/3DShapeNets/volumetric_data/chair/30/train/chair_000000182_2.mat"

