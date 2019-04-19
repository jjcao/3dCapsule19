import argparse

parser = argparse.ArgumentParser(description = '3dCapsule 2019')
parser.add_argument('--data_root', type = str,default = '/data/spe_database_old',help = 'it is a shared parameter')
parser.add_argument('--outf', type=str, default='../../data/spe_out',  help='output folder')# /Users/jjcao/data/spe_data_train_11348

parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='./checkpoints/model.pkl', help='optional, pre_trained model')

parser.add_argument('--env', type=str, default="3dCapsule", help='visdom environment')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)

###################################
##  shared parameters


#parser.add_argument('--data_root', type = str,default = '../../data',help = 'it is a shared parameter') # for my macbook

# default
parser.add_argument('--model', type=str, default = './model/SPENet_pointnetmini_PointGenCon_108_0.043558_s0.039577_p0.002866_3d0.000499_decoded0.000263_j0.00035_r0.000010.pkl',  help='saved/pre_trained model')
#parser.add_argument('--model', type=str, default = './model/SPENetSiam.pkl',  help='saved/pre_trained model')
parser.add_argument('--modelBeta', type=str, default = './model/SPENetSiam.pkl',  help='saved/pre_trained model')
parser.add_argument('--modelPose', type=str, default = './model/SPENetSiam.pkl',  help='saved/pre_trained model')
# for centering 
#parser.add_argument('--model', type=str, default = './model/SPENetSiam_pointnetmini_PointGenCon_92_0.119_s0.116_p0.001_3d0.0004_decoded0.0005_j0.0001-centerBinput-stnOutput-3dCodedTemplate.pkl',  help='saved/pre_trained model')
#parser.add_argument('--modelBeta', type=str, default = './model/SPENetBeta_pointnetmini_None_344_0.118_s0.118_3d0.0001_j0.00001-centerInput-stnOutput.pkl',  help='saved/pre_trained model')
#parser.add_argument('--modelPose', type=str, default = './model/SPENetPose_pointnetmini_None_138_0.002_p0.001_3d0.0004_j0.0001-centerInput-stnOutput.pkl',  help='saved/pre_trained model')
#parser.add_argument('--modelGen', type=str, default = './model/SPENetSiam_pointnetmini_PointGenCon_84_0.109_s0.106_p0.001_3d0.0004_decoded0.0002_j0.0001-centerBinput-stnOutput.pkl',  help='saved/pre_trained model')
# for not centering
#parser.add_argument('--model', type=str, default = './model/SPENetBeta_pointnetmini_None_118_0.025_s0.025_3d0.00001_j0.00001_centerB_notCenterInput.pkl',  help='saved/pre_trained model')
# parser.add_argument('--modelBeta', type=str, default = './model/SPENetBeta_pointnetmini_None_118_0.025_s0.025_3d0.00001_j0.00001.pkl',  help='saved/pre_trained model')
# parser.add_argument('--modelPose', type=str, default = './model/SPENetSiam.pkl',  help='saved/pre_trained model')
parser.add_argument('--center_input', default = True, type = bool, help = 'center input in dataset')
parser.add_argument('--trans_smpl_generated', default = 'stn', type = str, help = 'None, stn, center')

# should >= number of GPU*2. e.g. 72 batch in 3 GPU leads to 24 batch in each GPU. # If the batches number on each GPU == 1, nn.BatchNorm1d fails.
# large batch size => better convergence. # 16 for 6-9G gpu with decoder, 24 for ? without decoder
parser.add_argument('--batch_size', type=int, default=128, help='input batch size') #72=24*3=18*4, 96=24*4
parser.add_argument('--start_epoch', type=int, default = 0, help='')
parser.add_argument('--no_epoch', type=int, default = 121, help='number of epochs to train for')#121
parser.add_argument('--lr',type = float,default = 0.001,help = 'learning rate')#0.001 
parser.add_argument('--step_lr', type = float, default = 10, help = 'encoder learning rate.')
parser.add_argument('--step_save', type = float, default = 2, help = 'step for saving model.')

parser.add_argument('--shape_ratio',type = float, default = 40.0 ,help = 'weight of shape loss') #40 for GMOF loss function
parser.add_argument('--pose_ratio',type = float, default = 400.0, help = 'weight of pose')# 400 for GMOF loss function
#default: 400. 20 is enough for making sure that predicated pose parameter does not contain global rotation
parser.add_argument('--threeD_ratio',type = float, default = 400.0, help = 'weight of vertices decoded by smpl')
#default: 200. 20 is enough for making sure that predicated pose parameter does not contain global rotation
parser.add_argument('--j3d_ratio',type = float, default = 200.0, help = 'weight of 3d key points decoded by smpl')
parser.add_argument('--decoded_ratio',type = float, default = 400.0, help = 'weight of vertices decoded by decoder')#400,
#parser.add_argument('--with_chamfer',default = False, type = bool,help = 'use chamfer loss')
#parser.add_argument('--chamfer_ratio',type = float, default = 0.0, help = 'weight of 3d chamfer distance')#50

###################################
##  parameters for training
parser.add_argument('--network', type = str,default = 'SPENet',help = 'SPENet, SPENetSiam, SPENetBeta, SPENetPose')
parser.add_argument('--encoder', type = str,default = 'pointnetmini',help = 'pointnetmini, pointnet or pointnet2') 
parser.add_argument('--decoder', type = str,default = 'PointGenCon',help = 'None, PointGenCon or pointnet2 or dispNet?')

parser.add_argument('--with_stn', default = 'STN3dTR', type = str, help = 'use STN3dR, STN3dRQuad, STN3dTR, or None in encoder')
parser.add_argument('--with_stn_feat', default = False, type = bool, help = 'use stn feature transform in encoder or not')
parser.add_argument('--pervertex_weight', type = str, default = 'None', help = 'None or ')#./data/pervertex_weight_sdf.npz
parser.add_argument('--point_count', type=int, default=2400, help='the count of vertices in the input pointcloud for training')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--momentum',type = float,default = 0.9,help = 'momentum')
# weight decay = 0.0001, it is very important for training the network using adam
parser.add_argument('--wd', type = float, default = 0.0001, help = 'encoder weight decay rate.')
parser.add_argument('--ls', type = str, default = 'L2', help = 'loss function: L2, L1, or GMOF (from less robust to more robust).')

parser.add_argument('--vis', type=str, default= 'spe', help='visdom environment, use visualization in training')
parser.add_argument('--smpl_mean_theta_path', type = str, default = './data/neutral_smpl_mean_params.h5', help = 'the path for mean smpl theta value')
parser.add_argument('--smpl_model',type = str,
                    default = './data/neutral_smpl_with_cocoplus_reg.txt',
                    help = 'smpl model path')

########
# for reconstruction, correspondence 
parser.add_argument('--HR', type=int, default=0, help='Use high Resolution template for better precision in the nearest neighbor step ?')
parser.add_argument('--nepoch', type=int, default=3000, help='number of epochs to train for during the regression step')
parser.add_argument('--inputA', type=str, default =  "/data/MPI-FAUST/test/scans/test_scan_021.ply",  help='your path to mesh 0')
parser.add_argument('--inputB', type=str, default =  "/data/MPI-FAUST/test/scans/test_scan_011.ply",  help='your path to mesh 1')
# parser.add_argument('--inputA', type=str, default =  "data/example_0.ply",  help='your path to mesh 0')
# parser.add_argument('--inputB', type=str, default =  "data/example_1.ply",  help='your path to mesh 1')
#parser.add_argument('--num_points', type=int, default = 6890,  help='number of points fed to poitnet') # point_count
#parser.add_argument('--num_angles', type=int, default = 300,  help='number of angle in the search of optimal reconstruction. Set to 1, if you mesh are already facing the cannonical direction as in data/example_1.ply')
parser.add_argument('--clean', type=int, default=1, help='if 1, remove points that dont belong to any edges')
parser.add_argument('--scale', type=int, default=1, help='if 1, scale input mesh to have same volume as the template')
parser.add_argument('--project_on_target', type=int, default=0, help='if 1, projects predicted correspondences point on target mesh')

########
# for data generation
parser.add_argument('--human_count', type = int, default = 30000, help = 'the count of male/femal in generated database')
parser.add_argument('--sample_count', type = int, default = 0, help = 'the count of samples of a SMPL template mesh') # 2500
parser.add_argument('--op', type = str, default = 'generate', help = 'generate, distill, unify')
parser.add_argument('--gender', type = str, default = 'm', help = 'm for male, f for female, b for both')
parser.add_argument('--data_type', type = str, default = 'w', help = 'w for whole, f for front view, fb for front & back view')
# spe_dataset_train_specifiedPose
parser.add_argument('--database_train', type = str, default = 'spe_dataset_train', help = 'name')
parser.add_argument('--database_val', type = str, default = 'spe_dataset_val', help = 'name')


args = parser.parse_args()