# vo_fundamental

/**-------------------视觉里程计：特征点法-----------------------**/

$ 特征点提取与匹配: ORB, SIFT, SURF

$ 通过2D-2D的匹配特征点估计相机运动：对极约束，本质矩阵，基础矩阵

$ 通过2D-2D的匹配估计一个点的空间位置：三角化

$ 3D-2D的PnP问题：线性解法(直接线性变换)和Bundle Adjustment解法 

$ 3D-3D的ICP问题：线性解法(SVD)和Bundle Adjustment解法


/** ----------------- 视觉里程计：直接法 ------------------------**/

$ Lucas-Kanade光流法跟踪特征点： 	假设1 灰度不变假设；
								假设2 某一个窗口内的像素具有相同的运动


#usage: ./test_pose_3d2d  data/1.png  data/2.png  data/1_depth.png
