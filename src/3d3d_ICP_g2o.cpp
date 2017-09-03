#include "../include/3d3d_ICP_g2o.hpp"

// vertex
/** g2o::VertexSE3Expmap **/


// edge
// RGB-D相机每次可以观测到路标点的三维位置，从而产生一个3D观测数据。这是一个一元边，只关联到一个节点(camera pose)。
// g2o/sba中没有提供这样的边，需要我们自己实现。
class EdgeProjectXYZ2RGBD_PoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	EdgeProjectXYZ2RGBD_PoseOnly(const Eigen::Vector3d& point) : _point(point) {}

	virtual void computeError(){
		const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
		_error = _measurement - pose->estimate().map(_point);
	}

	virtual void linearizeOplus(){
		g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
		g2o::SE3Quat T(pose->estimate());
		Eigen::Vector3d xyz_trans = T.map(_point);
		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double z = xyz_trans[2];

		_jacobianOplusXi(0, 0) = 0;
		_jacobianOplusXi(0, 1) = -z;
		_jacobianOplusXi(0, 2) = y;
		_jacobianOplusXi(0, 3) = -1;
		_jacobianOplusXi(0, 4) = 0;
		_jacobianOplusXi(0, 5) = 0;

		_jacobianOplusXi(1, 0) = z;
		_jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;
        
        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
	}

	bool read(istream& in) {}
	bool write(ostream& out) const {}
protected:
	Eigen::Vector3d _point;
};


void _3d3d_ICP_g2o(	const std::vector<cv::Point3d>& pts1,
			 		const std::vector<cv::Point3d>& pts2,
			 		cv::Mat& R,
			 		cv::Mat& t){

	// 初始化g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block; // pose维度为 6, landmark 维度为 3
	// 线性方程求解器
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen< Block::PoseMatrixType >();
	// 矩阵块求解器
	Block* solver_ptr = new Block(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
	pose->setId(0);
	// ICP问题存在唯一解或者无穷多解。在唯一解的情况下，只要找到极小值解，那么这个极小值就是全局最优值，这也意味着ICP求解
	// 可以任意选定初始值；
	pose->setEstimate( g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0)) );
	optimizer.addVertex(pose);

	// edges
	// pts1 = R * pts2 + t
	int index = 1;
	std::vector<EdgeProjectXYZ2RGBD_PoseOnly*> edges;
	for(size_t i = 0; i < pts1.size(); i++){
		EdgeProjectXYZ2RGBD_PoseOnly* edge = new EdgeProjectXYZ2RGBD_PoseOnly( Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
		edge->setId(index);
		edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*>(pose));
		edge->setMeasurement( Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z) );
		edge->setInformation( Eigen::Matrix3d::Identity() * 1e3 );
		optimizer.addEdge(edge);
		index++;
		edges.push_back(edge);
	}

	clock_t t1 = clock();
	optimizer.setVerbose(true);
	optimizer.initializeOptimization();
	optimizer.optimize(50);
	clock_t t2 = clock();
	cout << "g2o optimization costs time: " << (t2 - t1) * 1000.0 / CLOCKS_PER_SEC << " ms\n";

	Eigen::Quaterniond quat = pose->estimate().rotation();
	Eigen::Matrix3d R_estimated = quat.toRotationMatrix();
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			R.at<double>(i, j) = R_estimated(i, j);

	Eigen::Vector3d t_estimated = pose->estimate().translation();
	for(int i = 0; i < 3; i++)
		t.at<double>(i, 0) = t_estimated(i, 0);
}