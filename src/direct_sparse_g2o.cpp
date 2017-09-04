#include "../include/direct_sparse_g2o.hpp"

// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
class Edge_SE3ProjectDirect : public BaseUnaryEdge<1, double, VertexSE3Expmap> {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Edge_SE3ProjectDirect(){}
	Edge_SE3ProjectDirect(Eigen::Vector3d point, Eigen::Matrix3d K, cv::Mat* p_img)
		: _p_world(point), _K(K), _p_img(p_img)
	{}

	virtual void computeError(){
		const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
		Eigen::Vector3d x_local = v->estimate().map(_p_world);
		double x = _K(0, 0) * x_local[0] / x_local[2] + _K(0, 2);
		double y = _K(1, 1) * x_local[1] / x_local[2] + _K(1, 2);
		// check x, y is in the image
		if(x < 4 || x + 4 > _p_img->cols || y < 4 || y + 4 > _p_img->rows){
			_error(0, 0) = 0.0;
			this->setLevel(1); //! sets the level of the edge
		}
		else{
			// _measurement = P1(I1), the grayscale in first image
			_error(0, 0) = getPixelValue(x, y) - _measurement;
		}
	}

	// plus in manifold
	virtual void linearizeOplus(){
		if(level() == 1){
			_jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
			return;
		}
		VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*>(_vertices[0]);
		Eigen::Vector3d xyz_trans = vtx->estimate().map( _p_world );

		double x = xyz_trans[0];
		double y = xyz_trans[1];
		double invz = 1.0 / xyz_trans[2];
		double invz_2 = invz * invz;

		double u = _K(0, 0) * x * invz + _K(0, 2);
		double v = _K(1, 1) * y * invz + _K(1, 2);

		// jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
        jacobian_uv_ksai(0, 0) = - _K(0, 0) * x * y *invz_2;
        jacobian_uv_ksai(0, 1) = _K(0, 0) * (1 + x * x * invz_2);
        jacobian_uv_ksai(0, 2) = - _K(0, 0) * y *invz;
        jacobian_uv_ksai(0, 3) = _K(0, 0) * invz;
        jacobian_uv_ksai(0, 4) = 0;
        jacobian_uv_ksai(0, 5) = - _K(0, 0) * x * invz_2;

        jacobian_uv_ksai(1, 0) = - _K(1, 1) * (1 + y * y * invz_2);
        jacobian_uv_ksai(1, 1) = _K(1, 1) * x * y * invz_2;
        jacobian_uv_ksai(1, 2) = _K(1, 1) * x * invz;
        jacobian_uv_ksai(1, 3) = 0;
        jacobian_uv_ksai(1, 4) = _K(1, 1) * invz;
        jacobian_uv_ksai(1, 5) = - _K(1, 1) * y * invz_2;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;
        jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2.0;
        jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2.0;

        _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
	}

	// dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {}
    virtual bool write ( std::ostream& out ) const {}

protected:
	// get a grayscale value from the second image (bilinear interpolation)
	inline double getPixelValue(double x, double y){
		double xx = x - floor(x);
		double yy = y - floor(y);
		//uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];
		/**    addr(M_ij) = M.data + M.step[0]*i + M.step[1]*j    **/
		uchar* data = _p_img->data + _p_img->step[0] * ( int(y) ) + _p_img->step[1] * ( int(x) );
		return double(
				(1 - xx) * (1 - yy) * data[0] +
				xx * (1 - yy) * data[1] +
				(1 - xx) * yy * data[ _p_img->step[0] ] +
				xx * yy * data[ _p_img->step[0] + 1 ]
				);
	}


public:
	Eigen::Vector3d _p_world; 	// 3D point in world frame
	Eigen::Matrix3d _K;			// Camera intrinsics
	cv::Mat* _p_img = NULL; 		// point to the second image

};



bool direct_sparse_g2o(const std::vector<Measurement>& measurements, 
					   cv::Mat& gray, 
					   Eigen::Matrix3d& K, 
					   Eigen::Isometry3d& T_cw){

	/** 优化变量为一个相机位姿，因此需要一个位姿顶点：VertexSE3Expmap */

	/** 误差项为单个像素的光度误差。由于整个优化过程中I1(p1)保持不变，我们可以把它当成一个
		固定的预设值，然后调整相机位姿，使I2(p2)接近这个值。于是，这种边只连接一个顶点，为
		一元边。*/


	// 初始化g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 1> > DirectBlock; // / 求解的向量是6＊1的
	DirectBlock::LinearSolverType* linearSolver = new g2o::LinearSolverDense< DirectBlock::PoseMatrixType >();
	DirectBlock* solver_ptr = new DirectBlock(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr ); 
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);
	optimizer.setVerbose(true);

	// add vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
	pose->setEstimate( g2o::SE3Quat(T_cw.rotation(), T_cw.translation()) );
	pose->setId(0);
	optimizer.addVertex(pose);

	// add edge
	int id = 1;
	for(Measurement m : measurements){
		Edge_SE3ProjectDirect* edge = new Edge_SE3ProjectDirect(m._p_world, K, &gray);
		edge->setMeasurement(m._grayscale);
		edge->setVertex(0, pose);
		edge->setInformation( Eigen::Matrix<double, 1, 1>::Identity() );
		edge->setId(id++);
		optimizer.addEdge(edge);
	}

	cout << "edges in graph: " << optimizer.edges().size() << endl;
	optimizer.initializeOptimization();
	optimizer.optimize(100);
	T_cw = pose->estimate();
}