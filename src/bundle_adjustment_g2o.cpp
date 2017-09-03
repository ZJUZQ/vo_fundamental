#include "../include/bundle_adjustment_g2o.hpp"

void bundle_adjustment_g2o(const std::vector<cv::Point3f> pts_3d,
						   const std::vector<cv::Point2f> pts_2d,
						   const cv::Mat& K,
						   cv::Mat R, 
						   cv::Mat t){

	// 初始化g2o
	/**
		// traits to summarize the properties of the fixed size optimization problem
  		template <int _PoseDim, int _LandmarkDim>
  		struct BlockSolverTraits{
			typedef Eigen::Matrix<double, PoseDim, PoseDim, Eigen::ColMajor> PoseMatrixType;
			typedef Eigen::Matrix<double, LandmarkDim, LandmarkDim, Eigen::ColMajor> LandmarkMatrixType;
    		typedef Eigen::Matrix<double, PoseDim, LandmarkDim, Eigen::ColMajor> PoseLandmarkMatrixType;
  		}


  		// Implementation of a solver operating on the blocks of the Hessian: H*△x = g
		template <typename Traits>
		class BlockSolver: public BlockSolverBase {}
	*/
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block; // // pose 维度为 6, landmark 维度为 3
	/**
		// linear solver which uses CSparse
		template <typename MatrixType>
		class LinearSolverCSparse : public LinearSolverCCS< MatrixType >{}

		PoseMatrixType 在 BlockSolverTraits 中定义:
		typedef Eigen::Matrix<double, PoseDim, PoseDim, Eigen::ColMajor> PoseMatrixType;
	*/
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse< Block::PoseMatrixType >(); // 线性方程求解器
    // allocate a block solver ontop of the underlying linear solver
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    // construct the Levenberg algorithm, which will use the given Solver for solving the linearized system.
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg (solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm( solver );


    // vertex
    /**
		* Templatized BaseVertex
		* D  : minimal dimension of the vertex, e.g., 3 for rotation in 3D
		* T  : internal type to represent the estimate, e.g., Quaternion for rotation in 3D
		template <int D, typename T>
		class BaseVertex : public OptimizableGraph::Vertex {}


    	// SE3 Vertex parameterized internally with a transformation matrix and externally with its exponential map
		class G2O_TYPES_SBA_API VertexSE3Expmap : public BaseVertex<6, SE3Quat>{}
    	SE3Quat: 这个类内部使用了四元素(Quaternion) + 位移向量来存储位姿 ，但同时支持李代数上的运算，
    			 比如对数映射(log)和李代数上增量(update函数)等；
    */
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    		 R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    		 R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat, Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))) );
    optimizer.addVertex(pose);

    int index = 1;
    for(const cv::Point3f p : pts_3d){ 	// landmarks
    	/**
    		// Point vertex, XYZ
 			class G2O_TYPES_SBA_API VertexSBAPointXYZ : public BaseVertex<3, Vector3D> {}
    	*/
    	g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
    	point->setId( index++ );
    	point->setEstimate( Eigen::Vector3d(p.x, p.y, p.z) );
    	/**
			true => this node should be marginalized out during the optimization
			g2o 中必须设置 marg;

			在求解 H*△x = g  时，因为矩阵H具有稀疏结构，在视觉SLAM中通常采用Schur消元，亦称为Marginalization(边缘化)。
			具体地，先消去 △x_points，求解 △x_cameras; 然后利用求得的 △x_camras进一步求解 △x_points.
    	 */
    	point->setMarginalized( true );
    	optimizer.addVertex(point);
    }

    // parameter: camera intrinsics
    /**
    	class G2O_TYPES_SBA_API CameraParameters : public g2o::Parameter
		{
		  public:
		    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		    CameraParameters();

		    CameraParameters(double focal_length,
		        			const Vector2D & principle_point,
		        			double baseline)
		      : focal_length(focal_length),
		      	principle_point(principle_point),
		      	baseline(baseline){}    			// baseline: 双目相机两个光圈中心的距离
		}
    */
    g2o::CameraParameters* camera  = new g2o::CameraParameters(
    	K.at<double>(0, 0),
    	Eigen::Vector2d(K.at<double>(0, 2), K.at<double>(1, 2)),
    	0);
    camera->setId(0);
    optimizer.addParameter(camera);


    // edges
    index = 1;
    for(const cv::Point2d p : pts_2d){
    	/**
    		class G2O_TYPES_SBA_API EdgeProjectXYZ2UV : public  BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>{
			  public:
			    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

			    EdgeProjectXYZ2UV();

			    bool read(std::istream& is);

			    bool write(std::ostream& os) const;

			    void computeError()  {
			      const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
			      const VertexSBAPointXYZ* v2 = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
			      const CameraParameters * cam
			        = static_cast<const CameraParameters *>(parameter(0));
			      Vector2D obs(_measurement);
			      _error = obs - cam->cam_map( v1->estimate().map( v2->estimate() )) ;
			    }

			    virtual void linearizeOplus();

			    CameraParameters * _cam;
			};

			Vector3D SE3Quat::map(const Vector3D & xyz) const
			{
				return _r*xyz + _t;
			}

			EdgeProjectXYZ2UV连接了前面的两个顶点(VertexSBAPointXYZ ， VertexSE3Expmap)，它的观测值为2维(空间点的
			像素坐标)。它的误差计算函数表达了投影方程的误差计算方法，即z - h(T, P)
	
    	*/
    	g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
    	edge->setId(index);
    	/**
    		// set the ith vertex on the hyper-edge to the pointer supplied
          	void setVertex(size_t i, Vertex* v)
    	*/
    	edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index))); // dynamic_cast 在运行时执行转换，验证转换的有效性
    	edge->setVertex(1, pose);
    	edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
    	/**
    		bool OptimizableGraph::Edge::setParameterId(int argNum, int paramId){
			    if ((int)_parameters.size()<=argNum)
			      	return false;
			    if (argNum<0)
			      	return false;
			    *_parameters[argNum] = 0;
			    _parameterIds[argNum] = paramId;
			    return true;
			}
    	*/
    	edge->setParameterId(0, 0);
    	edge->setInformation(Eigen::Matrix2d::Identity());
    	optimizer.addEdge(edge);
    	index++;
    }

    clock_t t1 = clock();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100); // int optimize(int iterations, bool online = false);
    clock_t t2 = clock();
    cout << "g2o optimization consts time: " << (t2 - t1) * 1000.0 / CLOCKS_PER_SEC << " ms\n";
    cout << endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;

    Eigen::Quaterniond quat = pose->estimate().rotation();
    R_mat = quat.toRotationMatrix();
    for(int i = 0; i < 3; i++)
    	for(int j = 0; j < 3; j++){
    		R.at<double>(i, j) = R_mat(i, j);
    	}

    Eigen::Vector3d t_estimated = pose->estimate().translation();
    for(int i = 0; i < 3; i++){
    	t.at<double>(i, 0) = t_estimated(i, 0);
    }
}