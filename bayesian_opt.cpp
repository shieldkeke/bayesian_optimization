#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

typedef struct
{
	std::vector<double> mu, cov;
}GP;

Eigen::MatrixXd Gaussian_Kernal(Eigen::MatrixXd x1, Eigen::MatrixXd x2){
	//hyperparam
	double sigema_f = 0.2, l = 0.5;
	Eigen::MatrixXd dist(x1.rows(), x2.rows());
	for (int i = 0; i < x1.rows(); i++){
		for (int j = 0; j < x2.rows(); j++){
			dist(i,j) =x1.row(i).dot(x1.row(i)) + x2.row(j).dot(x2.row(j)) - 2 * x1.row(i).dot(x2.row(j));
		}
	}
	return pow(sigema_f, 2) * (-0.5 / pow(l, 2) * dist).array().exp();
}

GP Gaussian_Process_Regression(std::vector<Eigen::VectorXd> train_X, std::vector<double> train_Y,std::vector<Eigen::VectorXd> predict_X)
{
	
	GP ret;
	int n_train = train_X.size(), n_pre = predict_X.size(), m = train_X[0].rows(); //data numbers and dimension
	Eigen::MatrixXd train_x(n_train, m), predict_x(n_pre, m), kfy(n_train, n_pre), kff(n_train, n_train), kyy(n_pre, n_pre), kff_inv(n_train, n_train);
	Eigen::VectorXd train_y(n_train);
    // std::cout<<n_train<<" "<<n_pre<<" "<<m<<std::endl;

	//errors
	if (train_X.size() == 0 || predict_X.size() == 0){
		std::cout<<"size = 0!"<<std::endl;
		return ret;
	}
	if (train_X.size() != train_Y.size()){
		std::cout<<"not the same length of train X and Y"<<std::endl;
		return ret;
	}
	if (train_X[0].rows() != predict_X[0].rows()){
		std::cout<<"not the same form of data"<<std::endl;
		return ret;
	} 

	
	//transfer from vector to matrix
	for (int i=0; i<train_X.size(); i++){
		train_x.row(i) = train_X[i];
		train_y(i) = train_Y[i];
	}
	for (int i=0; i<predict_X.size(); i++){
		predict_x.row(i) = predict_X[i];
	}
	kfy = Gaussian_Kernal(train_x, predict_x);
	kyy = Gaussian_Kernal(predict_x, predict_x);
	kff = Gaussian_Kernal(train_x, train_x);
	kff_inv = (kff + 1e-8 * Eigen::MatrixXd::Identity(n_train, n_train)).inverse() ;   
	Eigen::MatrixXd mu(n_pre, 1), cov(n_pre, n_pre);
	mu = kfy.transpose()*kff_inv*train_y;
	cov = kyy - kfy.transpose()*kff_inv*kfy;
	for (int i=0; i<n_pre; i++){
		ret.mu.push_back(mu(i,0));
		ret.cov.push_back(cov(i,i));
	}
	return ret;
}

int main()
{
	Eigen::VectorXd vec(2); // length of the vector
	
	std::vector<Eigen::VectorXd> train_x = {};
	std::vector<Eigen::VectorXd> test_x = {};
	std::vector<double> train_y = {};
	
	vec << 1583,1;
	train_x.push_back(vec);
	train_y.push_back(2);
	vec << 1584,1;
	train_x.push_back(vec);
	train_y.push_back(23);

	vec << 1585,1;
	test_x.push_back(vec);
	GP result;
	result= Gaussian_Process_Regression(train_x, train_y, test_x);
	std::cout<<result.mu[0]<<" "<<result.cov[0]<<std::endl;
}
