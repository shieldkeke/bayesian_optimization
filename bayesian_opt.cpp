#include <iostream>
#include "bayesian_opt.h"
#include <cmath>

bayesianOpt::bayesianOpt(){
	train_X = {};
	train_Y = {};
	srand(time(0));
	num_sample = 100;
}

Eigen::MatrixXd bayesianOpt::Gaussian_Kernal(Eigen::MatrixXd x1, Eigen::MatrixXd x2){
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

GP bayesianOpt::Gaussian_Process_Regression(std::vector<Eigen::VectorXd> predict_X)
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

void bayesianOpt::clear(){
	train_X = {};
	train_Y = {};
}

void bayesianOpt::test(){
	//test for gausian process
	Eigen::VectorXd vec(2); // length of the vector

	std::vector<Eigen::VectorXd> test_x = {};
	
	vec << 1583,1;
	train_X.push_back(vec);
	train_Y.push_back(2);
	vec << 1584,1;
	train_X.push_back(vec);
	train_Y.push_back(23);

	vec << 1585,1;
	test_x.push_back(vec);
	GP result;
	result= Gaussian_Process_Regression(test_x);
	std::cout<<result.mu[0]<<" "<<result.cov[0]<<std::endl;
}

Eigen::VectorXd bayesianOpt::acq_max(){
	double max_y = -100000;
	int max_idx = 0;
	std::vector<double> result = UCB(1.5);
	for (int i=0; i<num_sample; i++){
		if (result[i] > max_y){
			max_y = result[i];
			max_idx = i;
		}
	}
	return sample_x[max_idx];
}

std::vector<double> bayesianOpt::UCB(double k){
	GP result = Gaussian_Process_Regression(sample_x);
	std::vector<double> ret;
	for (int i=0; i<num_sample; i++){
		ret.push_back(result.mu[i] + k * result.cov[i]);
	}
	return ret;
}

void bayesianOpt::set_bound(std::vector<double> low, std::vector<double> high){
	if (low.size() != high.size()){
		std::cout<<"error bound format(not the same)"<<std::endl;
	}
	low_bound = low;
	high_bound = high;

	Eigen::VectorXd vec(low.size());
	for (int i=0; i<num_sample; i++){ //number of samples
		for (int j=0; j<low.size(); j++){
			vec(j) = low[j] + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(high[j]-low[j]))); //low - high random
		}
		sample_x.push_back(vec);
	}
}

optResult bayesianOpt::result(){
	double max_y = -100000;
	Eigen::VectorXd max_x;
	optResult ret;
	for (int i=0; i<train_X.size(); i++){
		if (train_Y[i] > max_y){
			max_y = train_Y[i];
			max_x = train_X[i];
		}
	}
	ret.x = max_x;
	ret.y = max_y;
	return ret;
}

Eigen::VectorXd bayesianOpt::update(Eigen::VectorXd x, double y){
	train_X.push_back(x);
	train_Y.push_back(y);
	Eigen::VectorXd suggestion = acq_max();
	return suggestion;
}

int main()
{
	bayesianOpt b;
	// b.test();
	std::vector<double> low = {0} , high = {2};
	Eigen::VectorXd vec(1);
	b.set_bound(low, high);
	vec << 0;
	b.update(vec, 0);
	vec << 0.5;
	Eigen::VectorXd suggestion = b.update(vec, sin(0.5));
	for (int i=0; i<20; i++){
		suggestion = b.update(suggestion, sin(suggestion(0)));
	}
	std::cout<<b.result().x<<" "<<b.result().y<<std::endl;

}
