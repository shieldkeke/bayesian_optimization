#ifndef __BAYESIAN_OPT__ 
#define __BAYESIAN_OPT__ 
#include <vector>
#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <ctime>
typedef struct
{
	std::vector<double> mu, cov;
}GP;

typedef struct{
    Eigen::VectorXd x;
    double y;
}optResult;

class bayesianOpt{

    public:
        void test();
        void clear();
        void set_bound(std::vector<double> low, std::vector<double> high);
        Eigen::VectorXd update(Eigen::VectorXd x, double y);
        bayesianOpt();
        optResult result();
    private:
        Eigen::MatrixXd Gaussian_Kernal(Eigen::MatrixXd x1, Eigen::MatrixXd x2);
        GP Gaussian_Process_Regression(std::vector<Eigen::VectorXd> predict_X);
        Eigen::VectorXd acq_max();
        std::vector<double> UCB(double k);

        std::vector<double> low_bound;
        std::vector<double> high_bound;
        std::vector<Eigen::VectorXd> train_X;
        std::vector<Eigen::VectorXd> sample_x;
        std::vector<double> train_Y;
        int num_sample;
};
#endif