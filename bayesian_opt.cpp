#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

template<typename T>

double Gaussian_process_Regression(std::vector<T> pre_position, std::vector<T> pre_gradient,double predict_position)
{
	std::vector<T> position = {}, gradient = {};
	position = pre_position;
    gradient = pre_gradient;
	Eigen::MatrixXd  K(position.size(), position.size());
	Eigen::MatrixXd  K_s(1, position.size());
	Eigen::MatrixXd  Y(position.size(), 1);
	Eigen::MatrixXd  result(1, 1);
	//hyperparam
	double sigema_f = 1.27, l = 1,sigema_n = 0.3;
	double K_ss = pow(sigema_f, 2) + pow(sigema_n, 2);
    
	for (int i = 0; i < position.size(); i++)
	{
		for (int j = 0; j < position.size(); j++)
		{	
			double rectify;
			rectify = (i == j) ? pow(sigema_n, 2) : 0;
			K(i, j) = pow(sigema_f, 2) * exp(-pow(position[i] - position[j], 2) / (2 * pow(l, 2))) + rectify;
		}
	}
	for (int i = 0; i < position.size(); i++)
	{
		K_s(0,i)= pow(sigema_f, 2) * exp(-pow(predict_position - position[i], 2) / (2 * pow(l, 2))) ;
	}
	for (int i = 0; i < gradient.size(); i++)
	{
		Y(i, 0) = gradient[i];
	}
	result = K_s * K.inverse() * Y;
	return result(0,0);
}

int main()
{
    std::vector<int> pre_position = { 1583,1584,1585,1586,1587,1588,1589 };
    std::vector<int> pre_gradient = { 2,23,114,246,245,114,23 };
    double predict_position=1583;
    double result;
    for (float i = predict_position; i < 1589; i += 0.5)
    {
        result= Gaussian_process_Regression(pre_position, pre_gradient, i);
        std::cout << result << "  ";
    }
    std::cout << std::endl;
   

}
