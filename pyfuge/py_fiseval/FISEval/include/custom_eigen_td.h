#ifndef CUSTOM_EG_TD
#define CUSTOM_EG_TD


#include <Eigen/Core>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXd;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatXi;


#define CLOCK_START clock_t t0 = clock();
#define CLOCK_GET double time_in_clock_ticks = clock() - t0;  double time_in_seconds = (double)time_in_clock_ticks / (double)CLOCKS_PER_SEC; cout << "time(ms): " << fixed << time_in_seconds * 1000 << endl;

#endif
