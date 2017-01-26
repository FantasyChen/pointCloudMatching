/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "CeresICP.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::HuberLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

//http://ceres-solver.org/faqs.html#solving

ceres::Solver::Options getOptions()
{
    // Set a few options
    ceres::Solver::Options options;
    //options.use_nonmonotonic_steps = true;
    options.preconditioner_type = ceres::IDENTITY;
    options.linear_solver_type = ceres::DENSE_QR;
//    options.max_num_iterations = 100;

    //    options.use_nonmonotonic_steps = true;
    //    options.preconditioner_type = ceres::SCHUR_JACOBI;
    //    options.linear_solver_type = ceres::DENSE_SCHUR;
    //    options.use_explicit_schur_complement=true;
    //    options.max_num_iterations = 100;

    return options;
}

/*
 * For small (a few hundred parameters) or dense problems use DENSE_QR.

For general sparse problems (i.e., the Jacobian matrix has a substantial number of zeros) use SPARSE_NORMAL_CHOLESKY. This requires that you have SuiteSparse or CXSparse installed.

For bundle adjustment problems with up to a hundred or so cameras, use DENSE_SCHUR.

For larger bundle adjustment problems with sparse Schur Complement/Reduced camera matrices use SPARSE_SCHUR. This requires that you build Ceres with support for SuiteSparse, CXSparse or Eigen’s sparse linear algebra libraries.

If you do not have access to these libraries for whatever reason, ITERATIVE_SCHUR with SCHUR_JACOBI is an excellent alternative.

For large bundle adjustment problems (a few thousand cameras or more) use the ITERATIVE_SCHUR solver. There are a number of preconditioner choices here. SCHUR_JACOBI offers an excellent balance of speed and accuracy. This is also the recommended option if you are solving medium sized problems for which DENSE_SCHUR is too slow but SuiteSparse is not available.

Note

If you are solving small to medium sized problems, consider setting Solver::Options::use_explicit_schur_complement to true, it can result in a substantial performance boost.
If you are not satisfied with SCHUR_JACOBI‘s performance try CLUSTER_JACOBI and CLUSTER_TRIDIAGONAL in that order. They require that you have SuiteSparse installed. Both of these preconditioners use a clustering algorithm. Use SINGLE_LINKAGE before CANONICAL_VIEWS.
 */

ceres::Solver::Options getOptionsMedium()
{
    // Set a few options
    ceres::Solver::Options options;
    std::cout << "linear algebra: " << options.sparse_linear_algebra_library_type << std::endl;
    std::cout << "linear solver: " << options.linear_solver_type << std::endl;
    //    options.use_nonmonotonic_steps = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    //    options.preconditioner_type = ceres::SPARSE_SCHUR;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    //If you are solving small to medium sized problems, consider setting Solver::Options::use_explicit_schur_complement to true, it can result in a substantial performance boost.
    //    options.use_explicit_schur_complement=true;
//    options.max_num_iterations = 50; // 50

    return options;
}

void solve(ceres::Problem &problem, bool smallProblem = true)
{
    ceres::Solver::Summary summary;
    ceres::Solve(smallProblem ? getOptions() : getOptionsMedium(), &problem, &summary);
//    if (!smallProblem) std::cout << "Final report:\n" << summary.FullReport();
//    std::cout << "Final report:\n" << summary.FullReport();
}

void isoToAngleAxis(const Eigen::Isometry3d& pose, double* cam)
{
    //    Matrix<const double,3,3> rot(pose.linear());
    //    cout<<"rotation : "<<pose.linear().data()<<endl;
    //    auto begin = pose.linear().data();
    RotationMatrixToAngleAxis(ColumnMajorAdapter4x3(pose.linear().data()), cam);
    Eigen::Vector3d t(pose.translation());
    cam[3] = t.x();
    cam[4] = t.y();
    cam[5] = t.z();
}

Eigen::Isometry3d axisAngleToIso(const double* cam)
{
    Eigen::Isometry3d poseFinal = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d rot;
    ceres::AngleAxisToRotationMatrix(cam, rot.data());
    poseFinal.linear() = rot;
    poseFinal.translation() = Eigen::Vector3d(cam[3], cam[4], cam[5]);
    return poseFinal; //.cast<float>();
}

Eigen::Isometry3d pointToPointICP(std::vector<Eigen::Vector3d> &src, std::vector<Eigen::Vector3d> &dst)
{
    double cam[6] = {0, 0, 0, 0, 0, 0};

    ceres::Problem problem;

    for (int i = 0; i < src.size(); ++i)
    {
        // first viewpoint : dstcloud, fixed
        // second viewpoint: srcCloud, moves
        ceres::CostFunction* cost_function = PointToPointError::Create(dst[i], src[i]);
        problem.AddResidualBlock(cost_function,
                //                NULL,
                new HuberLoss(0.1),
                cam);
    }

    solve(problem, false);
    return axisAngleToIso(cam);
}

Eigen::Isometry3d pointToPlaneICP(std::vector<Eigen::Vector3d> &src, std::vector<Eigen::Vector3d> &dst, std::vector<Eigen::Vector3d> &nor)
{
    double cam[6] = {0, 0, 0, 0, 0, 0};

    ceres::Problem problem;

    for (int i = 0; i < src.size(); ++i)
    {
        // first viewpoint : dstcloud, fixed
        // second viewpoint: srcCloud, moves
        // nor is normal of dst
        ceres::CostFunction* cost_function = PointToPlaneError::Create(dst[i], src[i], nor[i]);
        problem.AddResidualBlock(cost_function, 
                NULL, 
                cam);
    }

    solve(problem);

    return axisAngleToIso(cam);
}

void selectInlier(pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr refPointCloud, std::vector<bool>& inliers, double outlierThreshold)
{
    // Clear History
    inliers.clear();

    // KDTree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeRefPointCloud(new pcl::KdTreeFLANN<PointType>());
    kdtreeRefPointCloud->setInputCloud(refPointCloud);
    double maxCorrespondanceDist = outlierThreshold;

    // For search result
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // For each cur points
    int numOfInlier = 0;
    for (int i = 0; i < curPointCloud->points.size(); ++i)
    {
        PointType curPoint = curPointCloud->points[i];
        kdtreeRefPointCloud->nearestKSearch(curPoint, 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < maxCorrespondanceDist)
        {
            inliers.push_back(true);
            numOfInlier++;
        }
        else
        {
            inliers.push_back(false);
        }
    }

    std::cout << "Inliers: " << numOfInlier << "\n";
}

Eigen::Isometry3d MappingICP(pcl::PointCloud<pcl::PointXYZ>::Ptr refPointCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr curPointCloud, double outlierThreshold, double robustNormWidth)
{
    // Opt Input
    std::vector<Eigen::Vector3d> src;
    std::vector<Eigen::Vector3d> dst;
    std::vector<Eigen::Vector3d> nor;

    double cam[6] = {0, 0, 0, 0, 0, 0}; // Initial Cur Pos
    ceres::Problem problem; // Ceres Problem

    // KDTree
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeRefPointCloud(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    kdtreeRefPointCloud->setInputCloud(refPointCloud);
    double maxCorrespondanceDist = outlierThreshold; // 3.0m bewtween frame
    double robustPara = robustNormWidth; // 3.0m for two track alignment

    // For plane fitting
    cv::Mat matA0(5, 3, CV_64F, cv::Scalar::all(0));
    cv::Mat matB0(5, 1, CV_64F, cv::Scalar::all(-1));
    cv::Mat matX0(3, 1, CV_64F, cv::Scalar::all(0));

    // For search result
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // Outlier
    int numOfOutlier = 0;

    // For error metric selection
    int numOfPoints = curPointCloud->points.size();
    std::vector<bool> isPlanePointVec;
    int numOfPlanePoints = 0;

    // For each cur points
    for (int i = 0; i < curPointCloud->points.size(); ++i)
    {
        pcl::PointXYZ curPoint = curPointCloud->points[i];
        kdtreeRefPointCloud->nearestKSearch(curPoint, 5, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < maxCorrespondanceDist)
        {
            pcl::PointXYZ refPoint = refPointCloud->points[pointSearchInd[0]];

            for (int j = 0; j < 5; j++)
            {
                matA0.at<double>(j, 0) = refPointCloud->points[pointSearchInd[j]].x;
                matA0.at<double>(j, 1) = refPointCloud->points[pointSearchInd[j]].y;
                matA0.at<double>(j, 2) = refPointCloud->points[pointSearchInd[j]].z;
            }
            cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

            double pa = matX0.at<double>(0, 0);
            double pb = matX0.at<double>(1, 0);
            double pc = matX0.at<double>(2, 0);
            double pd = 1;

            double ps = sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                if (fabs(pa * refPointCloud->points[pointSearchInd[j]].x +
                         pb * refPointCloud->points[pointSearchInd[j]].y +
                         pc * refPointCloud->points[pointSearchInd[j]].z + pd) > 0.01)  //0.2
                {
                    planeValid = false;
                }
            }

            Eigen::Vector3d dstTmp, srcTmp, norTmp;
            dstTmp << refPoint.x, refPoint.y, refPoint.z;
            srcTmp << curPoint.x, curPoint.y, curPoint.z;
            norTmp << pa, pb, pc;
            dst.push_back(dstTmp);
            src.push_back(srcTmp);
            nor.push_back(norTmp);
            isPlanePointVec.push_back(planeValid);
        }
        else
        {
            numOfOutlier++;
        }
    }
//    std::cout << "Outliers: " << numOfOutlier << "\n";
    for (int i = 0; i < src.size(); ++i)
    {

        if(isPlanePointVec[i])
        {
            // first viewpoint : dstcloud, fixed
            // second viewpoint: srcCloud, moves
            ceres::CostFunction* cost_function = new PointToPlaneCostFunction(dst[i], src[i], nor[i]);
            problem.AddResidualBlock(cost_function,
                                     new HuberLoss(robustPara),
                                     cam);
            numOfPlanePoints++;
        } else
        {
            // first viewpoint : dstcloud, fixed
            // second viewpoint: srcCloud, moves
            ceres::CostFunction* cost_function = new PointToPointCostFunction(dst[i], src[i]);
            problem.AddResidualBlock(cost_function,
                                     new HuberLoss(robustPara),
                                     cam);
        }
    }
    std::cout << "numOfPlanePoints: " << numOfPlanePoints << "\n";

    solve(problem);
    return axisAngleToIso(cam);
}

Eigen::Isometry3d MappingICP(pcl::PointCloud<pcl::PointXYZI>::Ptr refPointCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, std::vector<bool>& inliers, double outlierThreshold)
{
    // Clear History
    inliers.clear();

    // Opt Input
    std::vector<Eigen::Vector3d> src;
    std::vector<Eigen::Vector3d> dst;
    std::vector<Eigen::Vector3d> nor;

    double cam[6] = {0, 0, 0, 0, 0, 0}; // Initial Cur Pos
    ceres::Problem problem; // Ceres Problem

    // KDTree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeRefPointCloud(new pcl::KdTreeFLANN<PointType>());
    kdtreeRefPointCloud->setInputCloud(refPointCloud);
    double maxCorrespondanceDist = outlierThreshold; // 3.0m bewtween frame
    double robustPara = 0.5;
//    double groundThreshold = -1.0;

    // For plane fitting
    cv::Mat matA0(5, 3, CV_64F, cv::Scalar::all(0));
    cv::Mat matB0(5, 1, CV_64F, cv::Scalar::all(-1));
    cv::Mat matX0(3, 1, CV_64F, cv::Scalar::all(0));

    // For search result
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // Outlier
    int numOfOutlier = 0;

    // For error metric selection
    int numOfPoints = curPointCloud->points.size();
    std::vector<bool> isPlanePointVec;

    // For each cur points
    for (int i = 0; i < curPointCloud->points.size(); ++i)
    {
        PointType curPoint = curPointCloud->points[i];
        kdtreeRefPointCloud->nearestKSearch(curPoint, 5, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] < maxCorrespondanceDist)
        {
            PointType refPoint = refPointCloud->points[pointSearchInd[0]];
            int closestPointScan = int(refPoint.intensity);

//            if (abs(int(curPoint.intensity) - closestPointScan) > 2.5)
//            {
//                numOfOutlier++;
//                inliers.push_back(false);
//                continue;
//            }
//            if (curPoint.z < groundThreshold)
//            {
//                numOfOutlier++;
//                inliers.push_back(false);
//                continue;
//            }

            for (int j = 0; j < 5; j++)
            {
                matA0.at<double>(j, 0) = refPointCloud->points[pointSearchInd[j]].x;
                matA0.at<double>(j, 1) = refPointCloud->points[pointSearchInd[j]].y;
                matA0.at<double>(j, 2) = refPointCloud->points[pointSearchInd[j]].z;
            }
            cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

            double pa = matX0.at<double>(0, 0);
            double pb = matX0.at<double>(1, 0);
            double pc = matX0.at<double>(2, 0);
            double pd = 1;

            double ps = sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                if (fabs(pa * refPointCloud->points[pointSearchInd[j]].x +
                         pb * refPointCloud->points[pointSearchInd[j]].y +
                         pc * refPointCloud->points[pointSearchInd[j]].z + pd) > 0.01)  //0.2
                {
                    planeValid = false;
                }
            }

            Eigen::Vector3d dstTmp, srcTmp, norTmp;
            dstTmp << refPoint.x, refPoint.y, refPoint.z;
            srcTmp << curPoint.x, curPoint.y, curPoint.z;
            norTmp << pa, pb, pc;
            dst.push_back(dstTmp);
            src.push_back(srcTmp);
            nor.push_back(norTmp);
            isPlanePointVec.push_back(planeValid);
            inliers.push_back(true);
        }
        else
        {
            numOfOutlier++;
            inliers.push_back(false);
        }
    }

    std::cout << "Outliers: " << numOfOutlier << "\n";

    for (int i = 0; i < src.size(); ++i)
    {
        if(isPlanePointVec[i])
        {
            // first viewpoint : dstcloud, fixed
            // second viewpoint: srcCloud, moves
            // nor is normal of dst
            ceres::CostFunction* cost_function = PointToPlaneError::Create(dst[i], src[i], nor[i]);
            problem.AddResidualBlock(cost_function,
                                         new HuberLoss(robustPara),
                                         cam);
        }
        else
        {
            // first viewpoint : dstcloud, fixed
            // second viewpoint: srcCloud, moves
            ceres::CostFunction* cost_function = PointToPointError::Create(dst[i], src[i]);
            problem.AddResidualBlock(cost_function,
                                     new HuberLoss(robustPara),
                                     cam);
        }

    }

    solve(problem);
    return axisAngleToIso(cam);
}

Eigen::Isometry3d OdometryICP(pcl::PointCloud<pcl::PointXYZI>::Ptr lastPointCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, std::vector<bool>& inliers, double outlierThreshold)
{
    // Clear History
    inliers.clear();

    // Opt Input
    std::vector<Eigen::Vector3d> src;
    std::vector<Eigen::Vector3d> dst;
    std::vector<Eigen::Vector3d> nor;
    std::vector<double> scale;

    double cam[6] = {0, 0, 0, 0, 0, 0}; // Initial Cur Pos
    ceres::Problem problem; // Ceres Problem

    // KDTree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeRefPointCloud(new pcl::KdTreeFLANN<PointType>());
    kdtreeRefPointCloud->setInputCloud(lastPointCloud);
    double maxCorrespondanceDist = outlierThreshold;
    double robustPara = 0.5;

    // Outlier
    int numOfOutlier = 0;

    // Feature Points
    int numOfPointFeature = 0;

    // For error metric selection
    int numOfPoints = curPointCloud->points.size();
    std::vector<bool> isPlanePointVec;

    // For search result
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // Tmp point
    PointType pointSel, tripod1, tripod2, tripod3;

    // For each cur points
    for (int i = 0; i < curPointCloud->points.size(); ++i)
    {
        PointType curPoint = curPointCloud->points[i];
        pointSel = curPointCloud->points[i];
        kdtreeRefPointCloud->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
        PointType refPoint;

        if (pointSearchSqDis[0] < maxCorrespondanceDist)
        {
            refPoint = lastPointCloud->points[pointSearchInd[0]];
            closestPointInd = pointSearchInd[0];
            int closestPointScan = int(lastPointCloud->points[closestPointInd].intensity);
            int surfPointsFlatNum = lastPointCloud->points.size();

            float pointSqDis, minPointSqDis2 = maxCorrespondanceDist, minPointSqDis3 = maxCorrespondanceDist;
            for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++)
            {
                if (int(lastPointCloud->points[j].intensity) > closestPointScan + 2.5)
                {
                    numOfOutlier++;
                    inliers.push_back(false);
                    break;
                }

                pointSqDis = (lastPointCloud->points[j].x - pointSel.x) *
                             (lastPointCloud->points[j].x - pointSel.x) +
                             (lastPointCloud->points[j].y - pointSel.y) *
                             (lastPointCloud->points[j].y - pointSel.y) +
                             (lastPointCloud->points[j].z - pointSel.z) *
                             (lastPointCloud->points[j].z - pointSel.z);

                if (int(lastPointCloud->points[j].intensity) <= closestPointScan)
                {
                    if (pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }
                else
                {
                    if (pointSqDis < minPointSqDis3)
                    {
                        minPointSqDis3 = pointSqDis;
                        minPointInd3 = j;
                    }
                }
            }
            for (int j = closestPointInd - 1; j >= 0; j--)
            {
                if (int(lastPointCloud->points[j].intensity) < closestPointScan - 2.5)
                {
                    break;
                }
                pointSqDis = (lastPointCloud->points[j].x - pointSel.x) *
                             (lastPointCloud->points[j].x - pointSel.x) +
                             (lastPointCloud->points[j].y - pointSel.y) *
                             (lastPointCloud->points[j].y - pointSel.y) +
                             (lastPointCloud->points[j].z - pointSel.z) *
                             (lastPointCloud->points[j].z - pointSel.z);

                if (int(lastPointCloud->points[j].intensity) >= closestPointScan)
                {
                    if (pointSqDis < minPointSqDis2)
                    {
                        minPointSqDis2 = pointSqDis;
                        minPointInd2 = j;
                    }
                }
                else
                {
                    if (pointSqDis < minPointSqDis3)
                    {
                        minPointSqDis3 = pointSqDis;
                        minPointInd3 = j;
                    }
                }
            }

            bool planeValid = true;
            float pa = 0.0;
            float pb = 0.0;
            float pc = 0.0;
            float pd = 0.0;
            if (minPointInd2 >= 0 && minPointInd3 >= 0)
            {
                tripod1 = lastPointCloud->points[closestPointInd];
                tripod2 = lastPointCloud->points[minPointInd2];
                tripod3 = lastPointCloud->points[minPointInd3];

                pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z)
                     - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x)
                     - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y)
                     - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
                pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;

                if (fabs(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z + pd) > 0.1)  //0.2
                {
                    planeValid = false;
                }
                if (fabs(pa * tripod2.x + pb * tripod2.y + pc * tripod2.z + pd) > 0.1)  //0.2
                {
                    planeValid = false;
                }
                if (fabs(pa * tripod3.x + pb * tripod3.y + pc * tripod3.z + pd) > 0.1)  //0.2
                {
                    planeValid = false;
                }
            }
            else
            {
                planeValid = false;
            }

            Eigen::Vector3d dstTmp, srcTmp, norTmp;
            dstTmp << refPoint.x, refPoint.y, refPoint.z;
            srcTmp << curPoint.x, curPoint.y, curPoint.z;
            norTmp << pa, pb, pc;
            dst.push_back(dstTmp);
            src.push_back(srcTmp);
            nor.push_back(norTmp);
            isPlanePointVec.push_back(planeValid);
            scale.push_back(double(10 * (curPoint.intensity - int(curPoint.intensity))));
            inliers.push_back(true);

        } else
        {
            numOfOutlier++;
            inliers.push_back(false);
        }
    }
    std::cout << "Outliers: " << numOfOutlier << "\n";

    for (int i = 0; i < src.size(); ++i)
    {
        if(isPlanePointVec[i])
        {
            // first viewpoint : dstcloud, fixed
            // second viewpoint: srcCloud, moves
            // nor is normal of dst
//            ceres::CostFunction* cost_function = PointToPlaneError::Create(dst[i], src[i], nor[i]);
            ceres::CostFunction* cost_function = PointToPlaneDistortionError::Create(dst[i], src[i], nor[i], scale[i]);
            problem.AddResidualBlock(cost_function,
                                     new HuberLoss(robustPara),
                                     cam);
        }
        else
        {
            // first viewpoint : dstcloud, fixed
            // second viewpoint: srcCloud, moves
//            ceres::CostFunction* cost_function = PointToPointError::Create(dst[i], src[i]);
            ceres::CostFunction* cost_function = PointToPointDistortionError::Create(dst[i], src[i], scale[i]);
            problem.AddResidualBlock(cost_function,
                                     new HuberLoss(robustPara),
                                     cam);
            numOfPointFeature++;
        }

    }
    std::cout << "Point Features: " << numOfPointFeature << "\n";

    solve(problem);
    return axisAngleToIso(cam);
}

void OdometryTransformToStart(pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, Eigen::Isometry3d fromCurToLast)
{
    double camera[6];
    isoToAngleAxis(fromCurToLast, camera);
    for(int i = 0; i < curPointCloud->points.size(); i++)
    {
        double p[3];
        p[0] = curPointCloud->points[i].x;
        p[1] = curPointCloud->points[i].y;
        p[2] = curPointCloud->points[i].z;
        // get scale
        double s = 10 * (curPointCloud->points[i].intensity - int(curPointCloud->points[i].intensity));
//        s = 1.0;
        // rotate by scale
        double cameraWithScale[6] = {s*camera[0], camera[1], s*camera[2], s*camera[3], s*camera[4], s*camera[5]};
        ceres::AngleAxisRotatePoint(cameraWithScale, p, p);
        // camera[3,4,5] are the translation.
        p[0] += cameraWithScale[3];
        p[1] += cameraWithScale[4];
        p[2] += cameraWithScale[5];
        // assign back
        curPointCloud->points[i].x = p[0];
        curPointCloud->points[i].y = p[1];
        curPointCloud->points[i].z = p[2];
    }
}


void OdometryTransformToEnd(pcl::PointCloud<pcl::PointXYZ>::Ptr curPointCloud, Eigen::Isometry3d fromCurToLast)
{
    double camera[6];
    isoToAngleAxis(fromCurToLast, camera);
    for(int i = 0; i < curPointCloud->points.size(); i++)
    {
        double p[3];
        p[0] = curPointCloud->points[i].x;
        p[1] = curPointCloud->points[i].y;
        p[2] = curPointCloud->points[i].z;

        double s = 1.0;
        // rotate by scale
        double cameraWithScale[6] = {s*camera[0], camera[1], s*camera[2], s*camera[3], s*camera[4], s*camera[5]};
        ceres::AngleAxisRotatePoint(cameraWithScale, p, p);
        // camera[3,4,5] are the translation.
        p[0] += cameraWithScale[3];
        p[1] += cameraWithScale[4];
        p[2] += cameraWithScale[5];
        // assign back
        curPointCloud->points[i].x = p[0];
        curPointCloud->points[i].y = p[1];
        curPointCloud->points[i].z = p[2];
    }
}


