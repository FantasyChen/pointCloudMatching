/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   CeresICP.h
 * Author: roy
 *
 * Created on December 3, 2016, 3:41 PM
 */

#ifndef CERESICP_H
#define CERESICP_H

#include "ceres/ceres.h"
#include "glog/logging.h"

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

#include <ceres/local_parameterization.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/types.h>
#include <ceres/rotation.h>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/transformation_validation_euclidean.h>
#include <pcl/common/transforms.h>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

typedef pcl::PointXYZI PointType;

Eigen::Isometry3d GPSICP(std::vector<double>& px, std::vector<double>& py, std::vector<double>& pz, std::vector<double>& gpsx, std::vector<double>& gpsy);

Eigen::Isometry3d MappingICP(pcl::PointCloud<pcl::PointXYZ>::Ptr refPointCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr curPointCloud, double outlierThreshold, double robustNormWidth);

Eigen::Isometry3d MappingICP(pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr refPointCloud, std::vector<bool>& inliers, double outlierThreshold);
void selectInlier(pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr refPointCloud, std::vector<bool>& inliers, double outlierThreshold);

Eigen::Isometry3d OdometryICP(pcl::PointCloud<pcl::PointXYZI>::Ptr lastPointCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, std::vector<bool>& inliers, double outlierThreshold);
void OdometryTransformToStart(pcl::PointCloud<pcl::PointXYZI>::Ptr curPointCloud, Eigen::Isometry3d fromCurToLast);
void OdometryTransformToEnd(pcl::PointCloud<pcl::PointXYZ>::Ptr curPointCloud, Eigen::Isometry3d fromCurToLast);

Eigen::Isometry3d pointToPointICP(std::vector<Eigen::Vector3d> &src, std::vector<Eigen::Vector3d> &dst);
Eigen::Isometry3d pointToPlaneICP(std::vector<Eigen::Vector3d> &src, std::vector<Eigen::Vector3d> &dst, std::vector<Eigen::Vector3d> &nor);
void isoToAngleAxis(const Eigen::Isometry3d& pose, double* cam);
Eigen::Isometry3d axisAngleToIso(const double* cam);

template <typename T>
ceres::MatrixAdapter<T, 1, 4> ColumnMajorAdapter4x3(T* pointer)
{
    return ceres::MatrixAdapter<T, 1, 4>(pointer);
}

class PointToPointCostFunction
        : public ceres::SizedCostFunction<3 /* number of residuals */,
                                          6 /* size of first parameter */>
{
public:
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    PointToPointCostFunction(const Eigen::Vector3d &dst, const Eigen::Vector3d &src) :
            p_dst(dst), p_src(src)
    {}
    virtual ~PointToPointCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        double p[3] = {p_src[0], p_src[1], p_src[2]};
        double camera[6] = {parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]};
        ceres::AngleAxisRotatePoint(camera, p, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = p[0] - p_dst[0];
        residuals[1] = p[1] - p_dst[1];
        residuals[2] = p[2] - p_dst[2];

        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = 0.0;
            jacobians[0][1] = p[2];
            jacobians[0][2] = -p[1];
            jacobians[0][3] = 1.0;
            jacobians[0][4] = 0.0;
            jacobians[0][5] = 0.0;

            jacobians[0][1 * 6 + 0] = -p[2];
            jacobians[0][1 * 6 + 1] = 0.0;
            jacobians[0][1 * 6 + 2] = p[0];
            jacobians[0][1 * 6 + 3] = 0.0;
            jacobians[0][1 * 6 + 4] = 1.0;
            jacobians[0][1 * 6 + 5] = 0.0;

            jacobians[0][2 * 6 + 0] = p[1];
            jacobians[0][2 * 6 + 1] = -p[0];
            jacobians[0][2 * 6 + 2] = 0.0;
            jacobians[0][2 * 6 + 3] = 0.0;
            jacobians[0][2 * 6 + 4] = 0.0;
            jacobians[0][2 * 6 + 5] = 1.0;
        }
        return true;
    }
};

class PointToPlaneCostFunction
        : public ceres::SizedCostFunction<1 /* number of residuals */,
                6 /* size of first parameter */>
{
public:
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    const Eigen::Vector3d& p_nor;
    PointToPlaneCostFunction(const Eigen::Vector3d &dst, const Eigen::Vector3d &src, const Eigen::Vector3d &nor) :
            p_dst(dst), p_src(src), p_nor(nor)
    {}
    virtual ~PointToPlaneCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        double p[3] = {p_src[0], p_src[1], p_src[2]};
        double camera[6] = {parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]};
        ceres::AngleAxisRotatePoint(camera, p, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = (p[0] - p_dst[0]) * p_nor[0] + \
                       (p[1] - p_dst[1]) * p_nor[1] + \
                       (p[2] - p_dst[2]) * p_nor[2];

        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = -p_nor[1]*p[2] + p_nor[2]*p[1];
            jacobians[0][1] = p_nor[0]*p[2] - p_nor[2]*p[0];
            jacobians[0][2] = -p_nor[0]*p[1] + p_nor[1]*p[0];
            jacobians[0][3] = p_nor[0];
            jacobians[0][4] = p_nor[1];
            jacobians[0][5] = p_nor[2];
        }
        return true;
    }
};

struct PointToPointError
{
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;

    PointToPointError(const Eigen::Vector3d &dst, const Eigen::Vector3d &src) :
    p_dst(dst), p_src(src)
    {
//        double xDif = dst[0] - dst[0];
//        double yDif = dst[1] - dst[1];
//        double zDif = dst[2] - dst[2];
//        double dif = (xDif*xDif + yDif*yDif + zDif*zDif);
//        std::cout << dif << "\n";
    }

    // Factory to hide the construction of the CostFunction object from the client code.

    static ceres::CostFunction* Create(const Eigen::Vector3d &observed, const Eigen::Vector3d &worldPoint)
    {
        return (new ceres::AutoDiffCostFunction<PointToPointError, 3, 6>(new PointToPointError(observed, worldPoint)));
    }

    template <typename T>
    bool operator()(const T * const camera, T* residuals) const
    {

        //            Eigen::Matrix<T,3,1> point;
        //            point << T(p_src[0]), T(p_src[1]), T(p_src[2]);

        T p[3] = {T(p_src[0]), T(p_src[1]), T(p_src[2])};
        ceres::AngleAxisRotatePoint(camera, p, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = p[0] - T(p_dst[0]);
        residuals[1] = p[1] - T(p_dst[1]);
        residuals[2] = p[2] - T(p_dst[2]);

//        T dif = (residuals[0]*residuals[0] + residuals[1]*residuals[1] + residuals[2]*residuals[2]);
//        std::cout << dif << "\n";
        return true;
    }
} ;

struct PointToPlaneError
{
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    const Eigen::Vector3d& p_nor;

    PointToPlaneError(const Eigen::Vector3d& dst, const Eigen::Vector3d& src, const Eigen::Vector3d& nor) :
    p_dst(dst), p_src(src), p_nor(nor)
    {
    }

    // Factory to hide the construction of the CostFunction object from the client code.

    static ceres::CostFunction* Create(const Eigen::Vector3d& observed, const Eigen::Vector3d& worldPoint, const Eigen::Vector3d& normal)
    {
        return (new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 6>(new PointToPlaneError(observed, worldPoint, normal)));
    }

    template <typename T>
    bool operator()(const T * const camera, T* residuals) const
    {

        T p[3] = {T(p_src[0]), T(p_src[1]), T(p_src[2])};
        ceres::AngleAxisRotatePoint(camera, p, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = (p[0] - T(p_dst[0])) * T(p_nor[0]) + \
                       (p[1] - T(p_dst[1])) * T(p_nor[1]) + \
                       (p[2] - T(p_dst[2])) * T(p_nor[2]);

        return true;
    }
};

struct PointToPointDistortionError
{
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    const double p_s; // scale

    PointToPointDistortionError(const Eigen::Vector3d &dst, const Eigen::Vector3d &src, const double s) :
            p_dst(dst), p_src(src), p_s(s)
    {
//        double xDif = dst[0] - dst[0];
//        double yDif = dst[1] - dst[1];
//        double zDif = dst[2] - dst[2];
//        double dif = (xDif*xDif + yDif*yDif + zDif*zDif);
//        std::cout << dif << "\n";
    }

    // Factory to hide the construction of the CostFunction object from the client code.

    static ceres::CostFunction* Create(const Eigen::Vector3d &observed, const Eigen::Vector3d &worldPoint, const double scale)
    {
        return (new ceres::AutoDiffCostFunction<PointToPointDistortionError, 3, 6>(new PointToPointDistortionError(observed, worldPoint, scale)));
    }

    template <typename T>
    bool operator()(const T * const camera, T* residuals) const
    {

        //            Eigen::Matrix<T,3,1> point;
        //            point << T(p_src[0]), T(p_src[1]), T(p_src[2]);

        T p[3] = {T(p_src[0]), T(p_src[1]), T(p_src[2])};
        T cameraWithScale[6] = {T(p_s)*camera[0], T(p_s)*camera[1], T(p_s)*camera[2], T(p_s)*camera[3], T(p_s)*camera[4], T(p_s)*camera[5]};
        ceres::AngleAxisRotatePoint(cameraWithScale, p, p);

        // camera[3,4,5] are the translation.
        p[0] += cameraWithScale[3];
        p[1] += cameraWithScale[4];
        p[2] += cameraWithScale[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = p[0] - T(p_dst[0]);
        residuals[1] = p[1] - T(p_dst[1]);
        residuals[2] = p[2] - T(p_dst[2]);

//        T dif = (residuals[0]*residuals[0] + residuals[1]*residuals[1] + residuals[2]*residuals[2]);
//        std::cout << dif << "\n";
        return true;
    }
} ;

struct PointToPlaneDistortionError
{
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    const Eigen::Vector3d& p_nor;
    const double p_s; // scale

    PointToPlaneDistortionError(const Eigen::Vector3d& dst, const Eigen::Vector3d& src, const Eigen::Vector3d& nor, const double s) :
            p_dst(dst), p_src(src), p_nor(nor), p_s(s)
    {
    }

    // Factory to hide the construction of the CostFunction object from the client code.

    static ceres::CostFunction* Create(const Eigen::Vector3d& observed, const Eigen::Vector3d& worldPoint, const Eigen::Vector3d& normal, const double scale)
    {
        return (new ceres::AutoDiffCostFunction<PointToPlaneDistortionError, 1, 6>(new PointToPlaneDistortionError(observed, worldPoint, normal, scale)));
    }

    template <typename T>
    bool operator()(const T * const camera, T* residuals) const
    {

        T p[3] = {T(p_src[0]), T(p_src[1]), T(p_src[2])};
        T cameraWithScale[6] = {T(p_s)*camera[0], T(p_s)*camera[1], T(p_s)*camera[2], T(p_s)*camera[3], T(p_s)*camera[4], T(p_s)*camera[5]};
        ceres::AngleAxisRotatePoint(cameraWithScale, p, p);

        // camera[3,4,5] are the translation.
        p[0] += cameraWithScale[3];
        p[1] += cameraWithScale[4];
        p[2] += cameraWithScale[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = (p[0] - T(p_dst[0])) * T(p_nor[0]) + \
                       (p[1] - T(p_dst[1])) * T(p_nor[1]) + \
                       (p[2] - T(p_dst[2])) * T(p_nor[2]);

        return true;
    }
};


#endif /* CERESICP_H */

