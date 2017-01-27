#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <time.h>

//#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


// Eigens
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <unordered_map>


#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/transformation_validation_euclidean.h>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/transformation_validation_euclidean.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/search/kdtree.h>

#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp>
#include <pcl-1.8/pcl/common/eigen.h>
#include <pcl-1.8/pcl/registration/transformation_validation.h>


#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <limits>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>



#include "CeresICP.h"
#include "utils.h"
// -----------------------
// -----Some Class--------
// -----------------------

class FeatureCloud
{
public:
    // A bit of shorthand
    typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
    typedef pcl::PointCloud<pcl::Normal> SurfaceNormals;
    typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
    typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

    FeatureCloud () :
            search_method_xyz_ (new SearchMethod),
            normal_radius_ (0.02f),
            feature_radius_ (0.02f)
    {}

    ~FeatureCloud () {}

    // Process the given cloud
    void
    setInputCloud (PointCloud::Ptr xyz)
    {
        xyz_ = xyz->makeShared();
        processInput ();
    }

    // Load and process the cloud in the given PCD file
    void
    loadInputCloud (const std::string &pcd_file)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr curCloud(new pcl::PointCloud<pcl::PointXYZ>);;
        pcl::PCLPointCloud2 cloud_blob;
        pcl::io::loadPCDFile (pcd_file.c_str(), cloud_blob);
        pcl::fromPCLPointCloud2(cloud_blob, *curCloud);
        xyz_ = curCloud->makeShared();
        processInput ();
    }

    // Get a pointer to the cloud 3D points
    PointCloud::Ptr
    getPointCloud () const
    {
        return (xyz_);
    }

    // Get a pointer to the cloud of 3D surface normals
    SurfaceNormals::Ptr
    getSurfaceNormals () const
    {
        return (normals_);
    }

    // Get a pointer to the cloud of feature descriptors
    LocalFeatures::Ptr
    getLocalFeatures () const
    {
        return (features_);
    }

protected:
    // Compute the surface normals and local features
    void
    processInput ()
    {
        computeSurfaceNormals ();
        computeLocalFeatures ();
    }

    // Compute the surface normals
    void
    computeSurfaceNormals ()
    {
        normals_ = SurfaceNormals::Ptr (new SurfaceNormals);

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
        cout << xyz_->points.size() <<endl;
        norm_est.setInputCloud(xyz_);
        norm_est.setSearchMethod(search_method_xyz_);
        norm_est.setRadiusSearch(normal_radius_);
        norm_est.compute (*normals_);
    }

    // Compute the local feature descriptors
    void
    computeLocalFeatures ()
    {
        features_ = LocalFeatures::Ptr (new LocalFeatures);

        pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
        fpfh_est.setInputCloud (xyz_);
        fpfh_est.setInputNormals (normals_);
        fpfh_est.setSearchMethod (search_method_xyz_);
        fpfh_est.setRadiusSearch (feature_radius_);
        fpfh_est.compute (*features_);
    }

private:
    // Point cloud data
    PointCloud::Ptr xyz_;
    SurfaceNormals::Ptr normals_;
    LocalFeatures::Ptr features_;
    SearchMethod::Ptr search_method_xyz_;

    // Parameters
    float normal_radius_;
    float feature_radius_;
};

class TemplateAlignment
{
public:

    // A struct for storing alignment results
    struct Result
    {
        float fitness_score;
        Eigen::Matrix4f final_transformation;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    TemplateAlignment () :
            min_sample_distance_ (0.05f),
            max_correspondence_distance_ (0.01f*0.01f),
            nr_iterations_ (500)
    {
        // Intialize the parameters in the Sample Consensus Intial Alignment (SAC-IA) algorithm
        sac_ia_.setMinSampleDistance (min_sample_distance_);
        sac_ia_.setMaxCorrespondenceDistance (max_correspondence_distance_);
        sac_ia_.setMaximumIterations (nr_iterations_);
    }

    ~TemplateAlignment () {}

    // Set the given cloud as the target to which the templates will be aligned
    void
    setTargetCloud (FeatureCloud &target_cloud)
    {
        target_ = target_cloud;
        sac_ia_.setInputTarget (target_cloud.getPointCloud ());
        sac_ia_.setTargetFeatures (target_cloud.getLocalFeatures ());
    }

    // Add the given cloud to the list of template clouds
    void
    addTemplateCloud (FeatureCloud &template_cloud)
    {
        templates_.push_back (template_cloud);
    }

    // Align the given template cloud to the target specified by setTargetCloud ()
    void
    align (FeatureCloud &template_cloud, TemplateAlignment::Result &result)
    {
        sac_ia_.setInputSource(template_cloud.getPointCloud ());
        sac_ia_.setSourceFeatures (template_cloud.getLocalFeatures ());

        pcl::PointCloud<pcl::PointXYZ> registration_output;
        sac_ia_.align (registration_output);

        result.fitness_score = (float) sac_ia_.getFitnessScore (max_correspondence_distance_);
        result.final_transformation = sac_ia_.getFinalTransformation ();
    }

    // Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
    void
    alignAll (std::vector<TemplateAlignment::Result, Eigen::aligned_allocator<Result> > &results)
    {
        results.resize (templates_.size ());
        for (size_t i = 0; i < templates_.size (); ++i)
        {
            align (templates_[i], results[i]);
        }
    }

    // Align all of template clouds to the target cloud to find the one with best alignment score
    int
    findBestAlignment (TemplateAlignment::Result &result)
    {
        // Align all of the templates to the target cloud
        std::vector<Result, Eigen::aligned_allocator<Result> > results;
        alignAll (results);

        // Find the template with the best (lowest) fitness score
        float lowest_score = std::numeric_limits<float>::infinity ();
        int best_template = 0;
        for (size_t i = 0; i < results.size (); ++i)
        {
            const Result &r = results[i];
            if (r.fitness_score < lowest_score)
            {
                lowest_score = r.fitness_score;
                best_template = (int) i;
            }
        }

        // Output the best alignment
        result = results[best_template];
        return (best_template);
    }

private:
    // A list of template clouds and the target to which they will be aligned
    std::vector<FeatureCloud> templates_;
    FeatureCloud target_;

    // The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
    float min_sample_distance_;
    float max_correspondence_distance_;
    int nr_iterations_;
};




float maxLengthOfBoundingBox = 10.0;
float minLengthOfBoundingBox = 1.0;


class BoundingBox
{
public:
    BoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudInput)
    {
        maxX = -FLT_MAX;
        maxY = -FLT_MAX;
        maxZ = -FLT_MAX;
        minX = FLT_MAX;
        minY = FLT_MAX;
        minZ = FLT_MAX;
        calcuBoundingBox(laserCloudInput);
//        std::cout << "point: " << laserCloudInput->points.size() << " X:[" << minX << "," << maxX << "]; " << "Y:[" << minY << "," << maxY << "]; " << "Z:[" << minZ << "," << maxZ << "];\n";

        // calculate maxLength
        isvalid = true;
        maxLength = maxX - minX;
        if(maxLength < (maxY - minY))
            maxLength = maxY - minY;
        if(maxLength < (maxZ - minZ))
            maxLength = maxZ - minZ;
        if(maxLength > maxLengthOfBoundingBox)
        {
            isvalid = false;
        }

        // calculate minLength
        minLength = maxX - minX;
        if(minLength < (maxY - minY))
            maxLength = maxY - minY;
        if(minLength < (maxZ - minZ))
            maxLength = maxZ - minZ;
        if(minLength < minLengthOfBoundingBox)
        {
            isvalid = false;
        }
    }
    void calcuBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudInput)
    {
        for (int i = 0; i < laserCloudInput->points.size(); ++i)
        {
            float x = laserCloudInput->points[i].x;
            float y = laserCloudInput->points[i].y;
            float z = laserCloudInput->points[i].z;
            if(x > maxX) maxX = x;
            if(y > maxY) maxY = y;
            if(z > maxZ) maxZ = z;
            if(x < minX) minX = x;
            if(y < minY) minY = y;
            if(z < minZ) minZ = z;
        }
    }
    bool checkIfContainedBy(BoundingBox& largerOne)
    {
        if(largerOne.minX > this->minX) return false;
        if(largerOne.minY > this->minY) return false;
        if(largerOne.minZ > this->minZ) return false;
        if(largerOne.maxX < this->maxX) return false;
        if(largerOne.maxY < this->maxY) return false;
        if(largerOne.maxZ < this->maxZ) return false;
        return true;
    }

    bool isvalid;
    float maxLength;
    float minLength;

private:
    float maxX;
    float maxY;
    float maxZ;
    float minX;
    float minY;
    float minZ;
};

// -----------------------
// -----Some Function-----
// -----------------------

typedef pcl::PointXYZI PointType;

boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

bool checkWithPriorInfo(pcl::PointXYZ& curPoint)
{
    if(curPoint.y < -1.5)  // ground plane
        return false;

    if(fabs(curPoint.z) > 15.0) // Inside the road
        return false;

    return true;

}

bool checkIfGroundPlaneCoeffi(pcl::ModelCoefficients::Ptr coefficients)
{
    double a = coefficients->values[0];
    double b = coefficients->values[1];
    double c = coefficients->values[2];
    double d = coefficients->values[3];
    double yFraction = fabs(b)/(a*a + b*b + c*c);


    if(yFraction > 0.9)
    {
        if(d < -1.8)
        {
            for (int k = 0; k < 4; ++k)
            {
                std::cout << coefficients->values[k] << " ";
            }
            cout << ", " << yFraction << " " << d << endl;
            return true;
        }
    }
    return false;
}

void findVehicles(std::string srcPointCloudPathString, std::string refPointCloudPathString)
{
    // Path
    std::string resultPrefixString = "../result/cloud_cluster_";
    std::string sparseResultPrefixString = "../result_sparse/cloud_cluster_";
//    std::string srcPointCloudPathString = "../1479967858051000.pcd";
//    std::string refPointCloudPathString = "../1479967858160000.pcd";

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudSrc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudRef(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PCLPointCloud2 cloud_blob;

    // Load pointcloud
    std::cout << "Loading..." << std::endl;
    pcl::io::loadPCDFile(srcPointCloudPathString.c_str(), cloud_blob);
    pcl::fromPCLPointCloud2(cloud_blob, *laserCloudSrc); //* convert from pcl/PCLPointCloud2 to pcl::PointCloud<T>
    pcl::io::loadPCDFile(refPointCloudPathString.c_str(), cloud_blob);
    pcl::fromPCLPointCloud2(cloud_blob, *laserCloudRef); //* convert from pcl/PCLPointCloud2 to pcl::PointCloud<T>

    // Create the segmentation object for the planar model
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;

    // Set all the parameters for plane fitting
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.01);

    /*---------- 1.Remove ground plane from full pointcloud ----------*/
    int nr_points = (int) laserCloudSrc->points.size ();
    int failureCounter = 0;
    while (laserCloudSrc->points.size () > 0)
    {
        // Segment the largest planar component from the remaining cloud1479967826.82_FullRes.pcd
        seg.setInputCloud(laserCloudSrc);
        seg.segment(*inliers, *coefficients);
//        if (inliers->indices.size () < 200)
        if (!checkIfGroundPlaneCoeffi(coefficients) || inliers->indices.size() < 20)
        {
            std::cout << "Could not estimate more ground plane mdoels for the given dataset." << std::endl;
            failureCounter++;
            if(failureCounter > 10) break;
            continue;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(laserCloudSrc);
        extract.setIndices(inliers);
        extract.setNegative(false);

        // Get the points associated with the planar surface
        extract.filter(*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points."
                  << std::endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative(true);
        extract.filter(*cloud_f);
        *laserCloudSrc = *cloud_f;
    }
    cout << laserCloudRef->points.size() << endl;

    /*---------- 2.Get outliers from full pointcloud ----------*/
    // KDTree
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeRefPointCloud(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    kdtreeRefPointCloud->setInputCloud(laserCloudRef);
    double outlierThreshold = 1.0; // 1.0m
    // For search result
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    std::vector<bool> outlierMarker(laserCloudSrc->points.size(), false);
    for (int i = 0; i < laserCloudSrc->points.size(); ++i)
    {
        pcl::PointXYZ curPoint = laserCloudSrc->points[i];
        kdtreeRefPointCloud->nearestKSearch(curPoint, 1, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[0] > outlierThreshold)
        {
            if(checkWithPriorInfo(curPoint))
                outlierMarker[i] = true;
        }
    }
    // Get outlier
    for (int i = 0; i < laserCloudSrc->width; i++)
    {
        pcl::PointXYZRGB point;

        point.x = laserCloudSrc->points[i].x;
        point.y = laserCloudSrc->points[i].y;
        point.z = laserCloudSrc->points[i].z;
        point.b = 0;
        point.g = 255;
        point.r = 0;

        if(outlierMarker[i])
        {
            point_cloud_ptr->push_back(point);
            cloud->push_back(laserCloudSrc->points[i]);
        }
    }
    for (int i = 0; i < laserCloudRef->width; i++)
    {
        pcl::PointXYZRGB point;

        point.x = laserCloudRef->points[i].x;
        point.y = laserCloudRef->points[i].y;
        point.z = laserCloudRef->points[i].z;
        point.b = 0;
        point.g = 0;
        point.r = 255;

        point_cloud_ptr->push_back(point);
    }
    cout << cloud->points.size() << endl;
    /*---------- 3.Get pointcloud groups from outliers ----------*/
    // Remove planes first
    std::cout << "Extracting from outliers...\n";
    int i=0;
    nr_points = (int) cloud->points.size ();
    while (cloud->points.size () > 0.1 * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () < 100)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud (cloud);
        extract.setIndices (inliers);
        extract.setNegative (false);

        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_f);
        *cloud = *cloud_f;
    }

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (2.0); // 2.0m
    ec.setMinClusterSize (10);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);

    std::vector<BoundingBox> vehicleBoundingBoxVec;
    int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        BoundingBox tmp(cloud_cluster);
        if(tmp.isvalid)
        {
            vehicleBoundingBoxVec.push_back(tmp);

            std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points. "
                      << tmp.maxLength << std::endl;
            std::stringstream ss;
            ss << sparseResultPrefixString << j << ".pcd";
            writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false); //*
            j++;
        }
    }

    /*---------- 4.Get pointcloud groups from full pointcloud ----------*/
    // Remove planes first
    std::cout << "Extracting from full...\n";
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.01);
    i=0;
    nr_points = (int) laserCloudSrc->points.size ();
//    while (laserCloudSrc->points.size () > 0.5 * nr_points)
//    {
//        // Segment the largest planar component from the remaining cloud
//        seg.setInputCloud (laserCloudSrc);
//        seg.segment (*inliers, *coefficients);
//        if (inliers->indices.size () < 200)
//        {
//            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
//            break;
//        }
//
//        // Extract the planar inliers from the input cloud
//        pcl::ExtractIndices<pcl::PointXYZ> extract;
//        extract.setInputCloud (laserCloudSrc);
//        extract.setIndices (inliers);
//        extract.setNegative (false);
//
//        // Get the points associated with the planar surface
//        extract.filter (*cloud_plane);
//        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
//
//        // Remove the planar inliers, extract the rest
//        extract.setNegative (true);
//        extract.filter (*cloud_f);
//        *laserCloudSrc = *cloud_f;
//    }

    // Remove ground points and some far points
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudSrcFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    for (int l = 0; l < laserCloudSrc->points.size(); ++l)
    {
        pcl::PointXYZ curPoint = laserCloudSrc->points[l];
        if(checkWithPriorInfo(curPoint))
            laserCloudSrcFiltered->push_back(laserCloudSrc->points[l]);
    }

    // Set KdTree object for the search method of the extraction
    tree->setInputCloud (laserCloudSrcFiltered);
    ec.setClusterTolerance (2.0); // 1.0m
    ec.setMinClusterSize (10);
    ec.setMaxClusterSize (300);
    ec.setSearchMethod (tree);
    ec.setInputCloud (laserCloudSrcFiltered);
    ec.extract (cluster_indices);
    j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (laserCloudSrcFiltered->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        bool isMatched = false;
        BoundingBox tmpBB(cloud_cluster);
        for (int k = 0; k < vehicleBoundingBoxVec.size(); ++k)
        {
            if(vehicleBoundingBoxVec[k].checkIfContainedBy(tmpBB))
                isMatched = true;

        }
        if(true)
        {
            std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points."
                      << std::endl;
            std::stringstream ss;
            ss << resultPrefixString << j << ".pcd";
            writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false); //*
            j++;
        }
    // Load pointcloud
    std::cout << "Loading..." << std::endl;
    pcl::io::loadPCDFile(srcPointCloudPathString.c_str(), cloud_blob);
    pcl::fromPCLPointCloud2(cloud_blob, *laserCloudSrc); //* convert from
    }

    /*---------- 5.Display the outliers(green) from src and full pointcloud from ref ----------*/
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    viewer = rgbVis(point_cloud_ptr);
//    viewer = simpleVis(laserCloudSrc);
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
}

void printHelp()
{
    cout << "Not enough argument.\n";
}





std::vector<std::pair<int, int>> clusterMatchingICP(std::string pointCloudPath1, int clusterNum1, std::string pointCloudPath2, int clusterNum2)
{
    // parameters
    int pointNumThreshold = 100;  // a cloud contains less points will be discarded.
    double scoreThreshold = 0;  // matches under this will be discarded.
    int maxIterNum = 100000;
    double distanceThreshold = 5;


    // load into data
    std::string cloudFilePrefix = "cloud_cluster_";
    std::string cloudFileSuffix = ".pcd";

    // make sure the first one is smaller
//    if (clusterNum1 > clusterNum2){
//        int temp = clusterNum1;
//        std::string tempPath = pointCloudPath1;
//        clusterNum1 = clusterNum2;
//        pointCloudPath1 = pointCloudPath2;
//        pointCloudPath2 = tempPath;
//        clusterNum2 = temp;
//    }
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters1;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters2;
    std::vector<std::pair<int, int>> matches;

    clusters1 = utils::loadCloudsFromDirectory(pointCloudPath1, clusterNum1);
    clusters2 = utils::loadCloudsFromDirectory(pointCloudPath2, clusterNum2);

    // Ceres ICP
//    std::set<int> unmatched;
//    for (int i = 0; i < clusterNum2; i++){
//        unmatched.insert(i);
//    }
//
//    for (int i = 0; i < clusterNum1; i++){
//        std::cout<< i <<endl;
//
//        // target and big loop variables initialize here
//        int bestMatch = -1;
//        double bestScore = -1;
//
//        for (int j = 0; j< clusterNum2; j++){
//            if (unmatched.find(j) == unmatched.end()){        //  first come, first use rule. may use a global optima later
//                continue;
//            }
//
//            // param and small loop variable initialize here
//            pcl::PointCloud<pcl::PointXYZ>::Ptr targetCloud(new pcl::PointCloud<pcl::PointXYZ>);
//            //targetCloud = clusters1[i]->makeShared();
//            pcl::copyPointCloud(*clusters1[i], *targetCloud);
//            pcl::PointCloud<pcl::PointXYZ>::Ptr refCloud(new pcl::PointCloud<pcl::PointXYZ>);    // create a temporary cloud to perform the iterative transformation
//            pcl::copyPointCloud(*clusters2[j], *refCloud);
//            //refCloud = clusters2[j]->makeShared();
//            std::vector<bool> inliers;
//            bool isConverge = false;
//            int iter = 0;
//            Eigen::Isometry3d fromCurToRef_estimated = Eigen::Isometry3d::Identity();
//            double robustNormWidth = 2.5;
//            // Ceres ICP iteration
//            while(!isConverge && iter < 1500)  // 100
//            {
//
//                double outlierThreshold = 200;
//                if(iter == 3)
//                    outlierThreshold = 20;
//
//                outlierThreshold -= iter*0.02;
//                outlierThreshold = outlierThreshold < 10 ? 10 : outlierThreshold;
//                Eigen::Isometry3d fromCurToRef_inc = MappingICP(targetCloud, refCloud, outlierThreshold, robustNormWidth);
//                std::cout << fromCurToRef_inc.matrix() << std::endl;
//                cout << refCloud->points.size()<<endl;
//                cout << targetCloud->points.size()<<endl;
//                // terminate rule
//                if(fabs(fromCurToRef_inc.translation()[0]) < 0.001 && fabs(fromCurToRef_inc.translation()[1]) < 0.001 && fabs(fromCurToRef_inc.translation()[2]) < 0.001) // 0.001
//                    isConverge = true;
//
//                for (int i = 0; i < refCloud->points.size(); i++)
//                {
//                    Eigen::Vector4d oriPoint;
//                    oriPoint << refCloud->points[i].x, refCloud->points[i].y, refCloud->points[i].z, 1.0;
//                    Eigen::Vector4d transformedPoint = fromCurToRef_inc * oriPoint;
//                    refCloud->points[i].x = transformedPoint[0];
//                    refCloud->points[i].y = transformedPoint[1];
//                    refCloud->points[i].z = transformedPoint[2];
//                }
//                fromCurToRef_estimated = fromCurToRef_inc*fromCurToRef_estimated;
//                std::cout << "iter: " << iter++ << std::endl;
//            }
//            // Two option: use squared distance or iter number.
//            // Try iter number first
//            std::cout << "Final: "<< fromCurToRef_estimated.matrix() << std::endl;
//            refCloud->width = (int) clusters2[j]->points.size();
//            refCloud->height = 1;
//
//            std::cout << "has converged:" << isConverge << std::endl;
//            if (isConverge){
//                double score = calcFitnessScore(*refCloud, *targetCloud);
//                cout << "Current Score : " << score << " " << endl;
//                pairwiseVisualizeCloud(targetCloud, refCloud);
//                if(score <= bestScore){     // if current score is larger and is over the threshold, save it
//                    bestScore = score;
//                    bestMatch = j;
//                }
//            }
//        }
//        if (bestMatch == -1){   // no valid match found
//            std::cout<<"unable to match No. " << i << " cluster" << endl;
//            continue;
//        }
//        unmatched.erase(bestMatch);   // remove from unmatched clusters
//        matches.push_back(std::make_pair(i, bestMatch));
//        std::cout << i<< "  " << bestMatch <<endl;
//    }



    // create ICP and config
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    //pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    // Set the max correspondence distance to 5m
    icp.setMaxCorrespondenceDistance(3); // 5
    // Set the maximum number of iterations (criterion 1)
    icp.setMaximumIterations(1000000);   // 100000  //1000000
    // Set the transformation epsilon (criterion 2)
    // icp.setTransformationEpsilon (1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    icp.setEuclideanFitnessEpsilon(0.01);   // 3

    std::set<int> unmatched;
    std::vector<double> distVec;
    for (int i = 0; i < clusterNum2; i++){
        unmatched.insert(i);
    }

    for (int i = 0; i < clusterNum1; i++){
        std::cout<< i <<endl;
        icp.setInputSource(clusters1[i]);
        double maxScore = 0;
        int bestMatch = -1;
        for (int j = 0; j< clusterNum2; j++){
            if (unmatched.find(j) == unmatched.end()){        //  first come, first use rule. may use a global optima later
                continue;
            }
            icp.setInputTarget(clusters2[j]);
            pcl::PointCloud<pcl::PointXYZ> Final;
            icp.align(Final);
            bool isConverge = icp.hasConverged();
            if (!isConverge){
                continue;
            }
            double score = icp.getFitnessScore();
            std::cout << "has converged:" << isConverge << " score: " <<
                      score << std::endl;
            std::cout << icp.getFinalTransformation() << std::endl;
            if (score > maxScore and score >= scoreThreshold){     // if current score is larger and is over the threshold, save it
//                pcl::PointCloud<pcl::PointXYZ>::Ptr transformed = utils::transformCloudFromMatrix(icp.getFinalTransformation(), clusters1[i]);
//                utils::pairwiseVisualizeCloud(transformed, clusters2[j]);
                maxScore = score;
                bestMatch = j;
            }
        }
        if (bestMatch == -1){   // no valid match found
            continue;
        }
        unmatched.erase(bestMatch);   // remove from unmatched clusters
        matches.push_back(std::make_pair(i, bestMatch));
        double distance = utils::calcDistanceBetweenClouds(*clusters1[i], *clusters2[bestMatch]);
        distVec.push_back(distance);
        std::cout << i<< "  " << bestMatch <<endl;
    }

    // show all matches
    for(int i =0; i<matches.size(); i++){
        const std::string current = "Match " +  std::to_string(matches[i].first) + " on " + std::to_string(matches[i].second) +
                                    " with distance of " + std::to_string(distVec[i]);
        cout << current << endl;
        utils::pairwiseVisualizeCloud(clusters1[matches[i].first], clusters2[matches[i].second], current);
    }
    return matches;

}




// feature based matching
void clusterMatchingFeature(std::string pointCloudPath1, int clusterNum1, std::string pointCloudPath2, int clusterNum2){
    // load into data
    std::string cloudFilePrefix = "cloud_cluster_";
    std::string cloudFileSuffix = ".pcd";
    if (clusterNum1 > clusterNum2){
        int temp = clusterNum1;
        std::string tempPath = pointCloudPath1;
        clusterNum1 = clusterNum2;
        pointCloudPath1 = pointCloudPath2;
        pointCloudPath2 = tempPath;
        clusterNum2 = temp;
    }

    for(int i=0; i< clusterNum1;i++) {
        // load targets
        std::string surPath = pointCloudPath1 + cloudFilePrefix + std::to_string(i) + cloudFileSuffix;
        pcl::PointCloud<pcl::PointXYZ>::Ptr curCloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PCLPointCloud2 cloud_blob;
        pcl::io::loadPCDFile (surPath.c_str(), cloud_blob);
        pcl::fromPCLPointCloud2(cloud_blob, *curCloud);
        std::vector<FeatureCloud> object_templates;
        object_templates.resize(0);
        cout<<"Points before downsampling:" << curCloud->points.size() << endl;
        for (int j = 0; j < clusterNum2; j++) {
            // load templates
            FeatureCloud template_cloud;
            std::string curPath = pointCloudPath2 + cloudFilePrefix + std::to_string(j) + cloudFileSuffix;
            template_cloud.loadInputCloud(curPath);
            std::cout << "POINTS:" << template_cloud.getPointCloud()->points.size() <<endl;
            object_templates.push_back(template_cloud);
        }
//        // Preprocess the cloud by...
//        // ...removing distant points
//        const float depth_limit = 5;
//        pcl::PassThrough<pcl::PointXYZ> pass;
//        pass.setInputCloud (curCloud);
//        pass.setFilterFieldName ("z");
//        pass.setFilterLimits (0, depth_limit);
//        pass.filter (*curCloud);

        // ... and downsampling the point cloud
//        const float voxel_grid_size = 0.5f;
//        pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
//        vox_grid.setInputCloud (curCloud);
//        vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
//        pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ>);
//        vox_grid.filter (*tempCloud);
//        curCloud = tempCloud;
        // Assign to the target FeatureCloud
        FeatureCloud target_cloud;
        target_cloud.setInputCloud (curCloud);
        std::cout << "Points after downsampling" << target_cloud.getPointCloud()->points.size() << endl;

        // Set the TemplateAlignment inputs
        TemplateAlignment template_align;
        for (size_t i = 0; i < object_templates.size (); ++i)
        {
            template_align.addTemplateCloud (object_templates[i]);
        }
        template_align.setTargetCloud (target_cloud);

        // Find the best template alignment
        TemplateAlignment::Result best_alignment;
        int best_index = template_align.findBestAlignment (best_alignment);
        const FeatureCloud &best_template = object_templates[best_index];
        cout << "Match " << i << " on " << best_index << endl;
        // Print the alignment fitness score (values less than 0.00002 are good)
        printf ("Best fitness score: %f\n", best_alignment.fitness_score);

        // Print the rotation matrix and translation vector
        Eigen::Matrix3f rotation = best_alignment.final_transformation.block<3,3>(0, 0);
        Eigen::Vector3f translation = best_alignment.final_transformation.block<3,1>(0, 3);

        printf ("\n");
        printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

    }

}

// do multi-frame matching among a list of clouds, based on the first frame
// assume all cluster in first frame is valid (may not necessary)
// visualize in the same window
void multiFrameMatching(std::string pointCloudPath, int cloudNum){
    std::string folderSuffix = "cloud";
    std::string clusterPrefix = "cloud_cluster_";
    std::string clusterSuffix = ".pcd";
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> results;
    std::vector<int> groups;
    for(int _begin = 1;_begin < cloudNum; _begin ++){
        std::string curFolder = pointCloudPath + folderSuffix + std::to_string(_begin) + '/';
        std::string nextFolder = pointCloudPath + folderSuffix + std::to_string(_begin+1) + '/';
        int curClusterNum = utils::getFileCountInFolder(curFolder)-1;   // -1 for the viewer.sh in folder
        int nextClusterNum = utils::getFileCountInFolder(nextFolder)-1;
        if(_begin == 1){   // create the vector to hold each cloud in the first frame
            groups.resize(curClusterNum);
            for(int i=0; i<curClusterNum; i++){
                std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> curClouds;
                std::string curPath = curFolder + clusterPrefix + std::to_string(i) + clusterSuffix;
                pcl::PointCloud<pcl::PointXYZ>::Ptr curCloud(new pcl::PointCloud<pcl::PointXYZ>);
                pcl::PCLPointCloud2 cloud_blob;
                pcl::io::loadPCDFile(curPath.c_str(), cloud_blob);
                pcl::fromPCLPointCloud2(cloud_blob, *curCloud); //* convert from pcl/PCLPointCloud2 to pcl::PointCloud<T>
                curClouds.push_back(curCloud->makeShared());
                results.push_back(curClouds);
                groups[i] = i;
            }
        }
        std::vector<std::pair<int, int>> matches = clusterMatchingICP(curFolder, curClusterNum, nextFolder, nextClusterNum);
        for(int i=0; i<groups.size(); i++){
            bool isFound = false;
            for (int j=0; j<matches.size(); j++){
                if(matches[j].first == groups[i]){
                    groups[i] == matches[j].second;
                    isFound = true;
                    break;
                }
            }
            if(!isFound){
                groups[i] = -1;
            }
        }
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> nextClusters = utils::loadCloudsFromDirectory(nextFolder, nextClusterNum);
        for(int i=0; i<groups.size(); i++){
            if(groups[i] == -1)
                continue;
            results[i].push_back(nextClusters[groups[i]]);
        }
    }
    for(int i=0; i<results.size(); i++){
        utils::multiVisualizeCloud(results[i]);
    }


}





int main(int argc, char** argv)
{
//    if (argc < 3)
//    {
//        printHelp();
//        return 1;
//    }
//    std::string srcPointCloudPathString = argv[1];
//    std::string refPointCloudPathString = argv[2];
//
//    cout << srcPointCloudPathString << endl << refPointCloudPathString << endl;
//    findVehicles(srcPointCloudPathString, refPointCloudPathString);
    std::string pointCloudPath1 = "../data2/cloud3/";
    std::string pointCloudPath = "../data2/";
    multiFrameMatching(pointCloudPath,utils::getFileCountInFolder(pointCloudPath));
    int clusterNum1 = 3;
    std::string pointCloudPath2 = "../data2/cloud4/";
    int clusterNum2 = 5;
//    clusterMatchingICP(pointCloudPath1, clusterNum1, pointCloudPath2, clusterNum2);
//    clusterMatchingFeature(pointCloudPath1, clusterNum1, pointCloudPath2, clusterNum2);


    return 0;
}
