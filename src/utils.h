//
// Created by lifan on 1/25/17.
//

#ifndef POINTS_GROUPING_UTILS_H
#define POINTS_GROUPING_UTILS_H


//#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>


// Eigens
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>



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



#include <dirent.h>

namespace utils{
    using namespace std;
    void pairwiseVisualizeCloud( pcl::PointCloud<pcl::PointXYZ>::Ptr first,  pcl::PointCloud<pcl::PointXYZ>::Ptr second, const std::string msg = ""){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h (first, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h (second, 255, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(first, tgt_h, "sample cloud");
        viewer->addPointCloud<pcl::PointXYZ>(second, src_h, "sample cloud 2");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud 2");
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        if (msg != "") {
            viewer->addText(msg, 5, 5, 10, 255, 0, 0,  "text");
        }
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }

    }
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> loadCloudsFromDirectory(std::string vPath, int cloudNum, const std::string
                        prefix = "cloud_cluster_", const std::string suffix = ".pcd"){
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds(cloudNum);
        for(int i=0; i<cloudNum; i++) {
            std::string curPath = vPath + prefix + std::to_string(i) + suffix;
            pcl::PointCloud<pcl::PointXYZ>::Ptr curCloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PCLPointCloud2 cloud_blob;
            pcl::io::loadPCDFile(curPath.c_str(), cloud_blob);
            pcl::fromPCLPointCloud2(cloud_blob, *curCloud); //* convert from pcl/PCLPointCloud2 to pcl::PointCloud<T>
            std::cout << "Loading..." << i << "  Size:" << curCloud->points.size() << endl;
            clouds[i] = curCloud->makeShared();  // make copy into vector
        }
        return clouds;
    }


    void multiVisualizeCloud( std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds, const std::string msg = ""){
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

        viewer->setBackgroundColor(0, 0, 0);
        int _size = clouds.size();
        for(int i=0; i<_size; i++) {
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(clouds[i], 255, 0, floor(255 *(double(i)/_size)));
            viewer->addPointCloud<pcl::PointXYZ>(clouds[i], src_h, "sample cloud" + std::to_string(i));
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,  "sample cloud" + std::to_string(i));

        }
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        if (msg != "") {
            viewer->addText(msg, 5, 5, 10, 255, 0, 0, "text");
        }
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }

    }


    double calcDistanceByTranslation(Eigen::Matrix<float, 4, 4> mat){
        // pass
        return sqrt(mat(0, 3)*mat(0, 3) + mat(1, 3)*mat(1, 3) + mat(2, 3)*mat(2, 3));
    }

    double calcFitnessScore(pcl::PointCloud<pcl::PointXYZ> cloud_a, pcl::PointCloud<pcl::PointXYZ> cloud_b){
        pcl::search::KdTree<pcl::PointXYZ> tree_b;
        tree_b.setInputCloud(cloud_b.makeShared());
        double sum_dist_a = 0;
        for (size_t i = 0; i < cloud_a.points.size (); ++i)
        {
            std::vector<int> indices (1);
            std::vector<float> sqr_distances (1);

            tree_b.nearestKSearch(cloud_a.points[i], 1, indices, sqr_distances);
            sum_dist_a += sqrt(sqr_distances[0]);
        }

        // compare B to A
        pcl::search::KdTree<pcl::PointXYZ> tree_a;
        tree_a.setInputCloud (cloud_a.makeShared ());
        double sum_dist_b = 0;
        for (size_t i = 0; i < cloud_b.points.size (); ++i)
        {
            std::vector<int> indices (1);
            std::vector<float> sqr_distances (1);

            tree_a.nearestKSearch (cloud_b.points[i], 1, indices, sqr_distances);
            sum_dist_b  += sqrt(sqr_distances[0]);
        }


        double dist = (sum_dist_a+sum_dist_b)/2/(cloud_b.points.size() + cloud_a.points.size());
        return 1/dist;

    }


    double calcDistanceBetweenClouds(pcl::PointCloud<pcl::PointXYZ> cloud_a, pcl::PointCloud<pcl::PointXYZ> cloud_b)
    {
        pcl::search::KdTree<pcl::PointXYZ> tree_b;
        tree_b.setInputCloud(cloud_b.makeShared());
        double sum_dist_a = 0;
        for (size_t i = 0; i < cloud_a.points.size (); ++i)
        {
            std::vector<int> indices (1);
            std::vector<float> sqr_distances (1);

            tree_b.nearestKSearch(cloud_a.points[i], 1, indices, sqr_distances);
            sum_dist_a += sqrt(sqr_distances[0]);
        }

        // compare B to A
        pcl::search::KdTree<pcl::PointXYZ> tree_a;
        tree_a.setInputCloud (cloud_a.makeShared ());
        double sum_dist_b = 0;
        for (size_t i = 0; i < cloud_b.points.size (); ++i)
        {
            std::vector<int> indices (1);
            std::vector<float> sqr_distances (1);

            tree_a.nearestKSearch (cloud_b.points[i], 1, indices, sqr_distances);
            sum_dist_b  += sqrt(sqr_distances[0]);
        }


        double dist = (sum_dist_a+sum_dist_b)/2/(cloud_b.points.size() + cloud_a.points.size());
        return dist;
    }


    int getFileCountInFolder(string vPath){
        DIR *dir;
        struct dirent *ent;
        int _count = 0;
        if ((dir = opendir (vPath.c_str())) != NULL) {
            /* print all the files and directories within directory */
            while ((ent = readdir (dir)) != NULL) {
                // cout << ent->d_name << endl;
                    _count ++;
            }
            closedir (dir);
            return _count-2;  // -2 for . and ..
        } else {
            /* could not open directory */
            perror ("");
            return EXIT_FAILURE;
        }

    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformCloudFromMatrix(Eigen::Matrix4f transform, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
        // Executing the transformation
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
        // You can either apply transform_1 or transform_2; they are the same
        pcl::transformPointCloud (*cloud, *transformed_cloud, transform);
        return transformed_cloud;
    }






}




#endif //POINTS_GROUPING_UTILS_H
