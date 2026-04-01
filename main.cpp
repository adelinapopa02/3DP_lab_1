#include "sgm.h"

#include <string>


int main(int argc, char** argv)
{

    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <data folder> <output disparity file> <disparity range> " << endl;
        return -1;
    }

    std::string data_dir(argv[1]);

    std::string firstFileName = data_dir + std::string("/right.png");
    std::string secondFileName = data_dir + std::string("/left.png");
    std::string monoRightFileName= data_dir + std::string("/right_mono.png");
    std::string monoLeftFileName= data_dir + std::string("/left_mono.png");
    std::string gtFileName= data_dir + std::string("/rightGT.png");
    std::string outputFileName(argv[2]);
    std::string outputAdaptedFileName(argv[2]);

    std::size_t found = outputAdaptedFileName.rfind(".");
    if (found != std::string::npos)
    {
      outputAdaptedFileName.replace (found,1,"_adapted.");
      std::cout<<"Output disparity file : "<< outputFileName<<std::endl;
      std::cout<<"Output disparity file (with adaptive penalties) : "<< outputAdaptedFileName<<std::endl;
    }
    else
    {
      cerr <<  "Output disparity file without extension (e.g., .png)!" << endl;
      return -1;
    }

    unsigned int disparityRange = atoi(argv[3]);

    cv::Mat firstImage, secondImage, monoRight, monoLeft, gt;

    firstImage = cv::imread(firstFileName.c_str(), IMREAD_GRAYSCALE);
    secondImage = cv::imread(secondFileName.c_str(), IMREAD_GRAYSCALE);
    monoRight = cv::imread(monoRightFileName.c_str(), IMREAD_GRAYSCALE);
    monoLeft = cv::imread(monoLeftFileName.c_str(), IMREAD_GRAYSCALE);
    gt = cv::imread(gtFileName.c_str(), IMREAD_GRAYSCALE);

    if(!firstImage.data || !secondImage.data)
    {
        cerr <<  "Could not open or find one of the images!" << endl;
        return -1;
    }


    sgm::SGM sgm(disparityRange), sgm_adapted(disparityRange, true);

    sgm.set(firstImage, secondImage, monoRight, monoLeft);

    std::cout<<"Computing disparities with constant penalities..."<<std::endl;
    sgm.compute_disparity();
    sgm.save_disparity(outputFileName.c_str());
    std::cout<<"Right Image MSE error with constant penalities: "<<sgm.compute_mse(gt)<<std::endl;

    sgm_adapted.set(firstImage, secondImage, monoRight, monoLeft);
    std::cout<<"Computing disparities with adaptive penalities..."<<std::endl;
    sgm_adapted.compute_disparity();
    sgm_adapted.save_disparity(outputAdaptedFileName.c_str());
    std::cout<<"Right Image MSE error with adaptive penalities: "<<sgm_adapted.compute_mse(gt)<<std::endl;

    return 0;
}
