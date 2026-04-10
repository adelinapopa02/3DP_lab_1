#include <algorithm>
#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <chrono>

#include "sgm.h"
#include <opencv2/core.hpp>
#include <eigen3/Eigen/Dense>
#define NUM_DIRS 3
#define PATHS_PER_SCAN 8

using namespace std;
using namespace cv;
using namespace Eigen;
static char hamLut[256][256];
static int directions[NUM_DIRS] = {0, -1, 1};

//compute values for hamming lookup table
void compute_hamming_lut()
{
  for (uchar i = 0; i < 255; i++)
  {
    for (uchar j = 0; j < 255; j++)
    {
      uchar census_xor = i^j;
      uchar dist=0;
      while(census_xor)
      {
        ++dist;
        census_xor &= census_xor-1;
      }
      
      hamLut[i][j] = dist;
    }
  }
}

namespace sgm 
{
  SGM::SGM(unsigned int disparity_range, bool adaptive_penalties, unsigned int p1, unsigned int p2,
           float conf_thresh, unsigned int window_height, unsigned window_width):
    disparity_range_(disparity_range), adaptive_penalties_(adaptive_penalties), p1_(p1), p2_(p2),
    conf_thresh_(conf_thresh), window_height_(window_height), window_width_(window_width)
  {
    compute_hamming_lut();
  }

  // set images and initialize all the desired values
  void SGM::set(const  cv::Mat &left_img, const  cv::Mat &right_img, const  cv::Mat &right_mono, const  cv::Mat &left_mono)
  {
    views_[0] = left_img;
    views_[1] = right_img;
    right_mono_ = right_mono;
    left_mono_ = left_mono;


    cv::Mat grad_x, grad_y;
    cv::Sobel(left_img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(left_img, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, right_grad_);
    cv::normalize(right_grad_, right_grad_, 0, 1.0, cv::NORM_MINMAX);

    mean_grad_val_ = (float)cv::mean(right_grad_)[0];

    height_ = left_img.rows;
    width_ = right_img.cols;
    pw_.north = window_height_/2;
    pw_.south = height_ - window_height_/2;
    pw_.west = window_width_/2;
    pw_.east = width_ - window_height_/2;
    init_paths();
    cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    inv_confidence_.resize(height_, vector<float>(width_));
    aggr_cost_.resize(height_, ul_array2D(width_, ul_array(disparity_range_)));
    path_cost_.resize(PATHS_PER_SCAN, ul_array3D(height_, ul_array2D(width_, ul_array(disparity_range_)))
    );
  }

  //initialize path directions
  void SGM::init_paths()
  {
    for(int i = 0; i < NUM_DIRS; ++i)
    {
      for(int j = 0; j < NUM_DIRS; ++j)
      {
        // skip degenerate path
        if (i==0 && j==0)
          continue;
        paths_.push_back({directions[i], directions[j]});
      }
    }
  }

  //compute costs and fill volume cost cost_
  void SGM::calculate_cost_hamming()
  {
    uchar census_left, census_right, shift_count;
    cv::Mat_<uchar> census_img[2];
    cv::Mat_<uchar> census_mono[2];
    cout << "\nApplying Census Transform" <<endl;
    
    for( int view = 0; view < 2; view++)
    {
      census_img[view] = cv::Mat_<uchar>::zeros(height_,width_);
      census_mono[view] = cv::Mat_<uchar>::zeros(height_,width_);

      for (int r = 1; r < height_ - 1; r++)
      {
        uchar *p_center = views_[view].ptr<uchar>(r),
              *p_census = census_img[view].ptr<uchar>(r);
        p_center += 1;
        p_census += 1;

        for(int c = 1; c < width_ - 1; c++, p_center++, p_census++)
        {
          uchar p_census_val = 0, m_census_val = 0, shift_count = 0;
          for (int wr = r - 1; wr <= r + 1; wr++)
          {
            for (int wc = c - 1; wc <= c + 1; wc++)
            {

              if( shift_count != 4 )//skip the center pixel
              {
                p_census_val <<= 1;
                m_census_val <<= 1;
                if(views_[view].at<uchar>(wr,wc) < *p_center ) //compare pixel values in the neighborhood
                  p_census_val = p_census_val | 0x1;

              }
              shift_count ++;
            }
          }
          *p_census = p_census_val;
        }
      }
    }

    cout <<"\nFinding Hamming Distance" <<endl;
    
    for(int r = window_height_/2 + 1; r < height_ - window_height_/2 - 1; r++)
    {
      for(int c = window_width_/2 + 1; c < width_ - window_width_/2 - 1; c++)
      {
        for(int d = 0; d < disparity_range_; d++)
        {
          long cost = 0;
          for(int wr = r - window_height_/2; wr <= r + window_height_/2; wr++)
          {
            uchar *p_left = census_img[0].ptr<uchar>(wr),
                  *p_right = census_img[1].ptr<uchar>(wr);


            int wc = c - window_width_/2;
            p_left += wc;
            p_right += wc + d;



            const uchar out_val = census_img[1].at<uchar>(wr, width_ - window_width_/2 - 1);


            for(; wc <= c + window_width_/2; wc++, p_left++, p_right++)
            {
              uchar census_left, census_right, m_census_left, m_census_right;
              census_left = *p_left;
              if (c+d < width_ - window_width_/2)
              {
                census_right= *p_right;

              }

              else
              {
                census_right= out_val;
              }


              cost += ((hamLut[census_left][census_right]));
            }
          }
          cost_[r][c][d]=cost;
        }
      }
    }
  }

  void SGM::compute_path_cost(int direction_y, int direction_x, int cur_y, int cur_x, int cur_path)
  {
    unsigned long prev_cost, best_prev_cost, no_penalty_cost, penalty_cost, 
                  small_penalty_cost, big_penalty_cost;

    //////////////////////////// Code to be completed (1/5) /////////////////////////////////
    // Complete the compute_path_cost() function that, given: 
    // i) a single pixel p defined by its coordinates cur_x and cur_y; 
    // ii) a path with index cur_path (cur_path=0,1,..., PATHS_PER_SCAN - 1, a path for 
    //     each direction), and;
    // iii) the direction increments direction_x and direction_y associated with the path 
    //      with index cur_path (that are the dx,dy increments to move along the path 
    //      direction, both can be -1, 0, or 1), 
    // should compute the path cost for p for all the possible disparities d from 0 to 
    // disparity_range_ (excluded, already defined). The output should be stored in the 
    // tensor (already allocated) path_cost_[cur_path][cur_y][cur_x][d], for all possible d.
    /////////////////////////////////////////////////////////////////////////////////////////

    // Constant penalties
    if(!adaptive_penalties_)
    {
      // if the processed pixel is the first or the last in the directional path:
      if( cur_y == pw_.north || cur_y == pw_.south || cur_x == pw_.east || cur_x == pw_.west)
      {
        for (unsigned int d = 0; d < (int)disparity_range_; d++)
        {
          // The path cost at the start is simply the initial matching cost (Hamming distance)
          path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d];
        }
      }
      else
      {
        // Identify the coordinates of the previous pixel along the current 1D path
        int prev_x = cur_x - direction_x;
        int prev_y = cur_y - direction_y;

        // Step 1: Find the minimum path cost among all disparities at the previous pixel.
        // This 'best_prev_cost' is used later for the P2 penalty and for normalization.
        best_prev_cost = path_cost_[cur_path][prev_y][prev_x][0];
        for (unsigned int d = 1; d < (int)disparity_range_; d++)
        {
          if (path_cost_[cur_path][prev_y][prev_x][d] < best_prev_cost)
            best_prev_cost = path_cost_[cur_path][prev_y][prev_x][d];
        }

        // Pre-calculate penalty constants
        small_penalty_cost = p1_;
        big_penalty_cost = best_prev_cost + p2_;

        // Step 2: Iterate through all possible disparities 'd' for the current pixel
        for (unsigned int d = 0; d < (int)disparity_range_; d++)
        {
          // Case A: No change in disparity from the previous pixel
          no_penalty_cost = path_cost_[cur_path][prev_y][prev_x][d];

          // Case B: Disparity shifts by exactly 1. 
          // We check boundaries to avoid out-of-bounds access and use 0xFFFFFFFF for impossible shifts.
          penalty_cost = std::min(
            (d > 0) ? path_cost_[cur_path][prev_y][prev_x][d - 1] + small_penalty_cost : 0xFFFFFFFF,
            (d < disparity_range_ - 1) ? path_cost_[cur_path][prev_y][prev_x][d + 1] + small_penalty_cost : 0xFFFFFFFF
          );

          // Step 3: Select the minimum of the three possible transitions (Stay, Shift 1, or Jump)
          prev_cost = std::min({no_penalty_cost, penalty_cost, big_penalty_cost});

          // Step 4: Final Path Cost Calculation
          // We add the local matching cost to the best transition cost.
          // We subtract 'best_prev_cost' to normalize the values.
          path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d] + prev_cost - best_prev_cost;
        }
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////// Code to be completed (5/5) /////////////////////////////////
    // Implement an alternative aggegate cost, based on Spatially Varying Penalties.
    // The goal is to make it "cheaper" for the algorithm to allow
    // a disparity jump when there is a strong gradient in the reference image. You have
    // to adapt the original penalties p1_ and p2_ for each pixel (x, y) based on the local
    // image gradient magnitude |∇I(x,y)| (pre-calculated float matrix right_grad_,
    // you can access it with right_grad_.at<float>(y, x), the gradient magnitude varies
    // from 0 to 1). Adjust the penalties such that:
    // In flat regions (low gradient magnitude), penalties  p1_ and p2_ remain high to
    // enforce smoothness.
    // In edge regions (high gradient magnitude), penalties  p1_ and p2_ are reduced to allow
    // the disparity to adapt to the object boundary.
    /////////////////////////////////////////////////////////////////////////////////////////

    // Adaptive penalties
    else
    {
      // if the processed pixel is the first or the last in the directional path:
      if( cur_y == pw_.north || cur_y == pw_.south || cur_x == pw_.east || cur_x == pw_.west)
      {
        for (unsigned int d = 0; d < (int)disparity_range_; d++)
        {
          path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d];
        }      
      }

      else
      {
        int prev_x = cur_x - direction_x;
        int prev_y = cur_y - direction_y;

        // Retrieve the normalized gradient magnitude (0.0 to 1.0)
        float grad_val = right_grad_.at<float>(cur_y, cur_x);
        
        // HEURISTIC: The Switch (Truncated) Logic
        // We only lower penalties if the gradient is strong enough to be a real depth edge.
        float dynamic_threshold = mean_grad_val_ * 1.5f;        
        
        if (grad_val < dynamic_threshold) 
        {
          // If the gradient is low, we assume it's noise or texture (like text on a cone).
          // We keep penalties high to enforce a smooth surface.
          small_penalty_cost = p1_;
          big_penalty_cost = p2_;
        } 
        else 
        {
          // If the gradient is high, we assume it's a physical object boundary.
          // We make it "cheaper" to jump disparity so the map stays sharp at the edges.
          small_penalty_cost = 1UL; 
          big_penalty_cost = 20UL; 
        }

        best_prev_cost = path_cost_[cur_path][prev_y][prev_x][0];
        
        for (unsigned d = 1; d < (int)disparity_range_; d++)
        {
          if (path_cost_[cur_path][prev_y][prev_x][d] < best_prev_cost)
            best_prev_cost = path_cost_[cur_path][prev_y][prev_x][d];
        }

        big_penalty_cost = best_prev_cost + big_penalty_cost;

        for (unsigned int d = 0; d < (int)disparity_range_; d++)
        {
          no_penalty_cost = path_cost_[cur_path][prev_y][prev_x][d];

          penalty_cost = std::min(
            (d > 0) ? path_cost_[cur_path][prev_y][prev_x][d - 1] + small_penalty_cost : 0xFFFFFFFF,
            (d < disparity_range_ - 1) ? path_cost_[cur_path][prev_y][prev_x][d + 1] + small_penalty_cost : 0xFFFFFFFF
          );

          prev_cost = std::min({no_penalty_cost, penalty_cost, big_penalty_cost});
          path_cost_[cur_path][cur_y][cur_x][d] = cost_[cur_y][cur_x][d] + prev_cost - best_prev_cost;        
        }
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    
  }

  
  void SGM::aggregation()
  {
    
    //for all defined paths
    for(int cur_path = 0; cur_path < PATHS_PER_SCAN; ++cur_path)
    {

      //////////////////////////// Code to be completed (2/5) /////////////////////////////////
      // Initialize the variables start_x, start_y, end_x, end_y, step_x, step_y with the 
      // right values, after that uncomment the code below
      /////////////////////////////////////////////////////////////////////////////////////////

      // Retrieve the unit direction (-1, 0, or 1) for the current path
      int dir_x = paths_[cur_path].direction_x;
      int dir_y = paths_[cur_path].direction_y;
      
      int start_x, start_y, end_x, end_y, step_x, step_y;

      // X-Axis Traversal Logic
      // If moving backward (dir_x == -1), start at the Right (East) and move Left.
      // Otherwise (dir_x == 1 or 0), start at the Left (West) and move Right.
      start_x = (dir_x == -1) ? pw_.east : pw_.west;
      end_x   = (dir_x == -1) ? pw_.west - 1 : pw_.east + 1;
      step_x  = (dir_x == -1) ? -1 : 1;

      // Y-Axis Traversal Logic
      // If moving upward (dir_y == -1), start at the Bottom (South) and move Up.
      // Otherwise (dir_y == 1 or 0), start at the Top (North) and move Down.
      start_y = (dir_y == -1) ? pw_.south : pw_.north;
      end_y   = (dir_y == -1) ? pw_.north - 1 : pw_.south + 1;
      step_y  = (dir_y == -1) ? -1 : 1;
      
      // Perform the 2D scan across the Processing Window.
      for(int y = start_y; y != end_y ; y += step_y)
      {
        for(int x = start_x; x != end_x ; x += step_x)
        {
          compute_path_cost(dir_y, dir_x, y, x, cur_path);
        }
      }
      /////////////////////////////////////////////////////////////////////////////////////////
    }
    
    float alpha = (PATHS_PER_SCAN - 1) / static_cast<float>(PATHS_PER_SCAN);
    //aggregate the costs
    for (int row = 0; row < height_; ++row)
    {
      for (int col = 0; col < width_; ++col)
      {
        for(int path = 0; path < PATHS_PER_SCAN; path++)
        {
          unsigned long min_on_path = path_cost_[path][row][col][0];
          int disp =  0;

          for(unsigned int d = 0; d < disparity_range_; d++)
          {
            aggr_cost_[row][col][d] += path_cost_[path][row][col][d];
            if (path_cost_[path][row][col][d]<min_on_path)
              {
                min_on_path = path_cost_[path][row][col][d];
                disp = d;
              }

          }
          inv_confidence_[row][col] += (min_on_path - alpha * cost_[row][col][disp]);

        }
      }
    }

  }


  void SGM::compute_disparity()
  {
      std::vector<double> sgm_samples;
      std::vector<double> mono_samples;
    
      calculate_cost_hamming();
      aggregation();
      disp_ = Mat(Size(width_, height_), CV_8UC1, Scalar::all(0));
      int n_valid = 0;
      for (int row = 0; row < height_; ++row)
      {
          for (int col = 0; col < width_; ++col)
          {
              unsigned long smallest_cost = aggr_cost_[row][col][0];
              int smallest_disparity = 0;
              for(int d=disparity_range_-1; d>=0; --d)
              {

                  if(aggr_cost_[row][col][d]<smallest_cost)
                  {
                      smallest_cost = aggr_cost_[row][col][d];
                      smallest_disparity = d; 

                  }
              }
              inv_confidence_[row][col] = smallest_cost - inv_confidence_[row][col];

              // If the following condition is true, the disparity at position (row, col) has a good confidence
              if (inv_confidence_[row][col] > 0 && inv_confidence_[row][col] <conf_thresh_)
              {
                //////////////////////////// Code to be completed (3/5) /////////////////////////////////
                // Since the disparity at position (row, col) has a good confidence, it can be added 
                // together with the corresponding unscaled disparity from the right-to-left initial 
                // guess right_mono_.at<uchar>(row, col) to the pool of disparity pairs that will be used
                // to estimate the unknown scale factor.    
                /////////////////////////////////////////////////////////////////////////////////////////

                // Store the trusted SGM disparity.
                sgm_samples.push_back(static_cast<double>(smallest_disparity));
                
                // Store the corresponding unscaled disparity value from the monocular guess.
                mono_samples.push_back(static_cast<double>(right_mono_.at<uchar>(row, col)));

                // These pairs (mono, sgm) represent the (x, y) points in our linear regression 
                // model: SGM = h * Mono + k.

                /////////////////////////////////////////////////////////////////////////////////////////
              }

              disp_.at<uchar>(row, col) = smallest_disparity*255.0/disparity_range_;

          }
      }

      //////////////////////////// Code to be completed (4/5) /////////////////////////////////
      // Using all the disparity pairs accumulated in the previous step, 
      // estimate the unknown scaling factor and scale the initial guess disparities 
      // accordingly. Finally,  and use them to improve/replace the low-confidence SGM 
      // disparities.
      /////////////////////////////////////////////////////////////////////////////////////////
      
      int n = sgm_samples.size();
      if (n > 100) // Ensure we have enough samples
      {
        // Initialize Eigen matrices for the Normal Equation: A * x = b
        // A contains the Monocular values and a column of 1s (for the offset k)
        // b contains the trusted SGM values
          Eigen::MatrixXd A(n, 2);
          Eigen::VectorXd b(n);

          for (int i = 0; i < n; i++) {
              A(i, 0) = mono_samples[i];  // Input feature
              A(i, 1) = 1.0;              // Multiplier for offset 'k'
              b(i) = sgm_samples[i];      // Target value
          }

          // Solve the overdetermined system using LDLT decomposition (Least Squares)
          // Result x contains [h, k]^T
          Eigen::Vector2d x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
          double h = x(0); 
          double k = x(1); 

          std::cout << "Calculated Scale (h): " << h << " Offset (k): " << k << std::endl;

          // Refinement Pass: Use our new math to fix the "bad" pixels
          for (int r = 0; r < height_; r++) {
              for (int c = 0; c < width_; c++) {
                // Only update pixels that were rejected by SGM (Low Confidence)
                  if (inv_confidence_[r][c] <= 0 || inv_confidence_[r][c] >= conf_thresh_) {
          
                      double m_val = static_cast<double>(right_mono_.at<uchar>(r, c));
                      
                      // Apply the calibrated linear model
                      double refined_disparity = h * m_val + k;

                      // Ensure the new value is within valid physical bounds [0, max_disparity]
                      refined_disparity = std::max(0.0, std::min(static_cast<double>(disparity_range_), refined_disparity));

                      // Normalize to 0-255 for the final output image display
                      disp_.at<uchar>(r, c) = static_cast<uchar>(refined_disparity * 255.0 / disparity_range_);
                  }
              }
          }
      }
      /////////////////////////////////////////////////////////////////////////////////////////

  }

  float SGM::compute_mse(const cv::Mat &gt)
  {
    cv::Mat1f container[2];
    cv::normalize(gt, container[0], 0, 85, cv::NORM_MINMAX);
    cv::normalize(disp_, container[1], 0, disparity_range_, cv::NORM_MINMAX);

    cv::Mat1f  mask = min(gt, 1);
    cv::multiply(container[1], mask, container[1], 1);
    float error = 0;
    for (int y=0; y<height_; ++y)
    {
      for (int x=0; x<width_; ++x)
      {
        float diff = container[0](y,x) - container[1](y,x);
        error+=(diff*diff);
      }
    }
    error = error/(width_*height_);
    return error;
  }

  void SGM::save_disparity(const char* out_file_name)
  {
    imwrite(out_file_name, disp_);
    return;
  }

}

