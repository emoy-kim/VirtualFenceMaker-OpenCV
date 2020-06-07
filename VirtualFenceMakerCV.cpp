#include "VirtualFenceMakerCV.h"

VirtualFenceMaker::VirtualFenceMaker(float actual_width, float actual_height) :
   ClickedPoint( -1, -1 ), ActualGroundWidth( actual_width ), ActualGroundHeight( actual_height ), 
   FenceRadius( 20.0f ), FenceHeight( 20.0f )
{
   Instance = this;

   GroundImage = cv::imread( std::string(CMAKE_SOURCE_DIR) + "/ground.jpg" );
   if (!GroundImage.empty()) {
      MeterToPixel = static_cast<float>(GroundImage.cols) / static_cast<float>(ActualGroundWidth);
   }
   else std::cout << "Cannot Load the Image...\n";
}

void VirtualFenceMaker::setCamera(
   int width, 
   int height, 
   float focal_length, 
   float pan_angle_in_degree,
   float tilt_angle_in_degree,
   float camera_height_in_meter
)
{
   MainCamera.Width = width;
   MainCamera.Height = height;
   MainCamera.FocalLength = focal_length;
   MainCamera.PanAngle = pan_angle_in_degree * static_cast<float>(CV_PI) / 180.0f;
   MainCamera.TiltAngle = tilt_angle_in_degree * static_cast<float>(CV_PI) / 180.0f;
   MainCamera.Translation.x = ActualGroundHeight * 0.5f;
   MainCamera.Translation.y = 0.0f;
   MainCamera.Translation.z = 0.0f;
   MainCamera.CameraHeight = camera_height_in_meter;
   MainCamera.CameraView = cv::Mat(height, width, CV_8UC3, cv::Scalar::all(255));

   MainCamera.Intrinsic = cv::Matx33f(
      focal_length, 0.0f, static_cast<float>(width) * 0.5f,
      0.0f, focal_length, static_cast<float>(height) * 0.5f,
      0.0f, 0.0f, 1.0f
   );

   const float sin_pan = sinf( MainCamera.PanAngle );
   const float cos_pan = cosf( MainCamera.PanAngle );
   MainCamera.PanningToCamera = cv::Matx33f(
      cos_pan, 0.0f, -sin_pan,
      0.0f, 1.0f, 0.0f,
      sin_pan, 0.0f, cos_pan
   );

   const float sin_tilt = sinf( MainCamera.TiltAngle );
   const float cos_tilt = cosf( MainCamera.TiltAngle );
   MainCamera.TiltingToCamera = cv::Matx33f(
      1.0f, 0.0f, 0.0f,
      0.0f, cos_tilt, -sin_tilt,
      0.0f, sin_tilt, cos_tilt
   );

   MainCamera.ToWorldCoordinate = MainCamera.PanningToCamera.inv() * MainCamera.TiltingToCamera.inv();

   FenceMask = cv::Mat::zeros(height, width, CV_8UC1);
}

bool VirtualFenceMaker::getWorldPoint(cv::Point3f* point3d, cv::Point2f* point2d, const cv::Point& camera_point, float object_height) const
{
   const float half_width = static_cast<float>(MainCamera.Width) * 0.5f;
   const float half_height = static_cast<float>(MainCamera.Height) * 0.5f;
   const float sin_tilt = sin( MainCamera.TiltAngle );
   const float cos_tilt = cos( MainCamera.TiltAngle );
   const float f_mul_sin_tilt = MainCamera.FocalLength * sin_tilt;

   cv::Point3f ground_point;
   ground_point.z = f_mul_sin_tilt + (static_cast<float>(camera_point.y) - half_height) * cos_tilt;

   if (ground_point.z <= 0.0f || MainCamera.CameraHeight < object_height) return false;

   ground_point.z = (MainCamera.CameraHeight - object_height) / ground_point.z;
   ground_point.x = (static_cast<float>(camera_point.x) - half_width) * ground_point.z;
   ground_point.y = (static_cast<float>(camera_point.y) - half_height) * ground_point.z;
   ground_point.z = MainCamera.FocalLength * ground_point.z;

   const cv::Point3f world_point = MainCamera.ToWorldCoordinate * ground_point + MainCamera.Translation;
   const cv::Point2f image_point(world_point.z * MeterToPixel, world_point.x * MeterToPixel);

   if (point3d != nullptr) *point3d = world_point;
   if (point2d != nullptr) *point2d = image_point;

   return 
      0.0f <= image_point.x && image_point.x < static_cast<float>(GroundImage.cols) && 
      0.0f <= image_point.y && image_point.y < static_cast<float>(GroundImage.rows);
}

cv::Vec3b VirtualFenceMaker::getPixelBilinearInterpolated(const cv::Point2f& image_point)
{
   const auto x0 = static_cast<int>(floor( image_point.x ));
   const auto y0 = static_cast<int>(floor( image_point.y ));
   const float tx = image_point.x - static_cast<float>(x0);
   const float ty = image_point.y - static_cast<float>(y0);
   const int x1 = cv::min( x0 + 1, GroundImage.cols - 1 );
   const int y1 = cv::min( y0 + 1, GroundImage.rows - 1 );

   const cv::Vec3b* curr = GroundImage.ptr<cv::Vec3b>(y0);
   const cv::Vec3b* next = GroundImage.ptr<cv::Vec3b>(y1);

   return cv::Vec3b{
      static_cast<uchar>(
         static_cast<float>(curr[x0](0)) * (1.0f - tx) * (1.0f - ty) + static_cast<float>(curr[x1](0)) * tx * (1.0f - ty)
         + static_cast<float>(next[x0](0)) * (1.0f - tx) * ty + static_cast<float>(next[x1](0)) * tx * ty
      ),
      static_cast<uchar>(
         static_cast<float>(curr[x0](1)) * (1.0f - tx) * (1.0f - ty) + static_cast<float>(curr[x1](1)) * tx * (1.0f - ty)
         + static_cast<float>(next[x0](1)) * (1.0f - tx) * ty + static_cast<float>(next[x1](1)) * tx * ty
      ),
      static_cast<uchar>(
         static_cast<float>(curr[x0](2)) * (1.0f - tx) * (1.0f - ty) + static_cast<float>(curr[x1](2)) * tx * (1.0f - ty)
         + static_cast<float>(next[x0](2)) * (1.0f - tx) * ty + static_cast<float>(next[x1](2)) * tx * ty
      )
   };
}

void VirtualFenceMaker::renderMainCamera()
{
   if (MainCamera.CameraView.empty()) {
      std::cout << "Set Main Camera First...\n";
      return;
   }

   cv::Point2f image_point;
   for (int j = 0; j < MainCamera.Height; ++j) {
      auto* view_ptr = MainCamera.CameraView.ptr<cv::Vec3b>(j);
      for (int i = 0; i < MainCamera.Width; ++i) {
         if (getWorldPoint( nullptr, &image_point, cv::Point(i, j) )) {
            view_ptr[i] = getPixelBilinearInterpolated( image_point );
         }
      }
   }
}

void VirtualFenceMaker::getCameraPoint(cv::Point& camera_point, const cv::Point3f& world_point) const
{
   const cv::Point3f p = MainCamera.TiltingToCamera * MainCamera.PanningToCamera * (world_point - MainCamera.Translation);
   cv::Point3f projected = MainCamera.Intrinsic * p;
   if (projected.z == 0.0f) projected.z = 1e-7f;
   projected.x /= projected.z;
   projected.y /= projected.z;
   camera_point.x = static_cast<int>(round( projected.x ));
   camera_point.y = static_cast<int>(round( projected.y ));
}

void VirtualFenceMaker::updateFenceHeight(int mouse_wheel_forward)
{
   if (mouse_wheel_forward >= 0) {
      FenceHeight += 5.0f;
      if (FenceHeight >= 70.0f) FenceHeight -= 5.0f; 
   }
   else {
      FenceHeight -= 5.0f;
      if (FenceHeight < 0.0f) FenceHeight = 0.0f;
   }
}

void VirtualFenceMaker::updateFenceRadius(int mouse_wheel_forward)
{
   if (mouse_wheel_forward >= 0) {
      FenceRadius += 5.0f;
      if (FenceRadius >= ActualGroundHeight * 0.5f) FenceRadius -= 5.0f; 
   }
   else {
      FenceRadius -= 5.0f;
      if (FenceRadius < 5.0f) FenceRadius = 5.0f;
   }
}

void VirtualFenceMaker::renderBoundingBox(const cv::Mat& image, const cv::Point3f& center) const
{
   const cv::Point3f upper_top_left(center.x - FenceRadius, center.y, center.z + FenceRadius);
   const cv::Point3f upper_top_right(center.x + FenceRadius, center.y, center.z + FenceRadius);
   const cv::Point3f upper_bottom_left(center.x - FenceRadius, center.y, center.z - FenceRadius);
   const cv::Point3f upper_bottom_right(center.x + FenceRadius, center.y, center.z - FenceRadius);

   std::vector<cv::Point> upper_bounding_points(4);
   getCameraPoint( upper_bounding_points[0], upper_top_left );
   getCameraPoint( upper_bounding_points[1], upper_top_right );
   getCameraPoint( upper_bounding_points[2], upper_bottom_right );
   getCameraPoint( upper_bounding_points[3], upper_bottom_left );

   for (int i = 0, j = 3; i < 4; j = i++) {
      cv::line( image, upper_bounding_points[i], upper_bounding_points[j], GOLDENROD_COLOR, 2 );
   }

   if (FenceHeight > 0.0f) {
      const cv::Point3f lower_top_left(center.x - FenceRadius, center.y + FenceHeight, center.z + FenceRadius);
      const cv::Point3f lower_top_right(center.x + FenceRadius, center.y + FenceHeight, center.z + FenceRadius);
      const cv::Point3f lower_bottom_left(center.x - FenceRadius, center.y + FenceHeight, center.z - FenceRadius);
      const cv::Point3f lower_bottom_right(center.x + FenceRadius, center.y + FenceHeight, center.z - FenceRadius);

      std::vector<cv::Point> lower_bounding_points(4);
      getCameraPoint( lower_bounding_points[0], lower_top_left );
      getCameraPoint( lower_bounding_points[1], lower_top_right );
      getCameraPoint( lower_bounding_points[2], lower_bottom_right );
      getCameraPoint( lower_bounding_points[3], lower_bottom_left );

      for (int i = 0, j = 3; i < 4; j = i++) {
         cv::line( image, lower_bounding_points[i], lower_bounding_points[j], GOLDENROD_COLOR, 2 );
      }

      for (int i = 0; i < 4; ++i) {
         cv::line( image, upper_bounding_points[i], lower_bounding_points[i], GOLDENROD_COLOR, 2 );
      }
   }
}

void VirtualFenceMaker::renderBoundingEllipse(const cv::Mat& image, const cv::Point3f& center)
{
   FenceMask = cv::Mat::zeros(image.size(), CV_8UC1);

   const cv::Point3f upper_circle_top(center.x, center.y, center.z + FenceRadius);
   const cv::Point3f upper_circle_bottom(center.x, center.y, center.z - FenceRadius);
   const cv::Point3f upper_circle_left(center.x - FenceRadius, center.y, center.z);
   const cv::Point3f upper_circle_right(center.x + FenceRadius, center.y, center.z);
   const cv::Point3f upper_circle_diagonal(center.x + FenceRadius / sqrt( 2.0f ), center.y, center.z + FenceRadius / sqrt( 2.0f ));

   std::vector<cv::Point> upper_bounding_points(5);
   getCameraPoint( upper_bounding_points[0], upper_circle_top );
   getCameraPoint( upper_bounding_points[1], upper_circle_bottom );
   getCameraPoint( upper_bounding_points[2], upper_circle_left );
   getCameraPoint( upper_bounding_points[3], upper_circle_right );
   getCameraPoint( upper_bounding_points[4], upper_circle_diagonal );

   cv::ellipse( image, cv::fitEllipse( upper_bounding_points ), CERULEAN_COLOR, 2 );

   if (FenceHeight > 0.0) {
      const cv::Point3f lower_circle_top(center.x, center.y + FenceHeight, center.z + FenceRadius);
      const cv::Point3f lower_circle_bottom(center.x, center.y + FenceHeight, center.z - FenceRadius);
      const cv::Point3f lower_circle_left(center.x - FenceRadius, center.y + FenceHeight, center.z);
      const cv::Point3f lower_circle_right(center.x + FenceRadius, center.y + FenceHeight, center.z);
      const cv::Point3f lower_circle_diagonal(center.x + FenceRadius / sqrt( 2.0f ), center.y + FenceHeight, center.z + FenceRadius / sqrt( 2.0f ));

      std::vector<cv::Point> lower_bounding_points(5);
      getCameraPoint( lower_bounding_points[0], lower_circle_top );
      getCameraPoint( lower_bounding_points[1], lower_circle_bottom );
      getCameraPoint( lower_bounding_points[2], lower_circle_left );
      getCameraPoint( lower_bounding_points[3], lower_circle_right );
      getCameraPoint( lower_bounding_points[4], lower_circle_diagonal );

      cv::ellipse( image, cv::fitEllipse( lower_bounding_points ), CERULEAN_COLOR, 2 );
      cv::ellipse( FenceMask, cv::fitEllipse( lower_bounding_points ), cv::Scalar(255), -1 );

      for (int i = 0; i < 4; ++i) {
         cv::line( image, upper_bounding_points[i], lower_bounding_points[i], CERULEAN_COLOR, 2 );
      }
   }
}

void VirtualFenceMaker::render(cv::Mat& viewer)
{
   cv::Mat ground;
   cv::resize( GroundImage, ground, GroundImage.size() / 4 );

   cv::Point2f image_point;
   cv::Point3f world_point;
   if (getWorldPoint( &world_point, &image_point, ClickedPoint, FenceHeight )) {
      const cv::Point center = image_point / 4;
      cv::circle( ground, center, 2, RED_COLOR, -1 );
      cv::circle( ground, center, static_cast<int>(FenceRadius * MeterToPixel / 4.0f), RED_COLOR, 2 );

      cv::Point projected_ground_point;
      const cv::Point3f ground_point(world_point.x, world_point.y + FenceHeight, world_point.z);
      getCameraPoint( projected_ground_point, ground_point );
      cv::line( viewer, ClickedPoint, projected_ground_point, RED_COLOR, 2 );
      
      renderBoundingBox( viewer, world_point );
      renderBoundingEllipse( viewer, world_point );
      
      cv::circle( viewer, ClickedPoint, 8, CYAN_COLOR, -1 );
   }

   cv::imshow( "Main Camera", viewer );
   cv::imshow( "Overhead View", ground );
}

void VirtualFenceMaker::pickPointCallback(int evt, int x, int y, int flags, void* param)
{
   static bool update = false;

   if (evt == cv::EVENT_LBUTTONDOWN) {
      update = true;
      ClickedPoint = cv::Point(x, y);
   }
   else if (evt == cv::EVENT_MOUSEWHEEL) {
      if (ClickedPoint.x >= 0) {
         update = true;
         if (flags & cv::EVENT_FLAG_CTRLKEY) updateFenceHeight( cv::getMouseWheelDelta( flags ) );
         else updateFenceRadius( cv::getMouseWheelDelta( flags ) );
      }
   }

   if (update) {
      update = false;
      
      cv::Mat viewer = static_cast<cv::Mat*>(param)->clone();
      render( viewer );
   }
}

void VirtualFenceMaker::pickPointCallbackWrapper(int evt, int x, int y, int flags, void* param)
{
   Instance->pickPointCallback( evt, x, y, flags, param );
}

void VirtualFenceMaker::renderFence()
{
   renderMainCamera();

   int key = -1;
   while (key != 27) {
      cv::Mat viewer = MainCamera.CameraView.clone();
      cv::namedWindow( "Main Camera", 1 );
      cv::imshow( "Main Camera", viewer );
      cv::setMouseCallback( "Main Camera", pickPointCallbackWrapper, &viewer );
      key = cv::waitKey();
   }
   cv::destroyAllWindows();
}

cv::Mat VirtualFenceMaker::getFenceMask() const
{
   return FenceMask.clone();
}