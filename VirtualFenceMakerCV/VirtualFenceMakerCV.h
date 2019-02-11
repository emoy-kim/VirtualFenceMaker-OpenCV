#pragma once

#include <OpenCVLinker.h>

using namespace std;
using namespace cv;

#define RED_COLOR   Scalar(0, 0, 255)
#define CYAN_COLOR Scalar(255, 255, 0)
#define CERULEAN_COLOR Scalar(211, 64, 2) 
#define GOLDENROD_COLOR Scalar(103, 214, 252)

class VirtualFenceMaker
{
	struct Camera
	{
		int Width;
		int Height;
		float FocalLength;
		float PanAngle;
		float TiltAngle;
		float CameraHeight;
		Mat CameraView;
		Matx33f Intrinsic;
		Matx33f PanningToCamera;
		Matx33f TiltingToCamera;
		Matx33f ToWorldCoordinate;
		Point3f Translation;

		Camera() : Width( 0 ), Height( 0 ), FocalLength( 0.0f ), PanAngle( 0.0f ), TiltAngle( 0.0f ), CameraHeight( 0.0f ) {}
	};

	static VirtualFenceMaker *Instance;

	Point ClickedPoint;

	Mat GroundImage;
	Mat FenceMask;
	float ActualGroundWidth;  // ActualGroundWidth(m) * MeterToPixel(pixel/m) = GroundImage.cols(pixel)
	float ActualGroundHeight; // ActualGroundHeight(m) * MeterToPixel(pixel/m) = GroundImage.rows(pixel)
	float MeterToPixel;
	float FenceRadius;
	float FenceHeight;
	Camera MainCamera;

	bool getWorldPoint(Point3f* point3d, Point2f* point2d, const Point& camera_point, const float& object_height = 0.0) const;
	Vec3b getPixelBilinearInterpolated(const Point2f& image_point);

	void renderMainCamera();
	
	void getCameraPoint(Point& camera_point, const Point3f& world_point) const;
	
	void updateFenceHeight(const int& mouse_wheel_forward);
	void updateFenceRadius(const int& mouse_wheel_forward);

	void renderBoundingBox(const Mat& image, const Point3f& center) const;
	void renderBoundingEllipse(const Mat& image, const Point3f& center);
	void render(Mat& viewer);
	void pickPointCallback(int evt, int x, int y, int flags, void* param);
	static void pickPointCallbackWrapper(int evt, int x, int y, int flags, void* param);


public:
	VirtualFenceMaker(const float& actual_width, const float& actual_height);

	void setCamera(
		const int& width, 
		const int& height, 
		const float& focal_length, 
		const float& pan_angle_in_degree, 
		const float& tilt_angle_in_degree,
		const float& camera_height_in_meter
	);
	void renderFence();
	Mat getFenceMask() const;
};