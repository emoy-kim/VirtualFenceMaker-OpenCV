/*
 * Author: Emoy Kim
 * E-mail: emoy.kim_AT_gmail.com
 * 
 * This code is a free software; it can be freely used, changed and redistributed.
 * If you use any version of the code, please reference the code.
 * 
 */

#pragma once

#include "_Common.h"

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
		cv::Mat CameraView;
		cv::Matx33f Intrinsic;
		cv::Matx33f PanningToCamera;
		cv::Matx33f TiltingToCamera;
		cv::Matx33f ToWorldCoordinate;
		cv::Point3f Translation;

		Camera() : Width( 0 ), Height( 0 ), FocalLength( 0.0f ), PanAngle( 0.0f ), TiltAngle( 0.0f ), CameraHeight( 0.0f ) {}
	};

	inline static VirtualFenceMaker* Instance = nullptr;

	cv::Point ClickedPoint;

	cv::Mat GroundImage;
	cv::Mat FenceMask;
	float ActualGroundWidth;  // ActualGroundWidth(m) * MeterToPixel(pixel/m) = GroundImage.cols(pixel)
	float ActualGroundHeight; // ActualGroundHeight(m) * MeterToPixel(pixel/m) = GroundImage.rows(pixel)
	float MeterToPixel;
	float FenceRadius;
	float FenceHeight;
	Camera MainCamera;

	bool getWorldPoint(cv::Point3f* point3d, cv::Point2f* point2d, const cv::Point& camera_point, float object_height = 0.0f) const;
	cv::Vec3b getPixelBilinearInterpolated(const cv::Point2f& image_point);

	void renderMainCamera();
	
	void getCameraPoint(cv::Point& camera_point, const cv::Point3f& world_point) const;
	
	void updateFenceHeight(int mouse_wheel_forward);
	void updateFenceRadius(int mouse_wheel_forward);

	void renderBoundingBox(const cv::Mat& image, const cv::Point3f& center) const;
	void renderBoundingEllipse(const cv::Mat& image, const cv::Point3f& center);
	void render(cv::Mat& viewer);
	void pickPointCallback(int evt, int x, int y, int flags, void* param);
	static void pickPointCallbackWrapper(int evt, int x, int y, int flags, void* param);


public:
	VirtualFenceMaker(float actual_width, float actual_height);

	void setCamera(
		int width, 
		int height, 
		float focal_length, 
		float pan_angle_in_degree, 
		float tilt_angle_in_degree,
		float camera_height_in_meter
	);
	void renderFence();
	cv::Mat getFenceMask() const;
};