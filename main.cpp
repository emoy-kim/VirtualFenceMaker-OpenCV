#include "VirtualFenceMakerCV.h"

int main()
{
	constexpr float ground_width_in_meter = 320.0f;
	constexpr float ground_height_in_meter = 240.0f;

	VirtualFenceMaker fence_maker(ground_width_in_meter, ground_height_in_meter);
	
	constexpr int width = 1280;
	constexpr int height = 720;
	constexpr float focal_length = 800.0f;
	constexpr float pan_angle_in_degree = 0.0f;
	constexpr float tilt_angle_in_degree = 30.0f;
	constexpr float camera_height_in_meter = 70.0f;

	fence_maker.setCamera( 
		width,
		height,
		focal_length,
		pan_angle_in_degree,
		tilt_angle_in_degree, 
		camera_height_in_meter
	);
	fence_maker.renderFence();

	const cv::Mat fence_mask = fence_maker.getFenceMask();
	cv::imshow( "Fence Mask", fence_mask );
	cv::waitKey();

	return 0;
}