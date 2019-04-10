#include "VirtualFenceMakerCV.h"

int main()
{
	const float ground_width_in_meter = 320.0f;
	const float ground_height_in_meter = 240.0f;

	VirtualFenceMaker fence_maker(ground_width_in_meter, ground_height_in_meter);
	
	const int width = 1280;
	const int height = 720;
	const float focal_length = 800.0f;
	const float pan_angle_in_degree = 0.0f;
	const float tilt_angle_in_degree = 30.0f;
	const float camera_height_in_meter = 70.0f;

	fence_maker.setCamera( 
		width,
		height,
		focal_length,
		pan_angle_in_degree,
		tilt_angle_in_degree, 
		camera_height_in_meter
	);
	fence_maker.renderFence();

	const Mat fence_mask = fence_maker.getFenceMask();
	imshow( "Fence Mask", fence_mask );
	waitKey();

	return 0;
}