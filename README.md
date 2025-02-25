# Video-transition-smoothness-using-optical-flow-farneback-approach-
First, it opens original time lapse video and read first frame in video Then it Convert frames to grayscale It estimates pixel motion using  opticalflowfarneback  Apply addWeighted to warp previous frame using flow Blend frame After Save blended frames using VideoWriter Save the smooth transition video   
