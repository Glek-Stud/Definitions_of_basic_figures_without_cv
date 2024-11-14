## Defining circles without using OpenCV

Python program to detect circles in binary images using a combination
of Sobel edge detection and the Hough Circle Transform. The code processes
n input image, identifies object boundaries, and then detects circles within
those boundaries based on customizable radius ranges and voting thresholds.

The threshold_factor is a key parameter that influences circle detection. 
It represents the minimum percentage of votes a circle must have in the
accumulator to be considered a valid detection. Adjusting this value helps 
in refining the circle detection output. 
Higher values (e.g., 0.8): Detects only strong circles with clear boundaries.
Lower values (e.g., 0.5): More lenient, allowing for the detection of weaker 
or less defined circles

Test input:
![image](https://github.com/user-attachments/assets/ffc3134d-32be-435b-a183-d71e6393660b)
![image](https://github.com/user-attachments/assets/4a95554a-764f-4a6a-b9f3-2fe6ae3413c5)
