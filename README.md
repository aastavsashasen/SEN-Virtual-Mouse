# SEN Virtual Mouse Program

Virtual mouse program with OpenCV written in Python. Goal is to create a program that allows control of the mouse with natural hand gestures that mimic those used to control a actual computer mouse. Uses Haar Cascade to detect the hand and track the hand, creating a mask that displays only the hand. 2 pieces of colored tape (color 1) on the palm of the hand that allow for mouse movement as well as scaling, and 1 piece of colored tape (color 2) on the first finger to allow for left mouse button control.

You can see a screenshot of what the final result looks like below.

![](/VMpics/final_screenshot.jpg)

## Getting Started

Here we will run through some prerequisites and basic set up, allowing to use this program in you own developmet environment.

### Prerequisites

You will need Numpy (general all round use), Pynput (for mouse control), OpenCV-Python (for computer vision), Matplotlib (displaying histograms), imutils (frame resizing) and Scipy (vector distance calculations).

To get these you can do the following in you command line (I have windows so I am using cmd in this case)

```
pip install numpy
pip install pynput
pip install opencv-python
pip install matplotlib
pip install imutils
pip install scipy
```

### Set-Up

Before running the code you will have to run through it and adjust the parameters depending on your environment, preferences and available colors. I personally use a green and blue electric tape. To run the program just run the code through your IDE (I use pycharm) or initiate it through the command line.

The process of setting up the colors involves taking screenshots of the colored tape attached to your hand in the scene in which you will be using the mouse, compiling these screenshots into a single image and placing that compiled image in the relevant path (hand/color1/COLOR1_final.jpg) or (hand/color2/COLOR2_final.jpg) and finally touching up be directly bounding the colours on a displayed histogram. Ill run you through an example.

First lets take a few screenshots of our hand with the tape attached. Make sure you get a range of positions to ensure you are less prone to color detection errors due to lighting conditions in you scene.

<img src="/VMpics/greentape_example.jpg" alt="drawing" width="500"/>

Simplifying this to isolate the relevant color would result in the following.

<img src="/VMpics/greentape_cropped_sample.jpg" alt="drawing" width="300"/>

Place this file in the following directory relative to the directory that SENVirtualMouse.py is in: 
/hand/color1/COLOR1_final.jpg
We then acquire a histogram and display this histogram using matplotlib. The Y-axis is Hue and the X-axis is saturation.

<img src="/VMpics/histogram_sample.jpg" alt="drawing" width="1000"/>

In SENVirtualMouse.py the line to show the color histograms is commented out (in line 165). You can identify and plot the histogram using the code shown below.
```
roi = cv2.imread("hand/color1/COLOR1_final.jpg")
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv_roi)
roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [1, 180, 0, 256])
plt.figure("Color1")
plt.imshow(roi_hist)
plt.show()
```
We now attempt to box this color to a certain extent in hopes to further reduce error. This is done in lines 152 and 158. However, if this bounding is too 'generous' expect a lot of error due to activation of irrelevant background colors and/or shadows. This process is essentially a trial and error. The result is shown below:

<img src="/VMpics/histogram_sample_adjusted.jpg" alt="drawing" width="1000"/>

I advise that you do the same for your background. Comparing the background histogram and the color1 and color2 histograms to make sure they do not clash.

<img src="/VMpics/Background_and_hist.jpg" alt="drawing" width="600"/>

I could not help it, I had to smile. Although it may be a little difficult to see, in this case you can clearly see that our color1 stands on its own compared to the background, thus is a good choice for a color to track and control the mouse. The background (including me) seems to consist of low hue and saturation values (off-white yellow/orange/red colors).

