import numpy as np
from pynput.mouse import Button, Controller  # control mouse
import cv2
from matplotlib import pyplot as plt
import imutils
from scipy.spatial import distance as dist  # calc euclidean distance
from collections import OrderedDict  # creates dictionary to remember objects in order
# ---------------------------------------
# The following CentroidTracker class is credited to PyImageSearch
# This is an amazing blog with truly insightful tutorials on a variety of Python implementations
# Find out more here: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
# Hand cascade classifier is obtained from (Github): https://github.com/OAID/CVGesture
# ---------------------------------------
# classes create objects
# when the class is called, __init__() is initiated
# self is used to access variables that belong to the class
class CentroidTracker():
    # accepts a single parameter, max # of frames before object is deleted
    def __init__(self, maxDisappeared = 20):
        # initialize the next objects id
        # appeared objects and disappeared objects dictionaries
        self.nextObjectID = 0
        # dicts use object id as key and store centroid
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # max number of frames that an object is allowed to be
        # disappeared before it is deleted
        self.maxDisappeared = maxDisappeared

    # to add new objects to the object dictionary
    def register(self, centroid):  # centroid tuple is (cX, cY)
        if not bool(self.objects):  # if dict is empty
            self.nextObjectID = 0
        # register an object with the next available ID
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        # once an object is registered, move to the next ID for the next object
        self.nextObjectID += 1


    # to remove objects
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    # implementing the tracker, takes in list of bounding box rectangles
    # which is a list of currently detected objects is form tuple (startX, startY, endX, endY)
    def update(self, rects):  # called every frame, returns our objects
        # print(self.objects.items())
        if len(rects) == 0:  # if there are no objects detected
            # mark all as disappeared
            # to loop through every key in the disappeared dictionary
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1  # 0=not disappeared, 1=disappeared for a frame
                # if we reached max frames for object, deregister the object
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
                    break
            # if there is no info to be updated, return objects
            return self.objects
        # initialize array of detected centroids (current frame)
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # if there are objects detected, compute centroids of each detected rectangle
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX)/2.0)
            cY = int((startY + endY)/2.0)
            inputCentroids[i] = (cX, cY)
        # we are currently not tracking anything, lets register detected centroids (from inputCentroids[])
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # update existing objects (cX, xY) based on new detected centroids
        # for centroid location that minimizes Euclidean distance
        else:
            # grab objectIDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute distance between each pair of object centroids (old) and input centroids (new)
            # must be np.array's, returns array of distances
            # using numpy allows us to sort and find min/max/etc...
            # cdist returns all distances between all points! nxn = (objectCentroids)x(inputCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # find smallest value in each row
            # sort row based on minimum values, so smallest is at 'front' of index list
            rows = D.min(axis=1).argsort()  # old object indexes of rows containing smallest to largest values
            # find smallest value in each column
            # sort using computed row index list
            cols = D.argmin(axis=1)[rows]  # new object indexes of cols containing smallest to largest values
            # to determine if we need to update, register or deregister
            # need to keep track of row and col indexes we have already examined
            usedRows = set()
            usedCols = set()
            # zip() takes 2 lists and pairs each item as tuple
            # set() can contain (only unique values (no indexing)) floats and lists and tuples
            # run through matrix values from smallest to largest min distances
            for (row, col) in zip(rows, cols):
                # ignore is already examined
                if row in usedRows or col in usedCols:
                    continue
                # where objectIDs is a list of object keys
                # rows = old objects, cols = new objects
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                # indicate that we set the new point
                usedRows.add(row)
                usedCols.add(col)
            # if we havent used a row, col index then its new
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            # consists of old object indexes we havent found new points for
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # consists of new object indexes that dont have any old points
            # more object centroids than input centroids, check and see if they have disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    # add to the disappeared counter for those objects
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # deregister object if its been missing too long
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # if input centroids is greater, than we have new objects
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects
# --------------------------------------------------------------------------
ct = CentroidTracker()  # activate the centroid tracker
palm2_cascade = cv2.CascadeClassifier("haarcascade/handcascade/palm_v4.xml")
# You can get the cascade here: https://github.com/OAID/CVGesture. The file is: palm_v4.xml

# HAND ROI, we actually don't use this, it's for potential masking in an updated version (coming soon...)
# roi = cv2.imread("hand/hand4.jpg")
# hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# hue, saturation, value = cv2.split(hsv_roi)
# roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [1, 180, 0, 256])

# ==================================================================== FILTERING

# now we use 2 colours (from tape) to control:
# 1) Mouse movement (I use green tape for this)
# 2) Mouse left click (I use blue tape for this)
# Note that you will have to do this yourself depending on what colored tape is available to you! (look @ readme.md)
# Also notice in line 150 & 156 I increase the tolerance for these specific colors, do this yourself depending on the
# relevant histograms
# GREEN TAPE ROI
roi_greentape = cv2.imread("hand/greentape/greentape_final.jpg")
hsv_roi_greentape = cv2.cvtColor(roi_greentape, cv2.COLOR_BGR2HSV)
hue_g, saturation_g, value_g = cv2.split(hsv_roi_greentape)
roi_hist_greentape = cv2.calcHist([hsv_roi_greentape], [0, 1], None, [180, 256], [1, 180, 0, 256])
roi_hist_greentape[45:100, 60:256] = 255  # increasing our tolerance for the specific color
# BLUE TAPE ROI
roi_b = cv2.imread("hand/blue2/blue_final.jpg")
hsv_roi_b = cv2.cvtColor(roi_b, cv2.COLOR_BGR2HSV)
hue_b, saturation_b, value_b = cv2.split(hsv_roi_b)
roi_hist_b = cv2.calcHist([hsv_roi_b], [0, 1], None, [180, 256], [1, 180, 0, 256])
roi_hist_b[105:140, 30:256] = 255


plt.figure("blue")
plt.imshow(roi_hist_b)
plt.figure("green")
plt.imshow(roi_hist_greentape)
# plt.show()  # uncomment this to visualize the histograms

# def myfilter (mask):  # for future version
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
#     frame_hist_mask = cv2.calcBackProject([mask], [0, 1], roi_hist, [0, 180, 0, 256], 1)
#     _, frame_hist_mask_thresh = cv2.threshold(frame_hist_mask, 0.01, 255, cv2.THRESH_BINARY)
#     # now we wish to clean up our mask to fully encapsulate the hand
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     hand_mask = cv2.filter2D(frame_hist_mask_thresh, -1, kernel)
#     # now lets dilate a tonne to ensure we get the whole hand
#     hand_mask_final = cv2.dilate(hand_mask, kernel, iterations=2)
#     return hand_mask_final


def myfilter_green(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    frame_green_mask = cv2.calcBackProject([mask], [0, 1], roi_hist_greentape, [0, 180, 0, 256], 1)
    _, frame_green_mask_thresh = cv2.threshold(frame_green_mask, 0.001, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    frame_green_mask_thresh = cv2.dilate(frame_green_mask_thresh, kernel, iterations=2)
    return frame_green_mask_thresh


def myfilter_b(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    frame_b_mask = cv2.calcBackProject([mask], [0, 1], roi_hist_b, [0, 180, 0, 256], 1)
    _, frame_b_mask_thresh = cv2.threshold(frame_b_mask, 0.01, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    frame_b_mask_thresh = cv2.erode(frame_b_mask_thresh, kernel, iterations=10)
    frame_b_mask_thresh = cv2.dilate(frame_b_mask_thresh, kernel, iterations=1)
    frame_b_mask_thresh = cv2.erode(frame_b_mask_thresh, kernel2, iterations=1)
    frame_b_mask_thresh = cv2.dilate(frame_b_mask_thresh, kernel2, iterations=2)
    frame_b_mask_thresh = cv2.dilate(frame_b_mask_thresh, kernel3, iterations=1)
    return frame_b_mask_thresh


def cut_corners(mask, top_left_p, top_right_p, bot_left_p, bot_right_p):
    # we cut a section where adjacent & opposite = 35 pixels
    # we also want to scale this cut with the box size
    scale_corners = np.sqrt((top_left_p[0] - top_right_p[0]) ** 2 + (top_left_p[1] - top_right_p[1]) ** 2)/10
    pix = 2 * scale_corners
    (top_left_x, top_left_y) = top_left_p
    top_left = np.array([[top_left_x, top_left_y], [top_left_x+pix, top_left_y], [top_left_x, top_left_y+pix]],
                        dtype=np.int32)
    (top_right_x, top_right_y) = top_right_p
    top_right = np.array([[top_right_x, top_right_y], [top_right_x-pix, top_right_y], [top_right_x, top_right_y+pix]],
                        dtype=np.int32)
    (bot_left_x, bot_left_y) = bot_left_p
    bot_left = np.array([[bot_left_x, bot_left_y], [bot_left_x+pix, bot_left_y], [bot_left_x, bot_left_y-pix]],
                        dtype=np.int32)
    (bot_right_x, bot_right_y) = bot_right_p
    bot_right = np.array([[bot_right_x, bot_right_y], [bot_right_x-pix, bot_right_y], [bot_right_x, bot_right_y-pix]],
                        dtype=np.int32)
    cv2.fillPoly(mask, [top_left], color=(0))  # masked out in black
    cv2.fillPoly(mask, [top_right], color=(0))
    cv2.fillPoly(mask, [bot_left], color=(0))
    cv2.fillPoly(mask, [bot_right], color=(0))
    return None


# First let us set up our videocapture from the webcam
cap = cv2.VideoCapture(0)

count = 0
print("Starting up...")
x = 0
y = 0

# NOTE: the bottom right of the screen is (1279, 719) for my resolution
mouse = Controller()
mouse_sensitivity = 9  # Adjust this according to your required mouse sensitivity
prev_points = []

# Lets initialize the tracker for the hand
trackers = cv2.MultiTracker_create()
tracker = cv2.TrackerCSRT_create()
boxt = []  # Stores the bounding rectangle of the cascade so we start tracking if the cascade disappears

# For the box around our palm
palm_off = 50
x_med = 0
y_med = 0

while True:  # starting our master loop

    _, frame = cap.read()  # Leave original frame
    frame_haar = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)  # To detect the Haar Cascades
    frame_hand = frame.copy()
    frame_hand_mask = frame.copy()  # For working on hand ONLY, ie) create a mask that tracks just the hand
    frame_hand_mask = cv2.cvtColor(frame_hand_mask, cv2.COLOR_BGR2GRAY)
    frame_hand_mask[:] = 0  # make it all black!

    # ==================================================================== MASKING
    # We detect our hand with a Haar Cascade.
    # Let us also track the hand (CSRT) to reduce error and allow us to click (change our open palm gesture,
    # at which time our haar cascade will deactivate) yet have the previously detected hand be tracked.
    # We originally fill a rectangle to create the mask, however, we get errors from similar background colors in
    # the resulting frames corners. So we cut these corners out using cut_corners
    palm2 = palm2_cascade.detectMultiScale(frame_haar, 1.3, 3)
    if len(palm2) != 0:  # 1st Priority Haar Cascade
        for rect in palm2:
            (x, y, w, h) = rect
            frame_haar = cv2.rectangle(frame_haar, (x, y-15), (x + w, y + h), (255), 2)
            cornerz = np.array([[x+10, y-15], [x+10, y+h], [x+w, y+h], [x+w, y-15]], dtype=np.int32)
            print("Cascade activated")
            cv2.fillPoly(frame_hand_mask, [cornerz], color=(255))
            cut_corners(frame_hand_mask, (x+10, y-15), (x+w, y-15), (x+10, y+h), (x+w, y+h))
            boxt = (x+10, y, w-10, h)  # for our tracker
            tracker = cv2.TrackerCSRT_create()
    elif len(boxt) != 0 and len(palm2) == 0:  # If our Cascade is not activated, track the most recent detected
        # create a new object tracker for the bounding box and add it to our multi-object tracker
        trackers = cv2.MultiTracker_create()
        tracker.clear()
        trackers.clear()
        trackers.add(tracker, frame, tuple(boxt))
    else:
        boxt = []
        trackers = cv2.MultiTracker_create()
        tracker.clear()
        trackers.clear()
    # You can not just delete a Tracker, leading to the above mess (high computational cost)
    # but we will work with it for now

    (success, boxes) = trackers.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        track_box = np.array([[x, y - 15], [x, y + h + 10], [x + w, y + h + 10], [x + w, y - 15]], dtype=np.int32)
        if len(palm2) == 0:  # we want to use the tracker if the cascade is not detected
            cv2.fillPoly(frame_hand_mask, [track_box], color=(255))
            cut_corners(frame_hand_mask, (x, y - 15), (x + w, y - 15), (x, y + h + 10), (x + w, y + h + 10))
            print("Tracker activated")

    # ================================================================================= MOUSE MOVEMENT
    # Create a mask to only show the hand, frame_hand_mask is the ouput of our cascade1 (or) tracker
    frame_hand = cv2.bitwise_and(frame_hand, frame_hand, mask=frame_hand_mask)
    frame_hand_read = frame_hand.copy()  # we are going to read the blue contours from here
    # We use the green tape adjusted histogram to identify the green tape on my palm...
    frame_greentape = myfilter_green(frame_hand)
    _, contours_green, hierarchy_green = cv2.findContours(frame_greentape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_green = cv2.drawContours(frame_hand, contours_green, -1, (255), 2)
    # Keep track of contours with ID labelling
    # To reduce error from similar background colors (other detected contours) we only register significantly
    # sized contours that appear in the palm of our hand
    rects = []
    for contour_green in contours_green:
        if cv2.contourArea(contour_green) > 100:  # Adjust this accordingly, too low and it considers erroneous contours
            # Compute the center of the contour
            M = cv2.moments(contour_green)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            listy = contour_green.tolist()
            x = listy[0][0][0]
            y = listy[0][0][1]
            xr, yr, w, h = cv2.boundingRect(contour_green)
            # Draw a green rectangle to visualize the bounding rectangle of the green tape
            # Lets make the bounding box bigger by using an offset, for visualization purposes onliy
            offset = 6
            cv2.rectangle(frame_hand, (xr - offset, yr - offset), (xr + w + offset, yr + h + offset), (0, 255, 0), 2)
            box = np.array([xr - offset, yr - offset, xr + w + offset, yr + h + offset])
            rects.append(box.astype("int"))

    objects = ct.update(rects)  # We store detected green tape contours for our Object Tracker
    # Loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # Draw both the ID of the object and the centroid of the object on the output frame
        cv2.circle(frame_hand, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # Now to control the mouse with our detected contour centers of the green tape on our palm
    # Having a list of the previous point will allow us to draw a vector in which direction to move our mouse
    if len(prev_points) > 1:  # Limit the size of previous points, we just need the immediate previous point
        del prev_points[0]
    if len(objects.keys()) == 2:  # Activate control of both green palm contours detected
        ps = list(objects.values())
        cv2.line(frame_hand, tuple(ps[0]), tuple(ps[1]), (255))  # Connect a line between green palm contours
        # Note that tuples can be indexed just like lists
        x_med = int((tuple(ps[0])[0] + tuple(ps[1])[0]) / 2)
        y_med = int((tuple(ps[0])[1] + tuple(ps[1])[1]) / 2)
        prev_points.append([x_med, y_med])
        cv2.circle(frame_hand, (x_med, y_med), 5, (0, 0, 255), -1)  # Control the mouse with this center point
        # We move the mouse relative to its previous position, using pynput
        # Let us also show its position in the previous frame (in pink)
        for p in prev_points:
            cv2.circle(frame_hand, tuple(p), 3, (255, 0, 255), -1)  # Previous point drawn in pink
        x_move = x_med - tuple(prev_points[0])[0]
        y_move = y_med - tuple(prev_points[0])[1]
        print("relative mouse motion in x,y = ", -x_move * mouse_sensitivity, y_move * mouse_sensitivity)
        if abs(x_move) > 0 and abs(y_move) > 0:  # In hopes to reduce random mouse movement when hand is still-
            # increase to 1 if a lot of random mouse movement when hand is still
            # or decrease mouse sensitivity (defined before main while loop) to reduce magnitude of these errors
            mouse.move(-x_move * mouse_sensitivity, y_move * mouse_sensitivity)  # Moves mouse

    # ==================================================================== MOUSE CLICKING
    # We have blue tape on our first finger to allow for left mouse button control.
    # When we perform a clicking motion, the contours are identified and a left click is registered
    # To reduce error from similar blue background colors (other detected contours) we only register significantly
    # sized contours that appear on our first finger
    # If the contour exists within the region of the palm then hold the left mouse button down for the time it exists
    frame_b = myfilter_b(frame_hand_read)
    _, contours_b, hierarchy_b = cv2.findContours(frame_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_b = cv2.drawContours(frame_hand, contours_b, -1, (0, 0, 255), 2)  # Drawn in red

    # Have a box on our palm that will determine whether the finger contour registers as a click or not (in white)
    # Its position will be determined by the mouse move point, and reduce error caused by blue detected out of
    # the region of our palm (false clicks registered). Lets call this box (in white) the click box
    if len(objects.keys()) == 2:  # If both green contours detected...
        ps = list(objects.values())
        # Distance between the green contours is as follows ...
        scale_dist = np.sqrt((tuple(ps[0])[0] - tuple(ps[1])[0])**2 + (tuple(ps[0])[1] - tuple(ps[1])[1])**2)
        # We want to change the size of our click box depending on the scale_dist
        # Allowing us to move our hand forward (changing its scale) and back and still
        # registering clicks with the same gesture
        # The following should be adjusted according to your ideal configuration
        # The click box (in white) should NOT extend out of the region of the upper palm
        scale_factor_y = int(round((scale_dist - 41)*1.5))
        scale_factor_x = int(round((scale_dist - 41)*1))
        # print("scale factor y = ", scale_factor_y)
        x_p = x_med - palm_off
        y_p = y_med - palm_off
        length_y = palm_off * 2
        length_x = palm_off * 2 - 10
        # Note that error between the green palm contours occurs, increasing the box size drastically due to a
        # increase in the scale_distance
        # So let us limit the possible scale factor, and not allow clicks if it is above this (limit the click box size)
        if scale_factor_y < 60:
            cv2.rectangle(frame_hand, (x_p + 20 - scale_factor_x, y_p - 10 - scale_factor_y),
                          (x_p + length_x + scale_factor_x, y_p + length_y - 50), (255, 255, 255), 1)  # Drawn in white
        # Now we draw blue contours for the first finger (left click) and register click if we are within the
        # bounds of the white click box
        for contour_b in contours_b:
            if cv2.contourArea(contour_b) > 200:  # Adjust this value accordingly
                print("counter area blue = ", cv2.contourArea(contour_b))
                # compute the center of the contour
                M = cv2.moments(contour_b)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                listy = contour_b.tolist()
                x = listy[0][0][0]
                y = listy[0][0][1]
                xr, yr, w, h = cv2.boundingRect(contour_b)
                # Draw a black rectangle to visualize the bounding rect of our first finger contour
                # (only for visualization purposes)
                offset = 6
                cv2.rectangle(frame_hand, (xr - offset, yr - offset), (xr + w + offset, yr + h + offset), (0, 0, 0),
                              2)  # Drawn in black
                cv2.circle(frame_hand, (cX, cY), 5, (0, 255, 255), -1)  # Centre of left click contour (in yellow)
                box = np.array([xr - offset, yr - offset, xr + w + offset, yr + h + offset])
                rects.append(box.astype("int"))
                # Now to process the click of the mouse
                # Only when our blue finger contour appears within the white self scaling palm box (click box)
                # following if statement is a compilation of many click error reducing statements including conditions:
                # within the white palm box, if its pressed already no need to click again, ignore scale factor errors
                click_count = 0
                if (x_p + 20 - scale_factor_x) < cX < (x_p + length_x + scale_factor_x) and \
                        (y_p - 10 - scale_factor_y) < cY < (y_p + length_y - 50) and \
                        click_count == 0 and (scale_factor_y < 60):
                    mouse.press(Button.left)
                    click_count = 1
                else:
                    mouse.release(Button.left)
                    click_count = 0

    cv2.imshow("Frame (original)", frame)  # The original frame directly from the webcam
    cv2.imshow("Mask for hand", frame_hand_mask)  # A mask of just the hand
    cv2.imshow("(Main) Frame of hand + interactions", frame_hand)  # Displays masked image including detected contours,
    # region of interest for clicking and other mouse control related visualizations
    cv2.imshow("Green Tape", frame_greentape)  # Shows detected green contours
    cv2.imshow("Blue Tape", frame_b)  # Shows detected blue contours


    key = cv2.waitKey(1)

    if key == 27:  # ESC to break loop
        print("Shutting Down...")
        break

# Release capture and destroy windows when done (upon pressing ESC button)
cap.release()
cv2.destroyAllWindows()

# ======================================================== FINAL NOTES/ FUTURE WORK
# Right click can be applied (and will be soon) using the same set up as left click
# No ideas for scrolling just yet
# Better isolation of the hand by color masking (external contour) to reduce errors caused by similar
# background colors to those used for mouse control
# Extension: Implementation of convolutional networks to identify range of gestures, removing the need for
# any tape on the hand. Hopefully creating a more accurate, friendly virtual mouse program that will be
# immune to a change in the surrounding scene and/or environmental lighting conditions




