import cv2
import time
import os
import Cards
import VideoStream

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10
path = os.path.dirname(os.path.abspath(__file__))
train_imgs = Cards.Load_Train_Fruits_Images(path + '/Fruits_Img/')
freq = cv2.getTickFrequency()
frame_rate_calc = 1
font = cv2.FONT_HERSHEY_SIMPLEX
videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 2, 0).start()
time.sleep(1)  # Give the camera time to warm up

while True:
    # Grab frame from video stream
    image = videostream.read()

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
    cnts_sort, cnt_is_card, cnts_count = Cards.find_cards(pre_proc)
    if len(cnts_sort) != 0:
        cards = []
        k = 0
        yellowCount = 0
        redCount = 0
        greenCount = 0
        purpleCount = 0
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(Cards.preprocess_card(cnts_sort[i], image))  # 加入已经扶正的卡片图像
                cards[k].yellowCount, cards[k].redCount, cards[k].greenCount, cards[k].purpleCount = \
                    Cards.find_fruits_in_each_card(cards[k], train_imgs)
                image = Cards.draw_results(image, cards[k])
                yellowCount = yellowCount + cards[k].yellowCount
                redCount = redCount + cards[k].redCount
                greenCount = greenCount + cards[k].greenCount
                purpleCount = purpleCount + cards[k].purpleCount
                k = k + 1
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1, (255, 0, 0), 2)
        if (yellowCount % 5) == 0 and yellowCount != 0:
            print("There are " + str(yellowCount) + " bananas on board!")
        elif (redCount % 5) == 0 and redCount != 0:
            print("There are " + str(redCount) + " strawberries on board!")
        elif (greenCount % 5) == 0 and greenCount != 0:
            print("There are " + str(greenCount) + " lemons on board!")
        elif (purpleCount % 5) == 0 and purpleCount != 0:
            print("There are " + str(purpleCount) + " grapes on board!")
    cv2.putText(image, "FPS: " + str(int(frame_rate_calc)), (10, 26), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "Cards on board: " + str(cnts_count), (10, 56), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.namedWindow('Card Detector', cv2.WINDOW_NORMAL)
    cv2.imshow("Card Detector", image)
    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
videostream.stop()