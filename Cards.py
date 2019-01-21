# Import necessary packages
import numpy as np
import cv2
import time


# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:

    def __init__(self):
        self.contour = [] # Contour of card
        self.width, self.height = 0, 0 # Width and height of card
        self.corner_pts = [] # Corner points of card
        self.center = [] # Center point of card
        self.warp = [] # 200x300, flattened, grayed, blurred image
        self.yellowCount = 0
        self.redCount = 0
        self.greenCount = 0
        self.purpleCount = 0

class Train_Fruits_Images:

    def __init__(self):
        self.img = []
        self.name = "Placeholder"

### Functions ###

def Load_Train_Fruits_Images(filepath):

    train_fruits_images = []
    i = 0
    
    for Fruit in ['yellow', 'red', 'green', 'purple']:
        train_fruits_images.append(Train_Fruits_Images())
        train_fruits_images[i].name = Fruit
        filename = Fruit + '.jpg'
        train_fruits_images[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_fruits_images

def preprocess_image(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    img_w, img_h = np.shape(image)[:2]
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh


def find_cards(thresh_image):

    dummy,cnts,hier = cv2.findContours(thresh_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]),reverse=True)

    if len(cnts) == 0:
        return [], []

    cnts_count = 0
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts),dtype=int)

    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
        # (size < CARD_MAX_AREA) and (size > CARD_MIN_AREA) and
        if ((hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1
            cnts_count = cnts_count + 1

    return cnts_sort, cnt_is_card, cnts_count


def preprocess_card(contour, image): #just return a single card image

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # 找出卡片包围矩形
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # 找出中点坐标
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # 使用透视变换将卡的图像提取成200x300
    qCard.warp = flattener(image, pts, w, h)
    return qCard


def find_fruits_in_each_card(qCard, train_fruits_images):  # 0~3 ['yellow', 'red', 'green', 'purple']
    img = qCard.warp

    for i in range(0,4):
        h,w = train_fruits_images[i].img.shape[:2]
        res = cv2.matchTemplate(img, train_fruits_images[i].img, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= 0.6)  # 准确度
        f = set()
        for pt in zip(*locs[::-1]):
            right_bottom = (pt[0] + w, pt[1] + h)
            cv2.rectangle(img, pt, right_bottom, (0, 0, 255), 2)
            sensitivity = 100
            f.add((round(pt[0]/sensitivity), round(pt[1]/sensitivity)))
        if i == 0:
            yellowCount = len(f)
        elif i == 1:
            redCount = len(f)
        elif i == 2:
            greenCount = len(f)
        elif i == 3:
            purpleCount = len(f)
            # 数水果
    cv2.namedWindow("Debugger", cv2.WINDOW_NORMAL)
    cv2.imshow("Debugger", img)
    return yellowCount, redCount, greenCount, purpleCount

# for debug
#        for loc in zip(*locs[::-1]):
#            cv2.rectangle(img, loc, (loc[0] + w, loc[1] + h), (0, 0, 255), 3)

def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)
    cv2.putText(image,(str(qCard.yellowCount) + ' Bananas'),(x-60,y-10),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(str(qCard.yellowCount) + ' Bananas'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,(str(qCard.redCount) + ' Starwberries'),(x-60,y+20),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(str(qCard.redCount) + ' Starwberries'),(x-60,y+20),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,(str(qCard.greenCount) + ' Lemons'),(x-60,y+50),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(str(qCard.greenCount) + ' Lemons'),(x-60,y+50),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,(str(qCard.purpleCount) + ' Grapes'),(x-60,y+80),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(str(qCard.purpleCount) + ' Grapes'),(x-60,y+80),font,1,(50,200,200),2,cv2.LINE_AA)
    # Draw card name twice, so letters have black outline

    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #s_diff = str(qCard.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    
    s = np.sum(pts, axis = 2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Need to create an array listing points in order of
    # [top left, top right, bottom right, bottom left]
    # before doing the perspective transform

    if w <= 0.8*h: # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)

        

    return warp
