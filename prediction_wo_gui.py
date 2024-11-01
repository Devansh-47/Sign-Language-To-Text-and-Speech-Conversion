import math
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from keras.models import load_model
import traceback

model = load_model('/cnn8grps_rad1_model.h5')
white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("C:\\Users\\devansh raval\\PycharmProjects\\pythonProject\\white.jpg", white)

capture = cv2.VideoCapture(0)

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29
step = 1
flag = False
suv = 0


def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def distance_3d(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2) + ((x[2] - y[2]) ** 2))


bfh = 0
dicttt=dict()
count=0
kok=[]

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands = hd.findHands(frame, draw=False, flipType=True)
        print(frame.shape)
        if hands:
            # #print(" --------- lmlist=",hands[1])
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            white = cv2.imread("C:\\Users\\devansh raval\\PycharmProjects\\pythonProject\\white.jpg")
            # img_final=img_final1=img_final2=0
            handz = hd2.findHands(image, draw=False, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                # x1,y1,w1,h1=hand['bbox']

                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), (pts[t + 1][0] + os, pts[t + 1][1] + os1),
                             (0, 255, 0), 3)
                cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1),
                         (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0),
                         3)
                cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0),
                         3)

                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)

                cv2.imshow("2", white)
                # cv2.imshow("5", skeleton5)

                # #print(model.predict(img))
                white = white.reshape(1, 400, 400, 3)
                prob = np.array(model.predict(white)[0], dtype='float32')
                ch1 = np.argmax(prob, axis=0)
                prob[ch1] = 0
                ch2 = np.argmax(prob, axis=0)
                prob[ch2] = 0
                ch3 = np.argmax(prob, axis=0)
                prob[ch3] = 0


                pl = [ch1, ch2]

                #condition for [Aemnst]
                l=[[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]
                if pl in l:
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                        ch1=0
                        #print("00000")

                #condition for [o][s]
                l=[[2,2],[2,1]]
                if pl in l:
                    if (pts[5][0] < pts[4][0] ):
                        ch1=0
                        print("++++++++++++++++++")
                        #print("00000")



                #condition for [c0][aemnst]
                l=[[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
                pl=[ch1,ch2]
                if pl in l:
                    if (pts[0][0]>pts[8][0] and pts[0][0]>pts[4][0] and pts[0][0]>pts[12][0] and pts[0][0]>pts[16][0] and pts[0][0]>pts[20][0]) and pts[5][0] > pts[4][0]:
                        ch1=2
                        #print("22222")

                # condition for [c0][aemnst]
                l = [[6,0],[6,6],[6,2]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[8],pts[16])<52:
                        ch1 = 2
                        #print("22222")


                ##print(pts[2][1]+15>pts[16][1])
                # condition for [gh][bdfikruvw]
                l = [[1,4],[1,5],[1,6],[1,3],[1,0]]
                pl = [ch1, ch2]

                if pl in l:
                    if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1]<pts[20][1] and pts[0][0]<pts[8][0] and pts[0][0]<pts[12][0] and pts[0][0]<pts[16][0] and pts[0][0]<pts[20][0]:
                        ch1 = 3
                        print("33333c")


                #con for [gh][l]
                l=[[4,6],[4,1],[4,5],[4,3],[4,7]]
                pl=[ch1,ch2]
                if pl in l:
                    if pts[4][0]>pts[0][0]:
                        ch1=3
                        print("33333b")

                # con for [gh][pqz]
                l = [[5, 3],[5,0],[5,7], [5, 4], [5, 2],[5,1],[5,5]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[2][1]+15<pts[16][1]:
                        ch1 = 3
                        print("33333a")

                # con for [l][x]
                l = [[6, 4], [6, 1], [6, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[4],pts[11])>55:
                        ch1 = 4
                        #print("44444")

                # con for [l][d]
                l = [[1, 4], [1, 6],[1,1]]
                pl = [ch1, ch2]
                if pl in l:
                    if (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                        ch1 = 4
                        #print("44444")

                # con for [l][gh]
                l = [[3, 6], [3, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[4][0]<pts[0][0]):
                        ch1 = 4
                        #print("44444")

                # con for [l][c0]
                l = [[2, 2], [2, 5],[2,4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[1][0] < pts[12][0]):
                        ch1 = 4
                        #print("44444")

                # con for [l][c0]
                l = [[2, 2], [2, 5], [2, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[1][0] < pts[12][0]):
                        ch1 = 4
                        #print("44444")

                # con for [gh][z]
                l = [[3, 6],[3,5],[3,4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]) and pts[4][1]>pts[10][1]:
                        ch1 = 5
                        print("55555b")



                # con for [gh][pq]
                l = [[3,2],[3,1],[3,6]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[4][1]+17>pts[8][1] and pts[4][1]+17>pts[12][1] and pts[4][1]+17>pts[16][1] and pts[4][1]+17>pts[20][1]:
                        ch1 = 5
                        print("55555a")

                # con for [l][pqz]
                l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[4][0]>pts[0][0]:
                        ch1 = 5
                        #print("55555")

                # con for [pqz][aemnst]
                l = [[0, 2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[0][0]<pts[8][0]  and  pts[0][0]<pts[12][0]  and pts[0][0]<pts[16][0]  and pts[0][0]<pts[20][0]:
                        ch1 = 5
                        #print("55555")



                # con for [pqz][yj]
                l = [[5, 7],[5,2],[5,6]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[3][0]<pts[0][0]:
                        ch1 = 7
                        #print("77777")

                # con for [l][yj]
                l = [[4, 6],[4,2],[4,4],[4,1],[4,5],[4,7]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[6][1] < pts[8][1]:
                        ch1 = 7
                        #print("77777")

                # con for [x][yj]
                l = [[6, 7],[0,7],[0,1],[0,0],[6,4],[6,6] ,[6,5],[6,1]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[18][1] > pts[20][1]:
                        ch1 = 7
                        #print("77777")


                # condition for [x][aemnst]
                l = [[0,4],[0,2],[0,3],[0,1],[0,6]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[5][0]>pts[16][0]:
                        ch1 = 6
                        #print("66666")

                # condition for [yj][x]
                l = [[7, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[18][1] < pts[20][1]:
                        ch1 = 6
                        #print("66666")


                # condition for [c0][x]
                l = [[2, 1],[2,2],[2,6],[2,7],[2,0]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[8],pts[16])>50:
                        ch1 = 6
                        #print("66666")

                # con for [l][x]

                l = [[4, 6],[4,2],[4,1],[4,4]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[4], pts[11]) < 60:
                        ch1 = 6
                        #print("66666")

                #con for [x][d]
                l = [[1,4],[1,6],[1,0],[1,2]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[5][0] - pts[4][0] - 15 > 0:
                        ch1 = 6


                # con for [b][pqz]
                l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 1
                        print("111111")


                # con for [f][pqz]
                l = [[6, 1],[6,0],[0,3],[6,4],[2,2], [0,6],[6,2],[7, 6],[4,6],[4,1],[4,2], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                                    pts[18][1] > pts[20][1]):
                        ch1 = 1
                        print("111112")

                l = [[6, 1], [6, 0],[4,2],[4,1],[4,6],[4,4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and
                            pts[18][1] > pts[20][1]):
                        ch1 = 1
                        print("111112")

                # con for [d][pqz]
                fg=19
                #print("_________________ch1=",ch1," ch2=",ch2)
                l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                                    pts[18][1] < pts[20][1]) and (pts[2][0]<pts[0][0]) and pts[4][1]>pts[14][1]):
                        ch1 = 1
                        print("111113")

                l = [ [4, 1], [4, 2],[4, 4]]
                pl = [ch1, ch2]
                if pl in l:
                    if (distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 1
                        print("1111993")



                l = [[3, 4], [3, 0], [3, 1], [3, 5],[3,6]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                         pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1]<pts[4][1]):
                        ch1 = 1
                        print("1111mmm3")

                l = [[6, 6],[6, 4], [6, 1],[6,2]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[5][0]-pts[4][0]-15<0:
                        ch1 = 1
                        print("1111140")



                # con for [i][pqz]
                l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7, 5], [7, 1], [7, 6], [7, 7]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                                 pts[18][1] > pts[20][1])):
                        ch1 = 1
                        print("111114")

                # con for [yj][bfdi]
                l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[4][0]<pts[5][0]+15) and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and
                                 pts[18][1] > pts[20][1])):
                        ch1 = 7
                        print("111114lll;;p")

                #con for [uvr]
                l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]
                pl = [ch1, ch2]
                if pl in l:
                    if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and
                         pts[18][1] < pts[20][1])) and pts[4][1]>pts[14][1]:
                        ch1 = 1
                        print("111115")



                # con for [w]
                fg=13
                l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]
                pl = [ch1, ch2]
                if pl in l:
                    if not(pts[0][0]+fg < pts[8][0] and pts[0][0]+fg < pts[12][0] and pts[0][0]+fg < pts[16][0]  and pts[0][0]+fg < pts[20][0]) and not(pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0]  and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
                        ch1 = 1
                        print("111116")

                # con for [w]

                l = [ [5, 0], [5, 5],[0,1]]
                pl = [ch1, ch2]
                if pl in l:
                    if pts[6][1]>pts[8][1] and pts[10][1]>pts[12][1] and pts[14][1]>pts[16][1]:
                        ch1 = 1
                        print("1117")



                #-------------------------condn for 8 groups  ends



                #-------------------------condn for subgroups  starts
                #
                if ch1 == 0:
                    ch1='S'
                    if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                        ch1 = 'A'
                    if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1] :
                        ch1 = 'T'
                    if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                        ch1 = 'E'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and  pts[4][1] < pts[18][1]:
                        ch1 = 'M'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0]  and  pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                        ch1 = 'N'


                if ch1 == 2:
                    if distance(pts[12], pts[4]) > 42:
                        ch1 = 'C'
                    else:
                        ch1 = 'O'

                if ch1 == 3:
                    if (distance(pts[8], pts[12])) > 72:
                        ch1 = 'G'
                    else:
                        ch1 = 'H'

                if ch1 == 7:
                    if distance(pts[8], pts[4]) > 42:
                        ch1 = 'Y'
                    else:
                        ch1 = 'J'

                if ch1 == 4:
                    ch1 = 'L'

                if ch1 == 6:
                    ch1 = 'X'

                if ch1 == 5:
                    if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                        if pts[8][1] < pts[5][1]:
                            ch1 = 'Z'
                        else:
                            ch1 = 'Q'
                    else:
                        ch1 = 'P'

                if ch1 == 1:
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] >pts[20][1]):
                        ch1 = 'B'
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <pts[20][1]):
                        ch1 = 'D'
                    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'F'
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'I'
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'W'
                    if  (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1]<pts[9][1]:
                        ch1 = 'K'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'U'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] >pts[9][1]):
                        ch1 = 'V'

                    if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'R'

                if ch1== 1 or 'E' or 'S' or 'X' or 'Y' or 'B':
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'Space'

                if ch1== 'E' or 'Y' or 'B':
                    if (pts[4][0] < pts[5][0] ):
                        ch1 = 'Next'

                if ch1== 'Next' or 'B' or 'C' or 'H' or 'F':
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[4][1]<pts[8][1] and pts[4][1]<pts[12][1] and pts[4][1]<pts[16][1] and pts[4][1]<pts[20][1]:
                        ch1 = 'Backspace'

                print("ch1=", ch1, " ch2=", ch2, " ch3=", ch3)
                kok.append(ch1)

                # # [0->aemnst][1->bfdiuvwkr][2->co][3->gh][4->l][5->pqz][6->x][7->yj]
                if ch1 != 1:
                    if (ch1,ch2) in dicttt:
                        dicttt[(ch1,ch2)] += 1
                    else:
                        dicttt[(ch1,ch2)] = 1

                frame = cv2.putText(frame, "Predicted " + str(ch1), (30, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    3, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("frame", frame)
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:
            # esc key
            break


    except Exception:
        print("==", traceback.format_exc())



dicttt = {key: val for key, val in sorted(dicttt.items(), key = lambda ele: ele[1], reverse = True)}
print(dicttt)
print(set(kok))
capture.release()
cv2.destroyAllWindows()


