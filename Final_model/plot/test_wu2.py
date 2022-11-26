import cv2

img = cv2.imread('3.jpg')

#图片信息
img_wide = len(img)
img_long = len(img[0])

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# cv2.imshow('test',hsv)
# cv2.waitKey(0)
ret, dst = cv2.threshold(hsv, 100,255, cv2.THRESH_BINARY)
# cv2.imshow('tst',dst)
# cv2.waitKey(0)
img_gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

#print(img_gray)

#取边界框
box = []
#正序取一半表框
for x in range(len(img_gray)):
    for y in range(len(img_gray[0])):
        boxx = []
        if img_gray[x][y] == 255:
            boxx.append(x)
            boxx.append(y)
            box.append(boxx)
            break
flag = len(box)
# print(len(box))
# print(box[0:90])
for x in range(len(img_gray)-1,0,-1):
    for y in range(len(img_gray[0])-1,0,-1):
        boxx = []
        if img_gray[x][y] == 255:
            boxx.append(x)
            boxx.append(y)
            box.append(boxx)
            break


def pic_corner(box,flag):
    flag_leftop_rightbottom = False
    flag_rightop_leftbottom =False

    left_top = []
    left_bottom = []
    right_top = []
    right_bottom = []

    if box[0][0] < box[-1][-1]:
        #左上右下形式
        distance_right_top = 99999
        real_coordinary_righttop = []
        #右上
        for x in box:
            real_d = (x[0] - 0) ** 2 + (x[-1] - img_long) ** 2
            if real_d < distance_right_top:
                distance_right_top = real_d
                real_coordinary_righttop = x
        #左下
        distance_letf_bottom = 99999
        real_coordinary_left_bottom = []
        for y in box:
            real_d = (y[0] - img_wide) ** 2 + (y[-1] - 0) ** 2
            if real_d < distance_letf_bottom:
                distance_letf_bottom = real_d
                real_coordinary_left_bottom = y
        flag_leftop_rightbottom = True
        #记录坐标
        # left_top.append(box[0][-1])
        # left_top.append(box[0][0])
        # right_bottom.append(box[flag][-1])
        # right_bottom.append(box[flag][0])
        left_top = box[0]
        right_bottom = box[flag]
        left_bottom = real_coordinary_left_bottom
        right_top = real_coordinary_righttop
        return left_top, left_bottom, right_top, right_bottom
    else:
        #右上左下形式
        distance_right_bottom = 99999
        real_coordinary_rightbottom = []
        # 右下
        for x in box:
            real_d = (x[0] - img_wide) ** 2 + (x[-1] - img_long) ** 2
            if real_d < distance_right_bottom:
                distance_right_bottom = real_d
                real_coordinary_rightbottom = x

        # 左上
        distance_letf_top = 99999
        real_coordinary_left_top = []
        for y in box:
            real_d = (y[0] - img_wide) ** 2 + (y[-1] - 0) ** 2
            if real_d < distance_letf_top:
                distance_letf_top = real_d
                real_coordinary_left_top = y
        flag_rightop_leftbottom =True
        left_top = real_coordinary_left_top
        right_bottom = real_coordinary_rightbottom
        left_bottom = box[flag]
        right_top = box[0]

        return left_top,left_bottom,right_top,right_bottom


if __name__ == '__main__':
    left_top, left_bottom, right_top, right_bottom = pic_corner(box,flag)
    cv2.circle(img_gray,(left_top[-1],left_top[0]),5,(0,0,0),-1)
    cv2.circle(img_gray, (left_bottom[-1], left_bottom[0]), 5, (0, 0, 0), -1)
    cv2.circle(img_gray, (right_top[-1], right_top[0]), 5, (0, 0, 0), -1)
    cv2.circle(img_gray, (right_bottom[-1], right_bottom[0]), 5, (0, 0, 0), -1)
    cv2.imshow('tst',img_gray)
    cv2.waitKey(0)









