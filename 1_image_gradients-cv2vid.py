import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_x = cv2.VideoWriter('sobel_x.avi', fourcc, 20.0, (640, 480))
out_y = cv2.VideoWriter('sobel_y.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = 9
        sobelx_64 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=s)
        sobely_64 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=s)

        out_x.write(sobelx_64)
        out_y.write(sobely_64)

        cv2.imshow('sobel_x.avi', sobelx_64)
        cv2.imshow('sobel_y.avi', sobely_64)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out_x.release()
out_y.release()
cv2.destroyAllWindows()
