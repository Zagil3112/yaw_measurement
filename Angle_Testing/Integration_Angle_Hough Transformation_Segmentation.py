import sys
import math
import cv2 as cv
import numpy as np

def main(argv):
    tetha = 0
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        src = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Segmentacion 
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Definir un intervalo del color azul en HSV

        ## Prueba 1 
        #lower_verde = np.array([39,130,141])
        #upper_verde = np.array([51,255,255])

        ## Prueba 2 

        lower_verde = np.array([39,96,121])
        upper_verde = np.array([51,255,255])
        

        # Umbralizar la imagen HSV para obtener solo los colores azules
        mask = cv.inRange(hsv, lower_verde, upper_verde)
       

        # Bitwise-AND mask and original image
        frame_seg = cv.bitwise_and(frame,frame, mask= mask)
        
        ## End Segmentacion


        dst = cv.Canny(mask, 50, 200, None, 3)
    
        # Copy edges to the images that will display the results in BGR
        cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
        cdstP = np.copy(cdst)
        
        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
        
        
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, 80, 50, 10)
        
        if linesP is not None:
            for i in range(0, 1):
                l = linesP[i][0]
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
                print("Punto1 :",l[0], l[1], end=' ')
                print("Punto2 :",l[2], l[3], end=' ')
                m = ((l[3]-l[1])/(l[2]-l[0]))
                #print("m =",m)
                tetha = np.rad2deg(np.arctan2((l[3]-l[1]),-(l[2]-l[0])))
                if tetha >= 0:
                    tetha = tetha -90
                else:
                    tetha = tetha +90
                
                print("tetha =",tetha," °")

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(cdstP,str(round(tetha,1))+' deg',(10,400), font, 2,(0,255,0),2,cv.LINE_AA)
        cv.imshow("Source", mask)
        #cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()  
    return 0
    
if __name__ == "__main__":
    main(sys.argv[1:])