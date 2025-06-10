import time
import cv2
import threading
import numpy as np
from robomaster import robot
from robomaster import camera
from ultralytics import YOLO


model = YOLO("D:\\241-251\\RoboMaster-SDK\\examples\\ass\mini_project3\\the_best.pt")

def frontSensor(sub_info) :
    global front_sensor
    front_sensor = sub_info[0]/10 
    return front_sensor

def irSensors(adc):   
    if adc == 0:
        adc = 0.1
    distance_cm = ((13*623)/(adc*2.2))+0.42
    # print(f'adc = {distance_cm}')
    return distance_cm

def center_of_camera(img):
    global center_x, center_y
    # height, width, _ = img.shape
    height = img.shape[0]
    width = img.shape[1]
    center_x = width // 2
    center_y = height // 2
    cv2.circle(img, (center_x, center_y), 1, (0, 0, 255), 2)
    cv2.putText(img, f'({center_x},{center_y})', (center_x-30, center_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.circle(img, (center_x, center_y), 1, (0, 0, 255), 2) # center
    # cv2.circle(img, (455, 360), 1, (0, 0, 255), 2) # left
    # cv2.circle(img, (825, 360), 1, (0, 0, 255), 2) # right

def green_color(frame):
    # frame = ep_camera.read_cv2_image(strategy="newest")
    global xg,yg,wg,hg,o_chick
    o_chick = frame
    hsv_chick = cv2.cvtColor(o_chick, cv2.COLOR_BGR2HSV)
    lower_green = np.array([66, 125, 69])  # ค่าสีต่ำสุดสำหรับสีเขียว
    upper_green = np.array([110, 255, 245])  # ค่าสีสูงสุดสำหรับสีเขียว
    green_mask = cv2.inRange(hsv_chick, lower_green, upper_green)
    clean_chick = cv2.bitwise_and(o_chick, o_chick, mask=green_mask)
    # ทำขอบสีเขียวรอบสิ่งของที่ใหญ่ที่สุด
    gray_chick = cv2.cvtColor(clean_chick, cv2.COLOR_BGR2GRAY)
    _, c_threshold = cv2.threshold(gray_chick, 1, 255, cv2.THRESH_BINARY)
    contours, Hierarchy = cv2.findContours(c_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # หากรอบที่ใหญ่ที่สุด
    max_area = 0
    max_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_contour = cnt

    if max_contour is not None:
        xg, yg, wg, hg = cv2.boundingRect(max_contour)
        cv2.rectangle(o_chick, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 1)
    return o_chick , (xg, yg, wg, hg)
    # cv2.destroyAllWindows()
    # ep_camera.stop_video_stream()
    # ep_robot.close()


def release_chicken():
    green_box =  green_color(frame)
    _, (xg, yg, wg, hg) = green_color(frame)
    speed = 20
    target = 35
    if xg is not None and yg is not None and wg is not None and hg is not None:
        if xg + wg / 2 < hg + yg / 2:
            ep_chassis.drive_wheels(w1=speed , w2=speed, w3=speed , w4=speed)
            if front_sensor < target:
                 ep_chassis.drive_wheels(w1=0 , w2=0, w3=0 , w4=0)
        else :
            pass




      








def show_pic():
    global boxlist , xd , frame
    
    while True:
        frame = ep_camera.read_cv2_image(strategy="newest")

        if frame is not None :
            pre = model.predict(frame, verbose = False , conf = 0.6)
            results = pre[0]
            boxlist = results.boxes
            cls = boxlist.cls
            coordinates = boxlist.xyxy
            for i in range(len(cls)) :
                    if cls[i] == 1 :
                        if state == 0:
                            xd = int(coordinates[i][0])
                            y = int(coordinates[i][1])
                            w = int(coordinates[i][2] - xd)
                            h = int(coordinates[i][3] - y)
                            x_center_chicken = (w // 2) + xd
                            y_center_chicken = (h // 2) + y
                            cv2.rectangle(frame, (xd, y), (xd + w, y + h), (0, 0, 255), 2)
                            cv2.circle(frame,(x_center_chicken, y_center_chicken), 1, (0, 0, 255), 2)
                        else :
                            xd = 0
                    if cls[i] == 0 :
                        xx = int(coordinates[i][0])
                        yy = int(coordinates[i][1])
                        ww = int(coordinates[i][2] - xx)
                        hh = int(coordinates[i][3] - yy)
                        xx_center_chicken = (ww // 2) + xx
                        yy_center_chicken = (hh // 2) + yy
                        cv2.rectangle(frame, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
                        cv2.circle(frame,(xx_center_chicken, yy_center_chicken), 1, (0, 255, 0 ), 2)
                        cv2.putText(frame, f'({xx},{yy},{ww},{hh},{xx_center_chicken},{yy_center_chicken})', (xx_center_chicken-30, yy_center_chicken+12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            frame = green_color(frame)


        center_of_camera(frame)
        cv2.imshow('Image with Bounding Box', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    cv2.destroyAllWindows()
    ep_camera.stop_video_stream()
    ep_robot.close()

def dead_chick_var():
    global x_dead, y_dead, w_dead, h_dead, x_center_chicken_dead, y_center_chicken_dead 
    comparative_val = 0
    lst_dead = []

    for z in boxlist :
        if z.cls == 1 :
            var_box1 = z.xyxy
            var_box1 = var_box1.tolist()
            lst_dead.append(var_box1)

    for dead in lst_dead :
        if int(dead[0][0]) > comparative_val :
            comparative_val = int(dead[0][0])
            var_chick_dead = dead[0]

    x_dead = int(var_chick_dead[0])
    y_dead = int(var_chick_dead[1])
    w_dead = int(var_chick_dead[2] - x_dead)
    h_dead = int(var_chick_dead[3] - y_dead)
    x_center_chicken_dead = (w_dead // 2) + x_dead
    y_center_chicken_dead = (h_dead // 2) + y_dead 

def alive_chick_var():
    global x_alive, y_alive, w_alive, h_alive, x_center_chicken_alive, y_center_chicken_alive
    comparative_val1 = 0
    lst_alive = []

    for zz in boxlist :
        if zz.cls == 0  :
            var_box2 = zz.xyxy
            var_box2 = var_box2.tolist()
            lst_alive.append(var_box2)
        else :
            print('nonono')

    for alive in lst_alive :
        if int(alive[0][3]) > comparative_val1 :
            comparative_val1 = int(alive[0][3])
            list_var_chick_alive = alive[0]

    if list_var_chick_alive is not None :
        x_alive = int(list_var_chick_alive[0])
        y_alive = int(list_var_chick_alive[1])
        w_alive = int(list_var_chick_alive[2] - x_alive)
        h_alive = int(list_var_chick_alive[3] - y_alive)
        x_center_chicken_alive = (w_alive // 2) + x_alive
        y_center_chicken_alive = (h_alive // 2) + y_alive
    else :
        print('nononno')



def x_motor_distance(camera_x, chicken_x, camera_y, chicken_y):
    global error_x, error_y
    speed = 25
    error_x = round(camera_x - chicken_x)
    error_y = round(chicken_y - camera_y)
    
    if -10 < error_x < 10: 
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.01)
    elif error_x > 0:
        print(f'r')
        ep_chassis.drive_wheels(w1=speed, w2=-speed , w3=speed, w4=-speed )
        time.sleep(0.1)
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    elif error_x < 0:
        ep_chassis.drive_wheels(w1=-speed , w2=speed, w3=-speed , w4=speed)
        print(f'l')
        time.sleep(0.1)
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
    time.sleep(0.001)

def y_motor_distance(servo1, servo2):
    global servo_1, servo_2
    servo_1 = servo1
    servo_2 = servo2
    
    if error_y > 3 :
        servo_2 += 2
        servo_1 += 2 
    if error_y < -3 :
        servo_2 -= 2
        servo_1 -= 2
    if servo_1 > 10 :
        servo_1 = 10
    if servo_2 > -65 :
        servo_2 = -65

    ep_servo.moveto(index=1, angle=servo_1).wait_for_completed()
    ep_servo.moveto(index=2, angle=servo_2).wait_for_completed()
    print(f'servo_1 = {servo_1}')
    print(f'servo_2 = {servo_2}')

def keep_dead_chick():
    global x_dead,state
    speed = 20
    target = 35

    if x_dead is not None :
        if servo_1 == 10 and servo_2 == -65 and xd != 0 :
        # if servo_1 == 10 and servo_2 == -65 :
            ep_servo.moveto(index=2, angle=-85).wait_for_completed()
            ep_servo.moveto(index=1, angle=15).wait_for_completed()
            ep_chassis.move(x=0.15, y=0, z=0, xy_speed=0.7).wait_for_completed()
            ep_gripper.close(power=50)
            time.sleep(1)
            ep_gripper.pause()
            ep_servo.moveto(index=1, angle=0).wait_for_completed()
            ep_servo.moveto(index=2, angle=0).wait_for_completed()
            state = 1
            # if  ep_gripper.close(power=50):
            #     ep_servo.moveto(index=1, angle=0).wait_for_completed()
            #     ep_servo.moveto(index=2, angle=0).wait_for_completed()
            ep_chassis.move(x=0, y=0, z=180, xy_speed=20).wait_for_completed()
           
        

            
        else :
            print(f'g')
            x_motor_distance(center_x, x_center_chicken_dead, center_y, y_center_chicken_dead)
            if -10 < error_x < 10 and xd != 0:
                if error_y < 20 :
                    ep_chassis.drive_wheels(w1=speed , w2=speed, w3=speed , w4=speed)
                    time.sleep(0.1)
                    ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                if error_y > 15 :
                    y_motor_distance(servo_1, servo_2)
                    

def wall():
    global front_sensor
    speed = 15
    target = 40
    print(f'front Distance {front_sensor}')
    if front_sensor <= target:
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
        time.sleep(0.5)
        # ep_chassis.move(x=0, y=90, z=0, xy_speed=0.7).wait_for_completed()
    else: 
        # print(f'distance{distance}')
        ep_chassis.drive_wheels(w1=speed, w2=speed, w3=speed, w4=speed)

def avoid_alive_chick(center_chick_alive):
    errorx_left = 420 - center_chick_alive
    errorx_right = 860 - center_chick_alive
    speed = 20
    if 85 <= h_alive < 89 :
        if abs(errorx_left) < abs(errorx_right) :
            if -10 < errorx_left < 10: 
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                time.sleep(0.01)
            elif errorx_left > 0:
                ep_chassis.drive_wheels(w1=speed, w2=-speed, w3=speed, w4=-speed)
                time.sleep(0.1)
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            elif errorx_left < 0:
                ep_chassis.drive_wheels(w1=-speed, w2=speed, w3=-speed, w4=speed)
                time.sleep(0.1)
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            time.sleep(0.001)
            print(f'error left : {errorx_left}')

        elif abs(errorx_left) > abs(errorx_right) :
            if -10 < errorx_right < 10: 
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
                time.sleep(0.01)
            elif errorx_right > 0:
                ep_chassis.drive_wheels(w1=speed, w2=-speed, w3=speed, w4=-speed)
                time.sleep(0.1)
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            elif errorx_right < 0:
                ep_chassis.drive_wheels(w1=-speed, w2=speed, w3=-speed, w4=speed)
                time.sleep(0.1)
                ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)
            time.sleep(0.001)
            print(f'error right : {errorx_right}')
    else :
        ep_chassis.drive_wheels(w1=speed , w2=speed, w3=speed , w4=speed)
        time.sleep(0.1)
        ep_chassis.drive_wheels(w1=0, w2=0, w3=0, w4=0)

if __name__ == '__main__':
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    ep_camera = ep_robot.camera
    ep_chassis = ep_robot.chassis 
    ep_sensor = ep_robot.sensor 
    ep_servo = ep_robot.servo
    ep_gripper = ep_robot.gripper
    ep_sensor_adaptor = ep_robot.sensor_adaptor
    ep_sensor.sub_distance(freq=5, callback=frontSensor)
    ep_camera.start_video_stream(display = False)
    right = ep_sensor_adaptor.get_adc(id=1, port=2)
    left = ep_sensor_adaptor.get_adc(id=2, port=2)
    state = 0
    show1 = threading.Thread(target=show_pic)
    show1.start()
    time.sleep(3)
    servo_1 = -5
    servo_2 = -80
    ep_servo.moveto(index=1, angle=0).wait_for_completed()
    ep_servo.moveto(index=2, angle=0).wait_for_completed()
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()
    
    while True :
        release_chicken()
    #     green_color(frame)
        # if state == 1:
        #      green_color
        #      pass
        # if state == 0:

    #         pass
     #alive_chick_var()
        # dead_chick_var()
        # green_color()
        # avoid_alive_chick(x_center_chicken_alive)
        # keep_dead_chick()
        # wall()