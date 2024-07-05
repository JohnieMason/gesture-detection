import socket
import RPi.GPIO as GPIO
import time

#setting the BCM GPIO Numbering
GPIO.setmode(GPIO.BCM)
#Setting the PWM Pin as pin 18
Servo_Pin=18
Solenoid_Pin=24
pushButton = 17
buzzer = 22

next_time = time.time()
hold_duration = 2

door_state = False
func_exit = False

#setting the servo_pin as the output
GPIO.setup(Servo_Pin, GPIO.OUT)
GPIO.setup(Solenoid_Pin, GPIO.OUT)
GPIO.setup(buzzer, GPIO.OUT)
GPIO.setup(pushButton, GPIO.IN, GPIO.PUD_UP)
#Setting the PWM frequency from 20ms period
pwm=GPIO.PWM(Servo_Pin, 50)
pwm.start(2.0)

def processAction(gesture) -> bool:
    global door_state
    if gesture=="du":
        print("Door Opening")
        pwm.ChangeDutyCycle(7.8)
        door_state = True
        return True
    elif gesture=="ud":
        print("Door Closing")
        pwm.ChangeDutyCycle(2.0)
        door_state = False
        return True
    return False

def assist():
    global next_time
    global door_state
    global fails
    if time.time() <= next_time:
        return False
    if not GPIO.input(pushButton): # when the override button is pressed
        GPIO.output(buzzer, GPIO.LOW) # turn off the alarm
        fails = 0
        if door_state: # if the door is already open, close it
            print("Door Closing")
            pwm.ChangeDutyCycle(2.0)
            door_state = False
        else: # if the door is already closed, open it
            print("Door Opening")
            pwm.ChangeDutyCycle(7.8)
            door_state = True
    return True

try:
    #creating the socket
    soc=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect(("192.168.137.1", 12345)) # connecting to the server
    soc.settimeout(0.001) #set timeout into 0.001 second
    ful="" #Holds the data we receive
    fails = 0
    while True:
        try:
            mystr=soc.recv(1).decode() #Receiving 1 byte otherwise timeout
            if len(mystr)==0:
                break #Socet has disconnected,so exit
            if mystr=="\n": #end of received messsage
                print(ful)
                if processAction(ful):
                    fails = 0
                    GPIO.output(buzzer, GPIO.LOW) # if the person succeeds in using a gesture, turn off the alarm
                else:
                    fails += 1
                if 3 < fails:
                    GPIO.output(buzzer, GPIO.HIGH) # more than three fails, set off the alarm
                ful="" #Reset the message
                if assist():
                    next_time = time.time() + hold_duration
                continue #ignore next lines
            ful += mystr #Add received byte to full message
        except socket.timeout:
            if assist():
                next_time = time.time() + hold_duration
            pass #ignore timeout
finally:
    pwm.ChangeDutyCycle(0)
    GPIO.cleanup()
    func_exit = True
    soc.close()
    exit()