#!/usr/bin/env python
"""grippers.py: Control WSG-50 gripper from Python.

Author:
    Achu Wilson - Nov 4 2021
    achuw@andrew.cmu.edu

Version
    0.1.0 - Nov 07 2021

License:
    MIT License
"""

import socket
from time import time, sleep
from enum import Enum
import atexit
from signal import signal, SIGPIPE, SIG_DFL
#Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
signal(SIGPIPE,SIG_DFL) 

class WSG50:
    '''
    Log in to the gripper config using the browser at 192.168.1.20
    Settings> Command Interface
            Interface: TCP/IP
            Use Text based Interface : Enabled
            TCP Port: 1000
            Enable Error on TCP Disconenct : True
            Enable CRC : False ( since we are using TCP)


    '''
    def __init__(self, TCP_IP = "192.168.1.20", TCP_PORT = 1000):
        self.TCP_IP = TCP_IP
        self.TCP_PORT = TCP_PORT 
        self.BUFFER_SIZE = 1024
        self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_sock.connect((TCP_IP, TCP_PORT))
        self.timeout = 15
        atexit.register(self.__del__)
        # Acknowledge fast stop from failure if any
        self.ack_fast_stop()

    def wait_for_msg(self, msg):
        since = time()
        while True:
            data = self.tcp_sock.recv(self.BUFFER_SIZE)
            if data == msg:
                ret = True
                break
            elif data.decode("utf-8").startswith("ERR"):
                ret = False
                print("ERROR ", data)
                break
            if time() - since >= self.timeout:
                ret =False
                print("TIMEOUT")
                break
            sleep(0.1)
        return ret

    def ack_fast_stop(self):
        msg = str.encode("FSACK()\n")
        self.tcp_sock.send(msg)
        return self.wait_for_msg(b"ACK FSACK\n")


    def homing(self):
        """
        Fully open the gripper
        """
        cmd="HOME()\n"
        msg = bytes(str(cmd).encode("ascii"))
        self.tcp_sock.send(msg)
        return self.wait_for_msg(b"FIN HOME\n")

    def move(self, position, velocity = 50):
        """
        Do not use the MOVE command to grip or release parts. Use the GRIP and RELEASE command instead.

        position: 0 to 110
        """
        #msg = str.encode(f"MOVE({position})\n")
        cmd = "MOVE("+str(position)+','+str(velocity)+")\n"
        msg = bytes(str(cmd).encode("ascii"))
        self.tcp_sock.send(msg)
        return self.wait_for_msg(b"FIN MOVE\n")

    def open(self):
        '''
        just open the gripper fully
        '''
        return self.move(110,250)

    def grip(self, force=10, position=0, velocity=50):
        '''
        force = 5 - 80N
        position = 0 - 110 mm : the expected closing position at which object is to be found, give 0 if unsure
        speed = 5 - 420 mm/s

        '''
        cmd = "GRIP("+str(force)+','+str(position)+','+str(velocity)+")\n"
        msg = bytes(str(cmd).encode("ascii"))
        self.tcp_sock.send(msg)
        return self.wait_for_msg(b"FIN GRIP\n")

    def release(self, distance=5):
        '''
        Release: Release object by opening fingers by 5 mm.
        '''
        cmd = "RELEASE("+str(distance)+")\n"
        msg = bytes(str(cmd).encode("ascii"))
        self.tcp_sock.send(msg)
        return self.wait_for_msg(b"FIN RELEASE\n")

    def bye(self):
        msg = str.encode("BYE()\n")
        self.tcp_sock.send(msg)
        return

    def __del__(self):
        self.bye()
