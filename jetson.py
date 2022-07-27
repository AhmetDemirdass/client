#!/usr/bin/python3 -W ignore::DeprecationWarning
from os import path
import socket
import time
from json import load

import numpy as np
from utils import find_by_key, data_map

import tensorflow as tf
from keras.models import load_model
from keras import backend 


import timeit


#from tensorflow.python.compiler.tensorrt import trt_convert as trt

print(tf.config.list_physical_devices("GPU"))

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


HOST = "127.0.0.1" 
PORT = 10000
SIZE = 1024
MODEL_PATH = "data/autoencoder_GRU_0.h5"

class communicate:
    def __init__(self) -> None:
        # Create a socket object
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Define the port on which you want to connect
        # connect to the server on local computer

    def connect(self,a) -> bool:
        return self.s.connect_ex(a) 
    
    def receive(self, size) -> str:
        r = self.s.recv(size).decode()
        return r
    def send(self, msg: str):
        self.s.sendall(msg.encode())

    def close(self) -> None:
        self.s.close()


class ml_model():
    def __init__(self, path=MODEL_PATH) -> None:
        self.history = np.empty((1,72,2))
        
        backend.clear_session()
        x_test = np.load("data/X_test.npy")[:,:, [0]]
        x_features = np.load("data/X_test_features.npy")[:,:, [0]]
        
        self.model = load_model(path,compile=False)
        a = np.dstack([np.zeros((30,72,1)),np.random.rand(30,72,1)])
        self.model.predict(a,use_multiprocessing = True)
        #self.model.predict(np.dstack([x_features[1000:1305,:,:], x_test[1000:1305,:,:]]),use_multiprocessing = True)

        
        

        #self.model.compile(optimizer="adam", loss="mae", )
        #self.model.summary()

        pass

    def add(self, arr : np.ndarray):
        """ self.command = (arr[0], arr[1])
        if(self.command == (3322,90)): 
            self.values_size = arr[2]
            if self.values_size != 0:
                self.values = arr[3::2] / (2**31 -1) #engineering value
                self.eclipse = arr[4::2]
                #print(f"Values Size: {self.values.size}, Eclipse Size: {self.eclipse.size}")
                #print(self.values)
                return True
            else:
                return False
        else:
            return False """
        self.value = arr[2] / (2**31 -1) #engineering value
        self.eclipse = arr[3]
        return((self.value, self.eclipse))
           
    

    def predict(self):
        """ self.test_pred = self.model.predict(np.dstack([self.eclipse, self.values]))
        t = np.abs(self.values - self.test_pred).max()
        print("threshold:", t)
        return t """
        self.test_pred = self.model.predict(self.history)
        print(np.abs(self.history[:,:,1] - self.test_pred))
        t = np.abs(self.history[:,:,1] - self.test_pred).max()
        
        print("max value:", t)
        return t


    





# MAIN
def main():
    msg =""
    c = communicate()
    m = ml_model()
    while c.connect((HOST,PORT)):
        pass
    
    history_ctr = 0
    threshold = 0.02
    max_value = 0
    while True:
        msg = c.receive(SIZE)
        if msg != "":
            test_input = np.fromstring(string=msg, dtype=int, sep=',')

            #print(test_input.shape)
            command = (test_input[0], test_input[1])
            
            if command == (3310,1):
                c.send(str(max_value))
                

            elif command == (3322,90):
                val = m.add(test_input)[0]
                eclipse = m.add(test_input)[1]
                
                if history_ctr > 71:
                    m.history = np.roll(m.history, -1, axis=1)
                    m.history[0,71,1] = val
                    m.history[0,71,0] = eclipse
                    print("Value:{}, Eclipse:{}".format(m.history[0,71,1], m.history[0,71,0]))

                    ###Here Prediction timing is calculated###
                    start = timeit.default_timer()
                    max_value = m.predict()
                    print(bcolors.WARNING +"Prediction execution time: {:5.2f} ms".format((timeit.default_timer() - start)* 1000.0)+ bcolors.ENDC)

                    print(m.history)
                else:
                    m.history[0,history_ctr,1] = val
                    m.history[0,history_ctr,0] = eclipse
                history_ctr += 1
                """ start = timeit.default_timer()
                if m.add(test_input):
                    # your code here
                    threshold_last = threshold    
                    threshold = m.predict()
                    if threshold_last != threshold:
                        print(bcolors.WARNING +"Prediction execution time: {:5.2f} ms".format((timeit.default_timer() - start)* 1000.0)+ bcolors.ENDC)
 """
            print("[CLIENT] Connected to Server! History Counter {}".format(history_ctr))
        else:
            print("[CLIENT] Server is waited!")
            time.sleep(1)


if __name__ == "__main__":
    main()   


    


read_msg = ""
read_list = []



""" while True:
        
    a = s.recv(1024).decode()
    #s.send("X".encode('utf-8'))
    print(a)
    #s.shutdown(socket.SHUT_RDWR)
    #s.close()
    time.sleep(2) """
    
    
#test_input = np.fromstring(string=s.recv(1024).decode(), dtype=int, sep=',')
#test_input_processed = np.round_((test_input / (2**32 -1)), decimals=8) #data_map(test_input, 0, 2**32 -1, 0, 1, True)
#print(test_input)

#s.recv(1024).decode()
                    
#s.close()



""" def model_prep():
    # PREPARE the MODEL """
