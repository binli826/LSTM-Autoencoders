# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:22:47 2018

@author: Bin
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import queue
from kafka import KafkaConsumer

import time, threading

batch_num =3
step_num = 4

consumer = KafkaConsumer('kdd99stream',
                         group_id='test-consumer-group',    # defined in consumer.properties file
                         bootstrap_servers=['localhost:9092'])
consumer.poll()
#go to end of the stream
consumer.seek_to_end()


def block_generator2queue(q,stop_event):
    
    while not stop_event.is_set():
        block = []
        for message in consumer:
            if stop_event.is_set():
                break
            row = message.value.decode("utf-8") 
            list_of_str = row.strip(",null,null,null,null,null,null,null,null").split(",")
            list_of_num = [float(n) for n in list_of_str]
            block.append(list_of_num)
            if len(block)>=batch_num*step_num:
                df = pd.DataFrame(np.array(block))
                q.put(df)
                
                print("Wrote a block to queue.")
                block.clear()
                
def read_block_from_queue(q,stop_event):
   
    while not stop_event.is_set():
        if q.empty() == False:
            b = q.get()
            print("read a block from queue")
            print(b)
        else :
            print("queue is empty")
            time.sleep(0.5)
            
def main():
    q = queue.Queue()
    stop_event = threading.Event()
    write = threading.Thread(target=block_generator2queue, name='WriteThread',args=(q,stop_event,),daemon=True)
    read = threading.Thread(target=read_block_from_queue, name='ReadThread',args=(q,stop_event,),daemon=True)
    

    try:
        write.start()
        read.start()
        while 1:
            time.sleep(.1)
    except (KeyboardInterrupt,SystemExit):
        stop_event.set()
        print("Threads closed.")
        
        
if __name__=="__main__":
    main()

    

    
