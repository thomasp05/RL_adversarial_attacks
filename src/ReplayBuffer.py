# Replay buffer class (code adapted from assigmnent 2)

import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, seed=None):
        self.__buffer_size = buffer_size
        self.data = [] 
        self.buffer_len = 1

        # fix the seed if provided 
        np.random.seed(seed)

    def store(self, element):
        """
        Stores an element. If the replay buffer is already full, deletes the oldest
        element to make space.
        """
        # insert element into buffer
        self.data.append(element)
        
        # update the index 
        self.buffer_len += 1
        if(self.buffer_len > self.__buffer_size): 
            self.buffer_len = self.__buffer_size
            del self.data[0]

    def get_batch(self, batch_size):
        """
        Returns a list of batch_size elements from the buffer.
        """
        ind = np.random.randint(0, self.buffer_len-1, batch_size)
        temp = [] 
        for i in range(batch_size): 
            temp.append(self.data[ind[i]])
        return temp    
