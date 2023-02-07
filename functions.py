""" Pytho Functions """

import numpy as np
import tensorflow as tf
import matplotlib.image as mpig

def Process_image(file):
    # 1. Deteremines if image is filepath or pixel values
    try:
        # if sum of pixel is more than 1,then we know its a pixel
        if np.sum(file) > 1:
            image = file
            # if image dimension is less than 3 (i.e == 2) we will need to add and extra dimension for channels (Black&White)
            if image.ndim < 3:
                image = image[:,:,np.newaxis]
    except Exception:
        # if image is filepath, we read it, which converts it to pixel values
        image = mpig.imread(file)
    
   
    
    # 2. Determines if maximum pixel is 1 or 255
    if image.max() == 1.0:
        image = np.where(image==0,1,0)
    else:
        image = np.where(image<128,1,0)
        
    # 3. Handles Image Segmentation
    idx = []
    imagelist = []
    finallist = []
    for i in range(image.shape[1]-2):
        if np.sum(image[:,i,0])!=0:
            idx.append(i)
            if np.sum(image[:,i+1,0])==0:
                imagelist.append(image[:,idx[0]:idx[-1],0])
            else:
                continue

        elif np.sum(image[:,i,0])==0:
            idx=[]

    
    # 4. Deletes zero-value arrays, checks if sum of pixel in each image is at least 500 and trims the pixels-horizontally
    for i in range(len(imagelist)):
        #print(np.sum(imagelist[i]))
        if np.sum(imagelist[i]) == 0 or np.sum(imagelist[i])<100:
            continue
        else:
            imagelist[i] = imagelist[i][(imagelist[i]!=0).any(axis=1)]
            finallist.append(imagelist[i])
                
    
    # 5. Handles padding, adding extra dimension and resizes the image so it's suitable for the model
    for i in range(len(finallist)):
        width = finallist[i].shape[0]
        heigth = finallist[i].shape[1]
        
        if width > heigth:
            pad_size = int((width - heigth)/2) + 10
            finallist[i] = np.pad(finallist[i], ((10,10),(pad_size, pad_size)), 
                                  mode='constant')
        else:
            pad_size = int((heigth - width)/2) + 10
            finallist[i] = np.pad(finallist[i], ((pad_size,pad_size),(10, 10)), 
                                  mode='constant')
        finallist[i] = finallist[i][:,:,np.newaxis]
        finallist[i] = tf.image.resize(finallist[i], (55,55)).numpy()


    # 6. converts pixels to black and white
    for i in range(len(finallist)):
        finallist[i] = np.where(finallist[i]==0,0,1)
    
    return np.array(finallist)


"==========================================================================="



def Calculate(eq_list):
    import operator
    from string import punctuation
    
    # list of acceptable operators for this project
    ops = {
    '+' : operator.add,
    '-' : operator.sub,
    '*' : operator.mul,
    '/' : operator.truediv,
    '%' : operator.mod,
    '^' : operator.xor,
    }
    
    punct_list = []
    string_ = "".join(eq_list)
    
    # finding all operators in the image/list
    for char in eq_list:
        if char in list(punctuation):
            punct_list.append(char)
        elif char == 'X' or char == 'times':
            punct_list.append(char)

    
    # if/else statement with try-except block to evaluate images and handle exceptions/errors
    if len(punct_list) == 2:
        try:
            lhs, (mid, rhs) = string_.split(punct_list[0])[0], (string_.split(punct_list[0])[1].split(punct_list[1])[0], string_.split(punct_list[0])[1].split(punct_list[1])[1])
            lhs, mid, rhs = int(lhs), int(mid), int(rhs)
            if 'X' in punct_list or 'times' in punct_list:
                punct_list = np.where((np.array(punct_list)=='X')|(np.array(punct_list)=='times'), '*', np.array(punct_list)).tolist()
            a = ops[punct_list[0]](lhs,mid)
            solution = ops[punct_list[1]](a, rhs)
            equation = f"{lhs} {punct_list[0]} {mid} {punct_list[1]} {rhs}"
        except Exception:
            print('Please Look at the Instructions and Examples to input the appropriate equation.\nThanks!')

    elif len(punct_list) == 1:
        try:
            lhs, rhs = string_.split(punct_list[0])[0], string_.split(punct_list[0])[1]
            lhs, rhs = int(lhs), int(rhs)
            if 'X' in punct_list or 'times' in punct_list:
                punct_list = np.where((np.array(punct_list)=='X')|(np.array(punct_list)=='times'), '*', np.array(punct_list)).tolist()
            solution = ops[punct_list[0]](lhs, rhs)
            equation = f"{lhs} {punct_list[0]} {rhs}"
        except Exception:
            print('Please Look at the Instructions and Examples to input the appropriate equation.\nThanks!')

    else:
        print('Please Look at the Instructions and Examples to input the appropriate equation.\nThanks!')
    
    
    # try-except block to handle UboundLocalError that might arise due to no available solution 
    # i.e(executed except blocks above)
    try:
        return solution, equation
    except UnboundLocalError:
        pass
    
