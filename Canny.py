import cv2
import matplotlib.pyplot as plt
import numpy as np


def flip180(arr):
    new_arr = arr.reshape(arr.size)
    new_arr = new_arr[::-1]
    new_arr = new_arr.reshape(arr.shape)
    return new_arr

def twodConv(f, W,method='zero'):
    print(f.shape)
    h,w = f.shape
    W = flip180(W)
    k1,k2 = W.shape
    p1,p2 = int((k1-1)/2),int((k2 - 1) / 2)
    padded_img = np.zeros((h+2*p1,w+2*p2),dtype = f.dtype)
    result_img = np.zeros_like(f)
    padded_img[p1:h+p1,p2:w+p2] = f
    if method == 'zero':
        padded_img = padded_img
    elif method == 'replicate':
        padded_img[0:p1,p2:p2+w] = f[0,:]
        padded_img[p1+h:p1*2+h,p2:p2+w] = f[h-1,:]
        for i in range(p2):
            padded_img[p1:p1+h,i] = f[:,0]
            padded_img[p1:p1+h,p2+w+i] = f[:,w-1]
        padded_img[0:p1,0:p2] =  f[0,0]
        padded_img[0:p1,p2+w:p2*2 +w] = f[0,w-1]
        padded_img[p1+h:p1*2 +h,0:p2] =  f[h-1,0]
        padded_img[p1+h:p1*2 +h,p2+w:p2*2 +w] = f[h-1,w-1]
    for i in range(0,h):
        for j in range(0,w):
            result_img[i,j] = np.sum(padded_img[i:k1+i,j:j+k2] * W)
    return result_img

def gaussKernel(sig,m=None):#m=(3×σ)*2 + 1
    if m is None:
        m = round(3*sig)*2 + 1
    if m < round(3*sig)*2 + 1:
        raise("warning:m is too small")
    if sig <=0:
        sig = 0.3*((m-1)*0.5 - 1) + 0.8
    result = np.zeros((m,m),np.float32)
    for i in range(m):
        for j in range(m):
            d = np.sqrt((i-int(m/2))**2 + (j-int(m/2))**2)
            result[i,j] = np.exp(-d**2/(2*sig**2))/(2*np.pi*sig**2)
    sum_m = np.sum(result)
    result = result/sum_m
    return result




def cal_grad(img):
    sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobely = np.array([[-1,-2,1],[0,0,0],[1,2,1]])
    h,w = img.shape
    gradx = np.zeros_like(img)
    grady = np.zeros_like(img)
    grads = np.zeros_like(img)
    thetas = np.zeros_like(img)
    where_grad = np.zeros_like(img)
    p1,p2 = 1,1
    padded_img = np.zeros((h+2*p1,w+2*p2),dtype = img.dtype)
    padded_img[p1:h+p1,p2:w+p2] = img
    
    for i in range(1,h+1):
        for j in range(1,w+1):
            temp = padded_img[i-1:i+2,j-1:j+2]
            temp_gradx = np.sum(temp*sobelx)
            temp_grady = np.sum(temp*sobely)
            gradx[i-1,j-1] = temp_gradx
            grady[i-1,j-1] = temp_grady
            grads[i-1,j-1] = np.sqrt(temp_gradx**2 + temp_grady**2)#callculate the grads
            theta = np.arctan(temp_gradx/temp_grady)#calculate theta
            thetas[i-1,j-1] = theta
            if theta >0 and theta <= np.pi / 4 :
                where_grad[i-1,j-1] = 0
            elif theta > np.pi/4 and theta <= np.pi/2:
                where_grad[i-1,j-1] = 1
            elif theta >= -np.pi/2 and theta <= -np.pi/4:
                where_grad[i-1,j-1] = 2
            elif theta > -np.pi/4 and theta < 0:
                where_grad[i-1,j-1] = 3

    return grads,thetas,where_grad

def rgb1gray(f,method='NTSC'):
    rows,cols,c = f.shape
    dtype_of_f = f.dtype
    grey_img = np.zeros((rows,cols),dtype=dtype_of_f)
    if method == 'NTSC':
        grey_img = f[:,:,0]*0.1140 + f[:,:,1]*0.5870 + f[:,:,2] * 0.2989#BGR
    elif method == 'average':
        grey_img = (f[:,:,0] + f[:,:,1] + f[:,:,2])/3
    else:
        raise("you should write the correct method")
    return grey_img

def NMS(img,where_grad,thetas,grads,lowthres):
    h,w = img.shape
    highvalue = np.max(img)
    lowvalue = lowthres * highvalue
    result = np.zeros_like(img)
    for i in range(1,h-1):
        for j in range(1,w-1):
            N= img[i-1,j]
            S = img[i+1,j]
            W = img[i,j-1]
            E = img[i,j+1]
            NW = img[i-1,j-1]
            NE = img[i-1,j+1]
            SW = img[i+1,j-1]
            SE = img[i+1,j+1]
            case ,theta = where_grad[i,j] ,thetas[i,j]
            tantheta = np.tan(theta)
            if case == 0:
                gp1 , gp2 = (1-tantheta) *E + tantheta * NE ,(1-tantheta) *W + tantheta * SW
            elif case == 1:
                gp1 , gp2 = (1-tantheta) *N + tantheta * NE ,(1-tantheta) *S + tantheta * SW
            elif case == 2:
                gp1 , gp2 = (1-tantheta) *N + tantheta * NW ,(1-tantheta) *S + tantheta * SE
            elif case == 3:
                gp1 , gp2 = (1-tantheta) *W + tantheta * NW ,(1-tantheta) *E + tantheta * SE

            if grads[i,j] >=gp1 and grads[i,j] >=gp2:
                if grads[i,j] >= highvalue:
                    grads[i,j] = highvalue
                    result[i,j] = 255

                elif grads[i,j] < lowvalue:
                    grad[i,j] = 0
                else:
                    grads[i,j] = lowvalue
            else:
                grads[i,j] = 0
        
    for i in range(1,h-1):
        for j in range(1,w-1):
            if grads[i,j] == lowvalue:
                temps = grads[i-1:i+2,j-1:j+2]
                if np.any(temps==highvalue):
                    result[i,j] = 255

    
    return result

if __name__ == "__main__":
    print("start")
    img_path = "lena512color.tiff"
    img = cv2.imread(img_path)
    img = rgb1gray(img)
    plt.imshow(img,cmap='gray')
    plt.show()
    print("read succesfull")
    highvalue = np.max(img)
    lowthres = 0.05
    GW = gaussKernel(sig=1)
    flitered_img = twodConv(img,GW)
    grads,thetas,where_grad = cal_grad(flitered_img)
    results = NMS(flitered_img,where_grad,thetas,grads,lowthres)
    plt.imshow(results,cmap='gray')
    plt.show()







