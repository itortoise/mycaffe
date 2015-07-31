import os
import scipy

def makeImageSample(imagepath):
    pathDir = os.listdir(imagepath)
    with open('train.txt','w') as f:
        imageList = [file for file in pathDir if file.endswith('.jpg')]
        for name in imageList:
            label = int(name[2:5])
            label = str(label)
            f.write(imagepath+'/'+name+' '+label+'\n')
            
            
if __name__ == '__main__'  :
    imagePath = '/home/math/irisDatabase/Lamp'
    makeImageSample(imagePath)
    
