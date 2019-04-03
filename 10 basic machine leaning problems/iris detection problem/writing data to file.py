#Converting linguistic data to numerical data for words in articles
#To get input and output matrix for ANN
import pandas as pd,numpy as np

m,n,classes,exInd = articleArr.shape[0],uniqueArt.shape[0],uniqueTag.shape[0],0
pd.DataFrame(uniqueArt).T.to_csv('articles.csv',index=False,header=None)
pd.DataFrame(uniqueTag).T.to_csv('tags.csv',index=False,header=None)
print('ok')
tempX = pd.DataFrame(0, index=np.arange(10000), columns=np.arange(n))
tempy = pd.DataFrame(0, index=np.arange(10000), columns=np.arange(classes))
while exInd < m:
    for word in uniqueArt:
        if word in articleArr[exInd]: tempX.ix[exInd][np.where(uniqueArt == word)[0][0]] = 1
    for tag in uniqueTag:
        if word in tagArr[exInd]: tempy[exInd][np.where(uniqueTag == word)[0][0]] = 1
    exInd += 1
    if exInd%10000 == 0:
        print(str(exInd/6733908),'%')
        tempX.to_csv('articles.csv',header=None, mode='a', index=False)
        tempy.to_csv('tags.csv',header=None, mode='a', index=False)
        if 6733908//exInd == 1:
            tempX = pd.DataFrame(0, index=np.arange(exInd,6733909), columns=np.arange(n))
            tempy = pd.DataFrame(0, index=np.arange(exInd,6733909), columns=np.arange(classes))
        else:
            tempX = pd.DataFrame(0, index=np.arange(exInd, exInd+10000), columns=np.arange(n))
            tempy = pd.DataFrame(0, index=np.arange(exInd, exInd+10000), columns=np.arange(classes))
    elif exInd == 6733908:
        tempX.to_csv('articles.csv',header=None, mode='a', index=False)
        tempy.to_csv('tags.csv',header=None, mode='a', index=False)
print('DONE!')
del temp
