import cv2
import numpy as np

n = input("Enter the number of images ")
n = int(n)
imgs = []
print("\nEnter images where the images to the right come first.\n")

for i in range(n):
    name = input("Enter file name ")
    imgs.append(cv2.imread(name))

features = []

class CombineImages():
    def __init__(self,img1, img2,method = 'sift',matching = 'flann'):
        self.imgs = [None,None]
        self.method = method
        self.matching = matching
        self.img1=img1
        self.img2=img2

    def getKpAndDescriptors(self,image):
	    if self.method == 'sift':
		    descriptor = cv2.xfeatures2d.SIFT_create()
	    elif self.method == 'surf':
		    descriptor = cv2.xfeatures2d.SURF_create()

	    kps,fts = descriptor.detectAndCompute(image,None)
	    kps = np.float32([kp.pt for kp in kps])
	    return {"keypoints":kps,"descriptors":fts}

    def compute(self,features,ratio_threshold = 0.7):
	    if self.matching == "flann":
		    FLANN_INDEX_KDTREE = 0
		    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		    search_params = dict(checks=50)   # or pass empty dictionary
		    matcher = cv2.FlannBasedMatcher(index_params,search_params)
		    #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
	    else:
		    matcher = cv2.BFMatcher(cv2.NORM_L2)

	    knn_matches = matcher.knnMatch(features[0]["descriptors"], features[1]["descriptors"], 2)
	    matches = []
	    for m in knn_matches:
		    if m[0].distance< ratio_threshold* m[1].distance and len(m)==2:
		        matches.append((m[0].trainIdx,m[0].queryIdx))

	    ptsA = np.float32([features[0]["keypoints"][i] for (_, i) in matches])
	    ptsB = np.float32([features[1]["keypoints"][i] for (i, _) in matches])
	    matrix, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,5.0)
	    return matches,matrix,status

    def drawMatches(self,imgs,features,matches,status):
	    (ha,wa) = imgs[0].shape[:2]
	    (hb,wb) = imgs[1].shape[:2]

	    vis = np.zeros((max(ha,hb),wa+wb,3),dtype = "uint8")
	    vis[0:ha,0:wa]=imgs[1]
	    vis[0:hb,wa:]=imgs[0]

	    for ((trainIdx,queryIdx),s) in zip(matches,status):
		    if s==1:
		        pta = (int(features[0]["keypoints"][queryIdx][0]),int(features[0]["keypoints"][queryIdx][1]))
		        ptb = (int(wa+features[1]["keypoints"][trainIdx][0]),int(features[1]["keypoints"][trainIdx][1]))
		        cv2.line(vis,pta,ptb,(0,255,0),1)
	    cv2.imshow("merge",vis)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()
	    return vis

    def combine(self):
        features = []
        print("getting descriptors")
        features.append(self.getKpAndDescriptors(self.img1))
        features.append(self.getKpAndDescriptors(self.img2))
        print("got descriptors")

        matches,matrix,status = self.compute(features)
        print("got matrix")
        #drawMatches
        width = self.img1.shape[1] + self.img2.shape[1]
        height = self.img1.shape[0]#,self.img2.shape[0])
        print("computing combined Image")
        image = cv2.warpPerspective(self.img1,matrix,(width,height))
        image[0:self.img2.shape[0],0:self.img2.shape[1]] = self.img2
        return image


#for img in imgs:
    #features.append(getKpAndDescriptors(img,'sift'))

def showImg(img):
    cv2.imshow("temp",img)          
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

i =0 
for i in range(len(imgs)-1):
    result = CombineImages(imgs[i],imgs[i+1]).combine()
    #cv2.imshow("temp",result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    print("press any key to go to next step")
    showImg(result)
    cv2.imwrite("result"+str(i)+".jpg",result)
    imgs[i+1]=result
    
cv2.imwrite("FINAL_IMAGE.jpg",imgs[len(imgs)-1])

#matches,matrix,status = compute(features,"flann")
#drawMatches(imgs,features,matches,status)
#print("creating the image now...")
#width = imgs[0].shape[1] + imgs[1].shape[1]
#height = max(imgs[1].shape[0],imgs[0].shape[0])
#ima#ge = cv2.warpPerspective(imgs[0],matrix,(width,height))
#image[0:imgs[0].shape[0],0:imgs[0].shape[1]] = imgs[0]
#image[0:imgs[1].shape[0],0:imgs[1].shape[1]] = imgs[1]
#print("done")
#cv2.imshow("result",image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
