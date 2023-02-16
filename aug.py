
import cv2
import numpy as np


# MIN_MATCHES = 20
# detector = cv2.ORB_create(nfeatures=5000)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=100)
# flann = cv2.FlannBasedMatcher(index_params,search_params)


# def load_input():
	





# def drowsyness_detector(frame, augment_list):

	

# 	cap = cv2.VideoCapture(0)
# 	ret, frame = cap.read()

# 	while(ret):
# 		ret, frame = cap.read()
		
		






def augment_detector(frame, augment_list,augment_list2):
    input_image=augment_list[0]
    aug_image=augment_list[1]
    input_keypoints=augment_list[2]
    input_descriptors =augment_list[3]

    MIN_MATCHES = augment_list2[0]
    detector = augment_list2[1]
    FLANN_INDEX_KDTREE = augment_list2[2]
    index_params = augment_list2[3]
    search_params = augment_list2[4]
    flann = augment_list2[5]

    # frame = cv2.flip(frame,1)
    if(len(input_keypoints)<20):
        return frame
    frame = cv2.resize(frame, (600,450))
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
    matches = compute_matches(input_descriptors, output_descriptors,flann)
    augment_list[0]=input_image
    augment_list[1]=aug_image
    augment_list[2]=input_keypoints
    augment_list[3]=input_descriptors


    augment_list2[0]=MIN_MATCHES
    augment_list2[1]=detector
    augment_list2[2]=FLANN_INDEX_KDTREE
    augment_list2[3]=index_params
    augment_list2[4]=search_params
    augment_list2[5]=flann


    if(matches!=None):
        if(len(matches)>10):
            src_pts = np.float32([ input_keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ output_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

			#Finally find the homography matrix
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			#matchesMask = mask.ravel().tolist()
            pts = np.float32([ [0,0],[0,399],[299,399],[299,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            M_aug = cv2.warpPerspective(aug_image, M, (600,450))

			#getting the frame ready for addition operation with Mask Image
            frameb = cv2.fillConvexPoly(frame,dst.astype(int),0)
            Final = frameb+M_aug
            augment_list[0]=input_image
            augment_list[1]=aug_image
            augment_list[2]=input_keypoints
            augment_list[3]=input_descriptors	


            augment_list2[0]=MIN_MATCHES
            augment_list2[1]=detector
            augment_list2[2]=FLANN_INDEX_KDTREE
            augment_list2[3]=index_params
            augment_list2[4]=search_params
            augment_list2[5]=flann
				#output_final = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            return Final
				#cv2.imshow('Finallli', Final)
        else:
            return frame
    else:
        return frame


def compute_matches(descriptors_input, descriptors_output,flann):
	# Match descriptors
	if(len(descriptors_output)!=0 and len(descriptors_input)!=0):
		matches = flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output,np.float32),k=2)
		good = []
		for m,n in matches:
			if m.distance < 0.69*n.distance:
				good.append(m)
		return good
	else:
		return None