#	#	#	#	#	#	#	Import libraries	#	#	#	#	#
import os
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np
from numpy import array
import glob
import pdb										#pdb.set_trace() to debug
import time
start_time = time.time();						#To measure the preformance of the code
#
#
#
#	#	#	#	#	#	#	Functions	#	#	#	#	#	#	#
#---Average color stats---------Finds the average non-white color with standard deviation in the letter
def avgColor(im1):
	whiteRange = 20;					#All colors components within 255 and 255-whiteRange, will be ignored
	nonWhiteRange = 255-whiteRange;
	im1_arr = np.asarray(im1);			#Converting im1 image information into a numpy read only array (saves memory)
	mask = ((im1_arr[:,:,0]<nonWhiteRange) & (im1_arr[:,:,1]<nonWhiteRange) & (im1_arr[:,:,2]<nonWhiteRange));	#This mask will output true for all pixels that satisfy the condition. So, if pix[100,70] satisfies the give condition, then mask[100,70]=True . If pix[20,10] does not satisfy the condition (and therefore, it is considered white), then mask[20,10]=False
	RGBdata = im1_arr[(mask)];			#Getting the color data of all considered pixels
	if len(RGBdata) != 0:				#Some pictures that are completely white will have len(RGBdata) = 0, so this is important
		RGBstd = np.std(RGBdata, axis=0);							#Finding the standard deviation of each color component	
		avgC = (np.sum(RGBdata, axis=0))/len(RGBdata);				#Finding the average color of each component
	else:
		avgC = [0,0,0]; RGBstd = [0,0,0];
	return [avgC]+[RGBstd]				#The format of the return is: np.array([[average[R,G,B]],[stdev[R,G,B]]])
#
#---Merge two images------------im1(word) will merge onto im2(background) with the offset position (a,b) (which is (0,0) by default)
def mergeImages(im1,im2,a=0,b=0):
	avgC,stdev = avgColor(im1);			#Average color and standard deviation of each color component
	whiteRange = 20;
	im1_arr = np.asarray(im1);			#Converting im1 image information into a numpy read only array (saves memory)
	maxC = [int((avgC[0]+stdev[0])*1.5+1),int((avgC[1]+stdev[1])*1.5+1),int((avgC[2]+stdev[2])*1.5+1)];	#Setting the boundary color component parameters (pixels' RGB beyond these will be tolerated/not copied/masked while pasting)
	if maxC[0]+maxC[1]+maxC[2] >745:	#If the allowed colors allows very white colors, then use the whiteRange instead
		maxC[0] = 255-whiteRange; maxC[1] = 255-whiteRange; maxC[2] = 255-whiteRange;
	mask = np.all(im1_arr<maxC, axis = 2);		#This mask will output true for all pixels' RGB data(axis=2) that satisfies the condition. So, if pix[100,70] satisfies the given condition, then mask[100,70]=True . If pix[20,10] does not satisfy the condition (and therefore, it is considered white), then mask[20,10]=False
	imMask = Image.fromarray(np.uint8(mask)*255);					#Converting the mask into an image mask of one channel
	a = int(a); b = int(b);				#In case they are float
	im2.paste(im1, (a,b), imMask);		#Applying the image mask while pasting the pixels will remove the background whiteness
	return None							#No need to return, since the image was passed through reference, and it has permanently deformed
#
#---Loads a configuration-------Outputs the list between the provided parameter and the next parameter of the config file. The parameter's name must exactly match the one in the config file
def readConfig(fileName, parameter,dataType):
	f = open(fileName, 'r');
	string = [];
	list = ["" for x in range(100000)];
	i = 0;
	while True:
		c=f.read(1);
		if c == '[':
			while True:
				c=f.read(1);
				if c == ']':
					break;
				else:
					string = string+[c];
		elif not c:								#If it is the end of file, then close
			list[0] = 0;
			f.close();
			list = filter(None, list);			#Remove empty elements
			return list
			#raise BREAK_ALL
		if "".join(string) == parameter:
			c=f.read(1);						#To ignore the first '/n'
			while True:
				c=f.read(1);
				if dataType == 'paths':
					if c != '\n' and (not c) == 0 and c != '[':
						list[i] = list[i] + c;
					elif c == '\n':
						i = i+1;
					elif (not c) or c == '[':
						f.close();
						list = filter(None, list);				#Remove empty elements
						return list
						#raise BREAK_ALL
				elif dataType == 'vector':
					if c != ',' and c != ' ' and c != '[' and c != '\n':
						list[i] = list[i] + c;
					elif c == ',':
						i = i+1;
					elif (not c) or c == '[' or c == '\n':
						f.close();
						list = filter(None, list);				#Remove empty elements
						list = map(float, list);				#Convert the strings to float values
						return list
						#raise BREAK_ALL
				elif dataType == 'var':
					if c != '\n' and c != ' ' and c != '[' and c != '\n':
						list[i] = list[i] + c;
					elif (not c) or c == '[' or c == '\n':
						f.close();
						list = filter(None, list);				#Remove empty elements
						list = map(float, list);
						return list
						#raise BREAK_ALL
		string = [];
	return list
#
#---Convert string to vector----Converts a string with numbers separated by commas, into a vector
def stringToVector(string):
	vector = [0]*len(string);
	n = 0;
	numString = [ord(c) for c in string]
	for i in range(0,len(numString),1):
		if numString[i] >= 48 and numString[i] <= 57:
			if numString[i-1]==45 and vector[n]==0:					#If numString[i] has a number, and it is the first number of the new vector component, and before it comes a negative sign, then the vector component is negative, else it is positive
				isPositive = -1;
			elif vector[n]==0:
				isPositive = 1;	
			vector[n] = vector[n]*10 + (numString[i]-48)*isPositive;
		elif numString[i] == 44:
			n=n+1;
	vector = vector[0:n+1];
	return vector
#
#---Locates a similar image-----Locates the rectangle bounds of a similar looking image inside a large image. The small image's background should be pure white for the program to tolerate it. And both images should be in RGB mode
def scanForImage(word, page_arr, searchBounds = (0,0,0,0), params = (20,2,10,3,500000), LR = 0):
	#word = the untouched (AKA original) word PIL image we are scanning for
	#page_arr = page PIL image converted into a numpy 2D array through the command: page_arr = np.asarray(pageIm, dtype = np.dtype('i2')); where pageIm is the original PIL loaded page image
	# searchBounds = (0,0,0,0);					#The seach bounds region (rectangle) to scan the page. searchBounds = (topleft corner y, topleft corner x, bottomright y, bottomright x). leave it to (0,0,0,0) to scan the entire page 
	#params = tuple containing the following parameters (in a respective order), each seperated by comma:
		# colorUncertainity = 20;					#Color of the word found in the page should be within +- colorUncertainity/2
		# unwhiteRadius = uw = 2;					#The number of pixels from the randomly selected posPixel, that shouldn't be white (ei: nothing in the square of length = 2*unwhiteRadius should be white. And posPixel will be in the center of this square)
		# numberOfIterations =  10;					#Max number of times we will test for consistency in the pixel layout
		# maxNumberOfNoMatches = 3;					#More number of misses than this will result in the tolerance of the word
		# maxInitialMatches = 500000;				#The maximum number of seach results after the first test. Anything larger will result in the repition of initial search with a different random pixel color
	#LR = 0;										#If LR = 0, then just carry the normal procedure. If LR = 1, then crop 1/3rd left portion of word image and search for it. If LR = 2, then crop 1/3rd right portion of word image and search for it.
	#Loading the parameters into variables and then assigning some other variables
	colorUncertainity = params[0]; unwhiteRadius = uw = params[1]; numberOfIterations =  params[2]; maxNumberOfNoMatches = params[3]; maxInitialMatches = params[4];
	unC = [colorUncertainity,colorUncertainity,colorUncertainity]; pageDim = page_arr.shape[0:2]; noMatchTimes = 0;
	#Converting the word into an appropriate format
	wordTrans = Image.new("RGB", word.size, (255,255,255));					#Blank white image of the word's dimensions
	mergeImages(word, wordTrans);											#Removes colors close to white
	wordTrans = wordTrans.crop(ImageOps.invert(wordTrans).getbbox());		#Remove white borders from the word
	wordTransSize = wordTrans.size;											#Noting the size of the original wordTrans in case we are dealing with a split word
	if LR == 1:
		wordTrans = wordTrans.crop((0,0,int(wordTransSize[0]/3),wordTransSize[1]));				#The left 1/3 portion of word image (cut as verticle stripe) 
	elif LR == 2:
		wordTrans = wordTrans.crop((int(wordTransSize[0]*2/3),0,wordTransSize[0],wordTransSize[1]));	#The right 1/3 portion of word image (cut as verticle stripe)
	wordTrans_arr = np.asarray(wordTrans, dtype = np.dtype('i2'));			#Converts the word into a numpy 2D array for fast matrix operations
	firstTime = True;
	maxIters = numberOfIterations;

	for i in range(100):
		while True:
		#Seaching for a suitable pixel color in word that isn't too closely located to another white pixel:
			pos = tuple((np.random.rand(2)*(np.array(wordTrans_arr.shape[0:2])-[2*uw,2*uw])+[uw,uw]).astype(int));			#Creating a random pixel location
			if np.all(np.all(wordTrans_arr[pos[0]-uw:pos[0]+uw+1, pos[1]-uw:pos[1]+uw+1] != [255,255,255], axis=2)):		#If that random pixel doesn't have whiteness in its surrounding (including itself)
				#A suitable random pixel has been found
				#print "Random pixel position in word=",pos,"with a color of:",wordTrans_arr[pos],"was selected";
				break;
			#Else, the random pixel does not meet the criteria of being far from a white pixel and a new pixel will be tested next loop
		
		if firstTime == True:
		#When a good pixel location in word has been found:
			#Calculate the seaching region bounds
			boundBoundaries = bx = list(pos) + list(pageDim - array(wordTrans_arr.shape[0:2]) + array(pos));	#Setting the scan rectangle boundaries
			if searchBounds[0] >= bx[0]:
				bx[0] = searchBounds[0];
			if searchBounds[1] >= bx[1]:
				bx[1] = searchBounds[1];
			if searchBounds[2] <= bx[2] and searchBounds[2] > bx[0]:
				bx[2] = searchBounds[2];
			if searchBounds[3] <= bx[3] and searchBounds[3] > bx[1]:
				bx[3] = searchBounds[3];
			#Subtract the color of the random pixel from all pixels of page_arr (within the search bounds and with the corrct offset) and then compare which pixels have a matching (+ uncertainity) color
			a = abs(page_arr[bx[0]:bx[2], bx[1]: bx[3]] - wordTrans_arr[pos]);
			b = np.all(a < unC, axis=2);
			k = np.where(b == True);
			k = (k[0] + bx[0], k[1] + bx[1]);
			matches = zip(*k);
			print len(matches),"initial matches to the word were found";
			if len(matches) == 0:
				#If no matches were found, then repeat the search for this word
				print "No matches were found, searching for the current word again";
				firstTime = True;
				continue;
			del a, b;
			iniPos = pos;							#Initial position of random pixel in the word
			if len(matches) < maxInitialMatches:	#If the number of matches isn't too large, then continue with the script otherwise choose another pixel color and hope for better luck this time
				firstTime = False;
		else:
			#If it isn't he first time, then:
			transition = array(pos) - array(iniPos);
			matches = [matches[z] for z in np.where(np.all(array([abs(page_arr[tuple(x)] - wordTrans_arr[pos]) for x in matches+transition]) < unC, axis=1) == True)[0]];
#			print len(matches),"matches to the word were found";
			if len(matches) == 0:
				#If no matches were found, then repeat the search for this word
				print "No matches were found, searching for the current word again";
				firstTime = True;
				numberOfIterations = i + maxIters;
				noMatchTimes = noMatchTimes + 1;
				if noMatchTimes > maxNumberOfNoMatches:
					#Too many no matches have occured, it's time to ignore the word
					del word; del wordTrans; del wordTrans_arr;
					return 0;
				continue;			
			if i >= numberOfIterations or len(matches) == 1:
				#A good number of tests were carried out or only one match remains. it's time to quit
				break;

	#Eleminating all matching pixels positions occuring in clusters except one is left for each cluster
	clusterList = [];
	isNotInCluster = True;							#It is set to true initially for the first match pixel position to be added to the clusterList
	for i in matches:
		for currentCluster in clusterList:
			isNotInCluster = isNotInCluster and not(i[0] > currentCluster[0] and i[0] < currentCluster[0]+wordTrans_arr.shape[0] and i[1] > currentCluster[1] and i[1] < currentCluster[1]+wordTrans_arr.shape[1]);	#If the current word pixel position is not within the word boundaries of another previous cluster, then isNotInCluster = True , else isNotInCluster = False
		if isNotInCluster == True:					#If the current word match pixel pos is not within any of the currently existing clusters, then make a new cluster with the given pixel position's word's top left boundary
			#A new cluster of matching word was found at i. otherwise, the current match is within the bounds of a pre-existing cluster
			clusterList = clusterList + [tuple(i - array(iniPos))];		#tuple(i - array(iniPos)) is the top left position of the word's boundary
	matches = clusterList;							#actual word matches = the list of clusters
	if len(matches) > 0:
		i = matches[0];
		a = i[0]; b = i[1]; c = a + wordTrans_arr.shape[0]; d = b + wordTrans_arr.shape[1];	#(a,b) = top left corner and (c,d) = bottom right corner of the matching word's boundaries
		#Now we will superimpose the word over the mushaf page (at the location we found) to insure that the word actually belongs there, and it's not just some dark region chosen by mistake
		#At first, we must convert all pure white pixels in the word image array into black ones, so that subtraction during superimposing will ignore the white pixels (which are black now)
		mask = np.any(wordTrans_arr != [255,255,255], axis = 2);
		#wordTrans_arr[~mask] = [0,0,0]; #converting all white pixels into black
		#avgDiff = np.mean(np.subtract((page_arr[a:c,b:d])[mask],wordTrans_arr[mask]), axis = 2);
		avgDiff = np.mean(abs((page_arr[a:c,b:d])[mask]-wordTrans_arr[mask]), axis = 0);
#INSTEAD OF SUBTRACTING page_arr from wordTrans_arr, just sum up each, and subtract their sum. and finally divide the difference by the size of ones in mask. THIS WILL GET YOU THE MEAN, NOT THE MEDIAN, WHICH WE ARE CURRENTLY USING
		print "avgDiff is", avgDiff;
		if np.all(avgDiff < 25, axis = 0):
#			print "Reduced matches to the word to",len(matches),"by eliminating matches occuring in clusters";
			del word; del wordTrans; del wordTrans_arr;
			if LR == 1:
				d = b + wordTransSize[0];
			elif LR == 2:
				b = d - wordTransSize[0];
			return (a,b,c,d);						#The returned tuple contains the top left corner boundary (a,b) and the bottom right boundary (c,d) for each word match found 
		else:
			return 0;								#The location we found probably doesn't contain the word!
#
#---Draws words' outlines-------Draws colored rectangles around the given word positions (which is a tuple that contains the topleft corner yx and bottomright corner yx positions)
def drawWordOutline(wordPos, page_arr, initialWordNumber ,width = 5):
	#page_arr = page PIL image converted into a numpy 2D array through the command: page_arr = np.asarray(pageIm, dtype = np.dtype('i2')); where pageIm is the original PIL loaded page image
	page = Image.fromarray(page_arr.astype('uint8'), 'RGB');
	draw = ImageDraw.Draw(page);
	fnt = ImageFont.truetype("verdana", 70);
	borderColor = (200,0,0);
	w = width = 5;										#Width of outline
	if isinstance(initialWordNumber, tuple) == True:
		k = 0;											#If the initialWordNumber is a tuple, then only label the words as the numbers given in the tuple	
		j = initialWordNumber[k];
	else:
		j = initialWordNumber;							#Word number increases after each iteration and gets printed to the output picture
	for i in wordPos:
		if i != 0:										#If the word isn't missing
			a = i[0]; b = i[1]; c = i[2]; d = i[3];		#(a,b) = top left corner and (c,d) = bottom right corner of the matching word's boundaries 
			page_word_crop = page.crop((b,a,d,c));
			draw.rectangle([(b-w,a-w),(d+w,c+w)], fill=borderColor);
			page.paste(page_word_crop, (b,a));
			m = int((b+d)/2); n = int((a+c)/2 - 45);
			draw.text((m,n), str(j), font=fnt, fill=borderColor);
			#Iterating border color
			if borderColor == (200,0,0):
				borderColor = (0,200,0);
			elif borderColor == (0,200,0):
				borderColor = (0,0,200);
			elif borderColor == (0,0,200):
				borderColor = (200,0,0);
		if isinstance(initialWordNumber, tuple) == True:
			k = k + 1;
			try: j = initialWordNumber[k];
			except: pass;
		else:
			j = j + 1;									#Iterate to the next number
	return page;
#
#---Saves the word positions----This function save the word positions to the txtFile preformatted text file, in the following format: "[Surah number(number only)]\n(word1)page_number, x_pos, y_pos, width, height\n(word2)page_number, x_pos, y_pos, width, height"
def saveWordPositions(wordMatches, wordNumbers, pageNumber = 1, surahNumber = 1 , mode = "OverwriteNew",txtFile = "C3 word positions.txt"):
#IF THERE'S A gap between the initialWordNumber and the last pre-saved word position, the fill the gap with enough zeros (ie: "0\n") that will equal the number of words in between the gap
#Also add a separate single word position save mode (or make a new function with this functionality), that will save the word positions of the given wordMatches and their respective wordNumbers
	#mode controls the mode of overwriting on existing data:
	#	"Overwrite0" only overwrites new data on top of "0" (missing position). No old data will be overwritten
	#	"OverwriteNew" will overwrite New position data over the old data, as long as the new data isn't a missing position ("0")
	#	"OverwriteAll" will overwrite all new data over all data, including any new missing position ("0")
	
	#preparing the list that stores word positions
	wordNumberMax = np.max(wordNumbers); wordNumberMin = np.min(wordNumbers);	
	#Opeining the txtFile
	f = open(txtFile, 'r');						#Opening the file
	fLines = (f.read()).split("\n");			#Each element of fLines ('File Lines') contains the data of each line. The split("\n") does the splitting job
	f.close();									#There's no point in keeping the file open (in the RAM) anymore, since we have the contents loaded into the RAM
	initialL = fLines.index('['+str(surahNumber)+']');	#Searching for "[surahNumber]" index in the list fLines. first word position will be in fLines[initialL + 1]
	finalL = fLines.index('['+str(surahNumber+1)+']');	#Searching for the line containing the next surah word positions	
	a = fLines[:initialL+1];					#From begining of txtFile to "[surahNumber]" (included)
	b = fLines[finalL-1:];						#From "\n" (included) prior to "[surahNumber+1]" to the end of file
	wordPos = fLines[initialL+1:finalL-1] + [str(pageNumber)+",0" for x in range(wordNumberMax - (finalL - initialL - 2))];	
	if mode == "OverwriteNew":
		for j in range(len(wordNumbers)):
			if wordMatches[j] != 0:
				#If the word position is not missing
				x = (wordMatches[j][1], wordMatches[j][0], wordMatches[j][3] - wordMatches[j][1], wordMatches[j][2] - wordMatches[j][0]);	#Converting the (y_pos, x_pos, y_pos+h, x_pos+w) format of wordMatch[j] into (x_pos, y_pos, w, h) format
				wordPos[wordNumbers[j]-1] = str(pageNumber) + "," + (str(x)[1:-1]).replace(' ', '');		#Converting the wordMatch data into the appropriate format. The [1:-1] erases the "(...)" brackets, and the replace function deletes whitespaces
	fLines = a + wordPos + b;
	open(txtFile, 'w').close();					#Clears the file from any contents
	f = open(txtFile, 'w');						#Reopening the file but to write in it this time
	for j in range(len(fLines)):
		f.write(fLines[j]+'\n');
	f.close();
	return 0;									#There's nothing to return
#
#	#	#	#	#	#	#	Block of code	#	#	#	#	#	#
print "Hello world";


#	+	+	+	+	+	+	Loading config files	+	+	+	+
print ("Loading configuration files");
#Loading variables
initalWord_finalWord = [int(i) for i in readConfig("config.ini", "Range of words to scan", "vector")];	#Inital word number and final word number
pageImNum = int(readConfig("config.ini", "Mushaf initial page number", "var")[0]);	#Page number of the first page (this page is supposed to contain the first word)
ext = readConfig("config.ini", "Page picture extention", "paths")[0];	#Image extention of the page, for example: '.jpg'
surahNumber = 20;														#Needed in order to know where to save the word position data

#Building paths
pageDir = readConfig("config.ini", "Page directory", "paths")[0];		#Path to the directory of the page images (which should contain all pages of the particular mushaf. For instance, it should contain: 1.jpg, 2.jpg ,..., 378.jpg , where all images corespond to the respective images of the pages)
wordDir = readConfig("config.ini", "Word directory", "paths")[0];		#Path to the directory of the word images (which should contain all words of the particular mushaf and surah. For instance, it should contain: 1.jpg, 2.jpg ,..., 1365.jpg , where all images corespond to the respective images of the words)
pageIm = Image.open(pageDir + "\\"+ str(pageImNum) + ext);				#Loading the first page image
wordImPaths = [];
for i in range(initalWord_finalWord[0], initalWord_finalWord[1]+1):
	#Building the paths of all word images
	if os.path.isfile(wordDir + "\\"+ str(i) +".jpg") == True:			#If the word image exists, then save the path
		wordImPaths = wordImPaths + [wordDir + "\\"+ str(i) +".jpg"];
	else:																#Otherwise, if the word image does not exists, save its path as zero (which will make it easy to identify missing words later on)
		wordImPaths = wordImPaths + [0];
		print "File at "+ wordDir + "\\"+ str(i) +".jpg" + " was not found. Ignoring that file";

#Break the program:
#helloWorldCrasher = Image.open("C:\\ThisAintADirectory.jpg");

#Assigning variables
page_arr = np.asarray(pageIm, dtype = np.dtype('i2'));					#Converting the page into a numpy 2D array
pageDim = page_arr.shape[0:2];
wordMatches = [];

#Setting the parameters of the search
colorUncertainity = 20;					#Color of the word found in the page should be within +- colorUncertainity
unwhiteRadius = uw = 2;					#The number of pixels from the randomly selected posPixel, that shouldn't be white (ei: nothing in the square of length = 2*unwhiteRadius should be white. And posPixel will be in the center of this square)
numberOfIterations =  10;				#Number of times we will test for consistency in the pixel layout
maxNumberOfNoMatches = 0;				#More number of misses than this will result in the tolerance of the word
maxInitialMatches = 500000;				#The maximum number of seach results after the first test. Anything larger will result in the repition of initial search with a different random pixel color
previousWord_TopOffset = 200;			#Each current word will be seached in the region based on the prevoius word's location; previousWord_TopOffset defines the number of y-pixels above the previous word's top in which the current word will be searched in
previousWord_BottomOffset = 500;		#previousWord_BottomOffset defines the number of y-pixels below the previous word's bottom in which the current word will be searched in
maxMissingWordsToNextPage = 4;			#Number of missing words to declare to move on to next page
missingWordCounter = 0;					#Counts number of consecutive missing words
startingWord = 0;						#Starting word number in the current page
parameters = (colorUncertainity, uw, numberOfIterations, maxNumberOfNoMatches, maxInitialMatches);

scanning = True;
while scanning:
	firstTime = True;
	for j in range(startingWord,len(wordImPaths)):
		if wordImPaths[j] != 0:							#If the word isn't missing:
			word = Image.open(wordImPaths[j]);			#Loading the word image
			#Random sampling
			print "Now searching for word number", j;
			if not firstTime:
				try:
					#Since so many errors often occur in this step due to multiple causes, it is better to use try than specify an if condition
					a = wordMatches[-1][0] - previousWord_TopOffset;
					b = wordMatches[-1][2] + previousWord_BottomOffset;
					searchRegion = (a,0,b,0);
				except:
					#If it's the first word or if the previous word was not found, then set the search region to the entire page (by setting it to (0,0,0,0), the scanForImage will scan the entire page)
					searchRegion = (0,0,0,0);
			else:
				#If it's the first word of the new page, set the searchRegion to (0,0,0,0) for full page scan			
				searchRegion = (0,0,0,0);
				firstTime = False;
			wordMatches += [scanForImage(word, page_arr, searchBounds = searchRegion, params = parameters, LR = 0)];
			if wordMatches[-1] != 0:
				missingWordCounter = 0;
			else:
				#The word was not found. Possible reasons are: the word was originally split, the word exists on the next page, or the program was just plain out of luck for choosing bad random pixels
				#We should now test for split word (this will also serve as a rescan for an out of luck bad scan). This will be done by splitting the word image into 3 vertical stripes (of equal width) and rescanning for the left and right stripes (middle will be ignored as it will probably contain both portions of the split word, making the scan unsuccessful)
				#Either the left or the right stripe will return a successful scan if the word is truely split
				print "word", j,"could not be found in the page. Rescanning with possiblity of split word";
				#Scanning for the left 1/3rd portion of word image
				x = scanForImage(word, page_arr, searchBounds = searchRegion, params = parameters, LR = 1);
				if x == 0:
					#If left portion is not found, then scan for the 1/3rd right portion
					x = scanForImage(word, page_arr, searchBounds = searchRegion, params = parameters, LR = 2);
				if x != 0:
					wordMatches[-1] = x;
					missingWordCounter = 0;		#The word was successfuly found
				elif x == 0:
					#The word is still not found, either it's not in the page, or we're just unlucky
					missingWordCounter = missingWordCounter + 1;
					print "word", j,"could not be found in the page. It will be ignored";					
	#ADD: ONLY TURN TO NEXT PAGE IF THE TOP SCAN REGION IN searchRegion IS BELOW 75% (OR SOMETHING ELSE) OF THE HEIGHT OF MUSHAF PAGE
					if missingWordCounter >= maxMissingWordsToNextPage:
						#Turn to next page and go back maxMissingWordsToNextPage number of words
						print "Done searching for words of page", pageImNum;
						print("Searching time: %s seconds" % (time.time() - start_time));
						#Drawing colored bordered rectangle around the matching words
						print "now saving the highlighted page"
						page = drawWordOutline(wordMatches[startingWord:], page_arr, initalWord_finalWord[0] + startingWord,width = 5);
						#Saving the image with a border around matching words
						page.save(str(pageImNum)+' index.png');
						#Saving the word positions
						wordNumbers = range(startingWord+initalWord_finalWord[0],initalWord_finalWord[0]+j+1);
						saveWordPositions(wordMatches[startingWord:], wordNumbers, pageNumber = pageImNum, surahNumber = surahNumber, mode = "OverwriteNew",txtFile = "C3 word positions.txt");
						#Moving onto next page
						print "Moving onto next page";
						missingWordCounter;
						del wordMatches[-maxMissingWordsToNextPage:]
						startingWord = j + 1 - maxMissingWordsToNextPage;
						pageImNum = pageImNum + 1;
						pageIm = Image.open(pageDir + "\\"+ str(pageImNum) + ext);
						page_arr = np.asarray(pageIm, dtype = np.dtype('i2'));					#Converting the page into a numpy 2D array
						break;					#Break out of this word searching loop in order to move back maxMissingWordsToNextPage number of words and repeat the search in the new page
		else:
			#If the word image file is missing, then just assign a zero to it
			wordMatches = wordMatches + [0];
		if j == len(wordImPaths) - 1:
			#If the last word has been reached and searched for, then break out of this word searching infinte loop
			scanning = False;
			break;

print "Done searching for similar word(s)";
print("Searching time: %s seconds" % (time.time() - start_time));

#Drawing colored bordered rectangle around the matching words
print "now saving the last highlighted page"
page = drawWordOutline(wordMatches[startingWord:], page_arr, initalWord_finalWord[0] + startingWord,width = 5);
#Saving the image with a border around matching words
page.save(str(pageImNum)+' index.png');
#Saving the word positions
wordNumbers = range(startingWord+initalWord_finalWord[0],initalWord_finalWord[0]+j+1);
saveWordPositions(wordMatches[startingWord:], wordNumbers, pageNumber = pageImNum, surahNumber = surahNumber, mode = "OverwriteNew",txtFile = "C3 word positions.txt");



#Suggestions:
#	Try Using b = np.all(np.sum(a, axis=2) < SOMENUMBER, axis=2); instead of b = np.all(a < d, axis=2);
#	Turn the "scan for word" into a function with several accessible parameters (Such as maxNumberOfNoMatches, previousWord_TopOffset and etc...)

#To do:
#	Add saving option to save the word locations into a txt file
#	Add command only option, where the program will start with command enabled rather than scanning for Images
#	Add a command to draw word location boxes by reading the word location file
#	Implement multipage functionality in the command section (mainly to operate repair word commamd with multipages, and loaded location boxes, and draw location boxes over multipage mushafs)



print("Total execution time: %s seconds" % (time.time() - start_time));
os.system("pause");


#	+	+	+	+	+	+	Console commands	+	+	+	+	+
while 1:
	command = raw_input("Would you like to execute a command? Possible commands are:\n"
						"Repair word xxx,yyy,zzz\n"
						"\twhere xxx or yyy or zzz = the list of number that need to get rescanned, each number must be seperated by a comma\n"
						"Save positions xx,yy\n"
						"\twhere xx represents the Mushaf number and yy represents the surah number\n"
						"\tThis command saves the position index of the words to the text file \"Cxx word positions\" under the the parameter [yy]\n"
						"Break program\n"
						"\tbreaks out of waiting for a command\n"
						"Enter your command:\n");
	if "Repair word" in command:
#	+	+	+	+	+	+	Rescan for words	+	+	+	+	+
		start_time = time.time();
		searchWordNumbers = [int(i) - initalWord_finalWord[0] for i in stringToVector(command[12:])];
		#Setting the parameters of the search
		maxNumberOfNoMatches = 1;
		parameters = (colorUncertainity, uw, numberOfIterations, maxNumberOfNoMatches, 1000000); #Same as previous, except having the maxInitialMatches being set to 1000000, since we are doing full page scans for each word
		page_arr = np.asarray(pageIm, dtype = np.dtype('i2'));
		wordSearchMatches = [];
		
		for j in searchWordNumbers:
			#Scanning for word
			word = Image.open(wordImPaths[j]);			#Loading the word image
			print "Now searching for word number", j;
			searchRegion = (0,0,0,0);
			wordMatches[j] = scanForImage(word, page_arr, searchBounds = searchRegion, params = parameters, LR = 0);
			wordSearchMatches = wordSearchMatches + [wordMatches[j]];
			if wordMatches[j] == 0:
				#The word was not found. Possible reasons are: the word was originally split, the word exists on the next page, or the program was just plain out of luck for choosing bad random pixels
				#We should now test for split word (this will also serve as a rescan for an out of luck bad scan). This will be done by splitting the word image into 3 vertical stripes (of equal width) and rescanning for the left and right stripes (middle will be ignored as it will probably contain both portions of the split word, making the scan unsuccessful)
				#Either the left or the right stripe will return a successful scan if the word is truely split
				print "word", j,"could not be found in the page. Rescanning with possiblity of split word";
				#Scanning for the left 1/3rd portion of word image
				x = scanForImage(word, page_arr, searchBounds = searchRegion, params = parameters, LR = 1);
				if x == 0:
					#If left portion is not found, then scan for the 1/3rd right portion
					x = scanForImage(word, page_arr, searchBounds = searchRegion, params = parameters, LR = 2);
				if x != 0:
					wordMatches[j] = x;
				else:
					print "word", j,"could not be found in the page. It will be ignored";
			word.show(title = str(j + initalWord_finalWord[0]));
		print "Done scanning for the word";
		print("Searching time: %s seconds" % (time.time() - start_time));
		
		#Drawing colored bordered rectangle around the matching words
		print "now drawing the highlighted image"
		page = drawWordOutline(wordSearchMatches, page_arr, tuple(array(searchWordNumbers) + initalWord_finalWord[0]) ,width = 5);
		page.show(title = "Highlighted page");
		del word, page;


	if command == "Save positions" in command:
		start_time = time.time();
		mushafNumber = int(stringToVector(command[15:])[0]);
		surahNumber = int(stringToVector(command[15:])[1]);



	elif command == "Break program":
		#Breaks the program out of the infinite loop waiting for a command
		print "No more commands will be taken";
		break;


	else:
#	+	+	+	+	+	+	Excecute some other command	+	+	+
		try:
			exec(command);
		except:
			print "Error!";







