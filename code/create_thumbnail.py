#Imports
import cv2
import os
import math
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import argparse
from os.path import isfile, isdir
import shutil
import imquality.brisque as brisque
import dlib
from mtcnn.mtcnn import MTCNN
import time
import platform
import csv
from datetime import datetime
import psutil
import tensorflow as tf
import sys
from brisque import BRISQUE
import json
from collections import ChainMap
#Paths
model_folder = "../models/"
frames_folder_outer = "../results/temp"
thumbnail_output = "../results/"
#Models
haarXml = model_folder + "haarcascade_frontalface_default.xml"
modelFile =  model_folder + "res10_300x300_ssd_iter_140000.caffemodel"
configFile = model_folder + "deploy.prototxt.txt"
eliteserien_logo_model = model_folder + "logo_eliteserien.h5"
soccernet_logo_model = model_folder + "logo_soccernet.h5"
surma_closeup_model = model_folder + "close_up_model.h5"

haarStr = "haar"
dlibStr = "dlib"
mtcnnStr = "mtcnn"
dnnStr = "dnn"
surmaStr = "surma"
svdStr = "svd"
laplacianStr = "laplacian"
ocampoStr = "ocampo"
eliteserienStr = "eliteserien"
soccernetStr = "soccernet"
filename_additional = "thumbnail"

#The probability score the image classifying model gives, is depending on which class it is basing the score on.
#It could be switched
close_up_model_inverted = False
#Execution time
frame_extraction=0
models_loading=0
logo_detection=0
closeup_detection=0
face_detection=0
blur_detection=0
iq_predicition=0
total=0
#frames extracted
numFramesExtracted=0
def main():
    parser = argparse.ArgumentParser(description="Thumbnail generator")
    parser.add_argument('-conf','--configuration', type=str,default="../config.json", required=False ,help="Full path to a configuration file formatted as JSON.")
    parser.add_argument('-iter','--iteration', type=int, required=False,help='Number of executions per configuration')
    parser.add_argument('-pa','--performanceAnalysis', type=bool,default=False,  required=False ,help="run performance analyis.")
    parser.add_argument("-tc", "--thumbnailCount", type=above_zero_int,  nargs=1, help="Number of desired output thumbnails " )
    #Logo detection models
    logoGroup = parser.add_mutually_exclusive_group(required=False)
    logoGroup.add_argument("-LEliteserien2019", action='store_true', help="Surma model used for logo detection, trained on Eliteserien 2019.")
    logoGroup.add_argument("-LSoccernet", action='store_true', help="Surma model used for logo detection, trained on Soccernet.")
    logoGroup.add_argument("-xl", "--xLogoDetection", default=True, action="store_false", help="Don't run logo detection.")

    #Close-up detection models
    closeupGroup = parser.add_mutually_exclusive_group(required=False)
    closeupGroup.add_argument("-CSurma", action='store_true', help="Surma model used for close-up detection.")
    closeupGroup.add_argument("-xc", "--xCloseupDetection", default=True, action="store_false", help="Don't run close-up detection.")

    #IQA models
    iqaGroup = parser.add_mutually_exclusive_group(required=False)
    iqaGroup.add_argument("-IQAOcampo", action='store_true', help="Ocampo model used for image quality assessment.")
    iqaGroup.add_argument("-xi", "--xIQA", default=True, action="store_false", help="Don't run image quality prediction.")

    #Blur detection models
    blurGroup = parser.add_mutually_exclusive_group(required=False)
    blurGroup.add_argument("-BSVD", action='store_true', help="SVD method used for blur detection.")
    blurGroup.add_argument("-BLaplacian", action='store_true', help="Laplacian method used for blur detection.")
    blurGroup.add_argument("-xb", "--xBlurDetection", default=True, action="store_false", help="Don't run blur detection.")


    #Face models
    faceGroup = parser.add_mutually_exclusive_group(required = False)
    faceGroup.add_argument("-dlib", action='store_true', help="Dlib detection model is slow, but presice.")
    faceGroup.add_argument("-haar", action='store_true', help="Haar detection model is fast, but unprecise.")
    faceGroup.add_argument("-mtcnn", action='store_true', help="MTCNN detection model is slow, but precise.")
    faceGroup.add_argument("-dnn", action='store_true', help="DNN detection model is fast and precise.")
    faceGroup.add_argument("-xf", "--xFaceDetection", default=True, action="store_false", help="Don't run the face detection.")

    #Flags fixing default values
    parser.add_argument("-cuthr", "--closeUpThreshold", type=restricted_float, nargs=1, help="The threshold value for the close-up detection model. The value must be between 0 and 1. The default is: " )
    parser.add_argument("-brthr", "--brisqueThreshold", type=float, nargs=1, help="The threshold value for the image quality predictor model. The default is: " )
    parser.add_argument("-logothr", "--logoThreshold", type=restricted_float,  nargs=1, help="The threshold value for the logo detection model. The value must be between 0 and 1. The default value is: ")
    parser.add_argument("-svdthr", "--svdThreshold", type=restricted_float,  nargs=1, help="The threshold value for the SVD blur detection. The default value is: " )
    parser.add_argument("-lapthr", "--laplacianThreshold", type=float,  nargs=1, help="The threshold value for the Laplacian blur detection. The default value is: " )
    parser.add_argument("-css", "--cutStartSeconds", type=positive_int,  nargs=1, help="The number of seconds to cut from start of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: " )
    parser.add_argument("-ces", "--cutEndSeconds", type=positive_int,  nargs=1, help="The number of seconds to cut from the end of the video. These seconds of video will not be processed in the thumbnail selection. The default value is: " )
    numFrameExtractGroup = parser.add_mutually_exclusive_group(required = False)
    numFrameExtractGroup.add_argument("-nfe", "--numberOfFramesToExtract", type=above_zero_int,  nargs=1, help="Number of frames to be extracted from the video for the thumbnail selection process. The default is: " )
    numFrameExtractGroup.add_argument("-fre", "--framerateToExtract", type=restricted_float,  nargs=1, help="The framerate wanted to be extracted from the video for the thumbnail selection process.")
    numFrameExtractGroup.add_argument("-fpse", "--fpsExtract", type=above_zero_float,  nargs=1, help="Number of frames per second to extract from the video for the thumbnail selection process.")
    parser.add_argument("-ds", "--downscaleProcessingImages", type=restricted_float,  nargs=1, help="The value deciding how much the images to be processed should be downscaled. The default value is: " )
    parser.add_argument("-dso", "--downscaleOutputImage", type=restricted_float,  nargs=1, help="The value deciding how much the output thumbnail image should be downscaled. The default value is: " )
    parser.add_argument("-as", "--annotationSecond", type=positive_int,  nargs=1, help="The second the event is annotated to in the video.")
    parser.add_argument("-bac", "--beforeAnnotationSecondsCut", type=positive_int,  nargs=1, help="Seconds before the annotation to cut the frame extraction.")
    parser.add_argument("-aac", "--afterAnnotationSecondsCut", type=positive_int,  nargs=1, help="Seconds after the annotation to cut the frame extraction.")
    parser.add_argument("-st", "--staticThumbnailSec", type=positive_int,  nargs=1, help="To generate a static thumbnail from the video, this flag is used. The second the frame should be clipped from should follow as an argument. Running this flag ignores all the other flags.")
    parser.add_argument("-fn", "--outputFilename", type=str,  nargs=1, help="Filename for the output thumbnail instead of default.")
    args = parser.parse_args()
    configuration = args.configuration
    defaultConfigFilePath="../default-config.json" 
    localConfigFilePath="../input/config.json"
    configFilePath=configuration if os.path.exists(configuration) else localConfigFilePath if os.path.exists(localConfigFilePath) else defaultConfigFilePath 
    configFile = open(configFilePath)
    config = json.load(configFile)
    defaultConfigFile=open(defaultConfigFilePath)
    defaultConfig=json.load(defaultConfigFile)
    defaultConfig.update(config)
    finalConfig=defaultConfig
    outputJsonStructure={
    "config":finalConfig,
    "output":[],
    "performanceMetrics":{}}
    initialSummaryJson=json.dumps(outputJsonStructure)
    hostAtsSummary = open(f"../host-ats-output.json", "w")
    hostAtsSummary.write(initialSummaryJson)
    hostAtsSummary.close()
    

    #ROOT_DIR = os.path.abspath(os.curdir)
    #Default values
    absolute_path = os.path.pardir
    inputRelativePath = finalConfig["inputPath"]
    outputRelativePath = finalConfig["outputPath"]
    inputPath = os.path.join(absolute_path, inputRelativePath)
    outputPath = os.path.join(absolute_path, outputRelativePath)

    close_up_threshold = finalConfig["close_up_threshold"]
    totalFramesToExtract = finalConfig["totalFramesToExtract"]
    faceDetModel = finalConfig["faceDetModel"]
    framerateExtract = finalConfig["framerateExtract"]
    fpsExtract = finalConfig["fpsExtract"]
    cutStartSeconds = finalConfig["cutStartSeconds"]
    cutEndSeconds = finalConfig["cutEndSeconds"]
    downscaleOnProcessing = finalConfig["downscaleOnProcessing"]
    downscaleOutput = finalConfig["downscaleOutput"]
    annotationSecond = finalConfig["annotationSecond"]
    beforeAnnotationSecondsCut = finalConfig["beforeAnnotationSecondsCut"]
    afterAnnotationSecondsCut = finalConfig["afterAnnotationSecondsCut"]
    staticThumbnailSec = finalConfig["staticThumbnailSec"]
    logo_model_name = finalConfig["logo_model_name"]
    logo_threshold = finalConfig["logo_threshold"]
    close_up_model_name = finalConfig["close_up_model_name"]
    iqa_model_name = finalConfig["iqa_model_name"]
    brisque_threshold = finalConfig["brisque_threshold"]
    blur_model_name = finalConfig["blur_model_name"]
    svd_threshold = finalConfig["svd_threshold"]
    laplacian_threshold = finalConfig["laplacian_threshold"]
    filename_output = finalConfig["filename_output"]
    performanceAnalysis=finalConfig["performanceAnalysis"]
    thumbnailCount=finalConfig["thumbnailCount"]
    


    
    iteration=args.iteration
    if args.performanceAnalysis:
     performanceAnalysis = args.performanceAnalysis

    if args.thumbnailCount:
     thumbnailCount = args.thumbnailCount[0]

    if args.staticThumbnailSec:
     staticThumbnailSec = args.staticThumbnailSec[0]
    if args.outputFilename:
     filename_output = args.outputFilename[0]

    #Trimming
    if args.annotationSecond:
        annotationSecond = args.annotationSecond[0]
    if args.beforeAnnotationSecondsCut:
        beforeAnnotationSecondsCut = args.beforeAnnotationSecondsCut[0]
    if args.afterAnnotationSecondsCut:
        afterAnnotationSecondsCut = args.afterAnnotationSecondsCut[0]
    if args.cutStartSeconds:
            cutEndSeconds = args.cutEndSeconds[0]
    #Down-sampling
    if args.numberOfFramesToExtract:
        totalFramesToExtract = args.numberOfFramesToExtract[0]
    if args.framerateToExtract:
        framerateExtract = args.framerateToExtract[0]
    if args.fpsExtract:
        fpsExtract = args.fpsExtract[0]
    if fpsExtract:
        totalFramesToExtract = None
        framerateExtract = None
    if framerateExtract:
        totalFramesToExtract = None
        fpsExtract = None
    if totalFramesToExtract:
        framerateExtract = None
        fpsExtract = None
    #Down-scaling
    if args.downscaleProcessingImages:
        downscaleOnProcessing = args.downscaleProcessingImages[0]
    if args.downscaleOutputImage:
        downscaleOutput = args.downscaleOutputImage[0]

    #Logo detection
    runLogoDetection = args.xLogoDetection
    if not runLogoDetection and not finalConfig["logo_model_name"] :
        logo_model_name = ""
        runLogoDetection=False
    if args.LEliteserien2019:
        logo_model_name = eliteserienStr
    elif args.LSoccernet:
        logo_model_name = soccernetStr
    if args.logoThreshold:
        logo_threshold = args.logoThreshold[0]

    #Close-up detection
    runCloseUpDetection = args.xCloseupDetection
    if not runCloseUpDetection and not finalConfig["close_up_model_name"]:
        close_up_model_name = ""
        runCloseUpDetection=False
    if args.CSurma:
        close_up_model_name = surmaStr
    if args.closeUpThreshold:
        close_up_threshold = args.closeUpThreshold[0]

    #Face detection
    runFaceDetection = args.xFaceDetection
    if not runFaceDetection and not finalConfig["faceDetModel"]:
        faceDetModel = ""
        runFaceDetection=False
    if args.dlib:
        faceDetModel = dlibStr
    elif args.haar:
        faceDetModel = haarStr
    elif args.mtcnn:
        faceDetModel = mtcnnStr
    elif args.dnn:
        faceDetModel = dnnStr

    #Image Quality Assessment
    runIQA = args.xIQA
    if not runIQA or finalConfig["iqa_model_name"]=="":
        iqa_model_name = ""
        runIQA=False
    if args.IQAOcampo:
        iqa_model_name = ocampoStr
    if args.brisqueThreshold:
        brisque_threshold = args.brisqueThreshold[0]
    #Blur detection
    runBlur = args.xBlurDetection
    if not runBlur or  finalConfig["blur_model_name"]=="":
        blur_model_name = ""
        runBlur=False
    if args.BSVD:
        blur_model_name = svdStr
    elif args.BLaplacian:
        blur_model_name = laplacianStr
    if args.svdThreshold:
        svd_threshold = args.svdThreshold[0]
    if args.laplacianThreshold:
        laplacian_threshold = args.laplacianThreshold[0]

    # START SANITY CHECK -- WHICH MODULES ARE GOING TO RUN
    print("----------------------------")
    print(f"**SC** runBlur is set to {runBlur} with {blur_model_name}")
    print(f"**SC** runIQA is set to {runIQA} with {iqa_model_name}")
    print(f"**SC** runFaceDetection is set to {runFaceDetection} with {faceDetModel}")
    print(f"**SC** runCloseUpDetection is set to {runCloseUpDetection} with {close_up_model_name}")
    print(f"**SC** runLogoDetection is set to {runLogoDetection} with {logo_model_name}")
    print("----------------------------")




    processFolder = False
    processFile = False
    if os.path.isdir(inputPath):
        processFolder = True
        if inputPath[-1] != "/":
            inputPath = inputPath + "/"
        print("is folder")
    elif os.path.isfile(inputPath):
        processFile = True
        print("is file")
        name, ext = os.path.splitext(inputPath)
        if ext != ".ts" and ext != ".mp4" and ext != ".mkv":
            raise Exception("The input file is not a video file")
    else:
        raise Exception("The input destination was neither file or directory")

    try:
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)

    except OSError:
        print("Error: Couldn't create thumbnail output directory")
        return

    if staticThumbnailSec:
        get_static(inputPath, staticThumbnailSec, downscaleOutput, thumbnail_output)
        return
    loadingModelsStarts=time.time()
    if close_up_model_name == surmaStr:
        print("Loading Surma Model for close up detection module...")
        close_up_model = keras.models.load_model(surma_closeup_model)

    if logo_model_name == eliteserienStr:
        print("Loading eliteserien Model for logo detection module...")
        logo_detection_model = keras.models.load_model(eliteserien_logo_model)
    elif logo_model_name == soccernetStr:
        print("Loading soccernet Model for logo detection module...")
        logo_detection_model = keras.models.load_model(soccernet_logo_model)
    loadingModelsEnds=time.time()
    models_loading=loadingModelsEnds-loadingModelsStarts
    def logMetrics():
        total=frame_extraction+models_loading+logo_detection+closeup_detection+face_detection+blur_detection+iq_predicition
        performanceMetrics={
                "platform":platform.system(),
                "date_time":datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "cpu_logical":psutil.cpu_count(logical=True),
                "cpu_physical":psutil.cpu_count(logical=False),
                "cpu_max_freq(Mhz)":psutil.cpu_freq().max,
                "total_ram(GB)":"{0:.2f}".format(psutil.virtual_memory().total/1000000000),
                "available_ram(GB)":"{0:.2f}".format(psutil.virtual_memory().available/1000000000),
                "gpu_acceleration":tf.test.gpu_device_name() if tf.test.gpu_device_name() else "disabled",
                "args":"".join(sys.argv[1:]).split("-"),
                "framesToExtract":totalFramesToExtract,
                "downscaleOnProcessing":downscaleOnProcessing,
                "logo_detection_model":logo_model_name,
                "closeup_detection_model":close_up_model_name,
                "face_detection_model":faceDetModel,
                "blur_detection_model":blur_model_name,
                "iq_prediction_model":iqa_model_name,
                "frame_extraction_time":"{0:.3f}".format(frame_extraction) if frame_extraction>0 else "disabled",
                "logo_detection_time":"{0:.3f}".format(logo_detection)if logo_detection>0 else "disabled",
                "closeup_detection_time":"{0:.3f}".format(closeup_detection)if closeup_detection>0 else "disabled",
                "face_detection_time":"{0:.3f}".format(face_detection)if face_detection>0 else "disabled",
                "blur_detection_time":"{0:.3f}".format(blur_detection)if blur_detection>0 else "disabled",
                "iq_prediction_time":"{0:.3f}".format(iq_predicition)if iq_predicition>0 else "disabled",
                "models_loading_time":"{0:.3f}".format(models_loading) if models_loading>0 else "disabled",
                "total_execution_time":"{0:.3f}".format(total),
                "frames_extracted":numFramesExtracted,
                "iteration":iteration
                }
        performanceMetricsFile= open("../host-ats-output.json","r")
        data = json.load(performanceMetricsFile)
        performanceMetricsFile.close()
        performanceMetricsFile = open("../host-ats-output.json","w")
        data["performanceMetrics"]=performanceMetrics
        performanceMetricsFile.write(json.dumps(data))
        performanceMetricsFile.close()
    if processFile:
        create_thumbnail(name + ext, downscaleOutput, downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBlur, blur_model_name, svd_threshold, laplacian_threshold, runIQA, iqa_model_name, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, logo_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut, filename_output,outputPath,thumbnailCount)
        if performanceAnalysis==True:
             logMetrics()

        
    elif processFolder:
        for f in os.listdir(inputPath):
            name, ext = os.path.splitext(f)
            if ext == ".ts" or ext == ".mp4" or ext == ".mkv":
                create_thumbnail(inputPath + name + ext, downscaleOutput , downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBlur, blur_model_name, svd_threshold, laplacian_threshold, runIQA, iqa_model_name, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, logo_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut, filename_output,outputPath,thumbnailCount)
                if performanceAnalysis==True:
                    logMetrics()

    

def create_thumbnail(video_path, downscaleOutput, downscaleOnProcessing, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runBlur, blur_model_name, svd_threshold, laplacian_threshold, runIQA, iqa_model_name, runLogoDetection, runCloseUpDetection, close_up_threshold, brisque_threshold, logo_threshold, cutStartSeconds, cutEndSeconds, totalFramesToExtract, fpsExtract, framerateExtract, annotationSecond, beforeAnnotationSecondsCut, afterAnnotationSecondsCut, filename_output,outputPath,thumbnailCount):
    frameExtractionStarts=time.time()
    video_filename = video_path.split("/")[-1]
    frames_folder_outer=outputPath+"/temp/"+video_filename.split(".")[0]+"/"
    frames_folder = frames_folder_outer+"/frames/"
    if not os.path.exists(frames_folder_outer):
        os.makedirs(frames_folder_outer)
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    #frames_folder = frames_folder_outer + "/"

    # Read the video from specified path

    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    fps = cam.get(cv2.CAP_PROP_FPS)
    duration = totalFrames/fps

    if annotationSecond:
        if beforeAnnotationSecondsCut:
            cutStartSeconds = annotationSecond - beforeAnnotationSecondsCut
        if afterAnnotationSecondsCut:
            cutEndSeconds = duration - (annotationSecond + afterAnnotationSecondsCut)


    cutStartFrames = fps * cutStartSeconds
    cutEndFrames = fps * cutEndSeconds


    if totalFrames < cutStartFrames + cutEndFrames:
        raise Exception("All the frames are cut out")
    

    remainingFrames = totalFrames - (cutStartFrames + cutEndFrames)
    remainingSeconds = remainingFrames / fps



    if fpsExtract:
        totalFramesToExtract = math.floor(remainingSeconds * fpsExtract)
        print(f"Total frames to be extracted ==>{totalFramesToExtract} ")
    if framerateExtract:
        totalFramesToExtract = math.floor(remainingFrames * framerateExtract)
        print(f"Total frames to be extracted ==>{totalFramesToExtract} ")

    currentframe = 0
    # frames to skip
    frame_skip = (totalFrames-(cutStartFrames + cutEndFrames))//totalFramesToExtract
    global numFramesExtracted
    numFramesExtracted = 0
    stopFrame = totalFrames-cutEndFrames

    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe > stopFrame:
            break
        if currentframe < cutStartFrames:
            currentframe += 1
            continue
        if currentframe % frame_skip == 0 and numFramesExtracted < totalFramesToExtract:
            # if video is still left continue creating images
            name = frames_folder + 'frame' + str(currentframe) + '.jpg'
            #name = 'frame' + str(currentframe) + '.jpg'
            width = int(frame.shape[1] * downscaleOnProcessing)
            height = int(frame.shape[0] * downscaleOnProcessing)
            dsize = (width, height)
            img = cv2.resize(frame, dsize)
            cv2.imwrite(name, img)
            numFramesExtracted += 1
        currentframe += 1
        frameExtractionEnds=time.time()
           
    priority_images = groupFrames(frames_folder, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runLogoDetection, runCloseUpDetection, close_up_threshold, logo_threshold)
    for priority in priority_images:
        priority = dict(sorted(priority.items(), key=lambda item: item[1], reverse=True))
    #Drop 4th priority (logos)
    priority_images_no_logos=priority_images[:3]
    #Flatten all frames from different priorities
    merged_priorities=dict(ChainMap(*priority_images_no_logos)).keys()
    # 1-Frames that will be kept after running IQA AND blur detection.
    # 2-A frame is kept only if it passes the tests of the two modules.
    valid_frames = []
    for frame in merged_priorities:

        if runBlur and runIQA:
            blurDetectionStarts=time.time()
            if blur_model_name == svdStr:
                    blur_score = estimate_blur_svd(frame)
                    if blur_score > svd_threshold:
                        break
            if blur_model_name == laplacianStr:
                    blur_score = estimate_blur_laplacian(frame)
                    if blur_score < laplacian_threshold:
                        continue
            blurDetectionEnds=time.time()
            global blur_detection
            blur_detection=blurDetectionEnds-blurDetectionStarts
            IQAStarts=time.time()
            if iqa_model_name == ocampoStr:
                score = predictBrisque(frame)
                if score < brisque_threshold:
                    continue
                
            valid_frames.append(frame)

            IQAEnds=time.time()
            if runIQA:
                global iq_predicition
                iq_predicition=IQAEnds-IQAStarts
            
        else:
            # if blur and IQA modules are disabled, we keep the frame.
            for frame in merged_priorities:
                valid_frames.append(frame)


    outputFolder=outputPath+"/"+video_filename
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    numberOfOutputThumbnails=thumbnailCount if thumbnailCount >=1 else 1
    for item in valid_frames[:numberOfOutputThumbnails]:
                frameNumber=int(item.split("/")[-1].split(".")[0].replace("frame",""))
                cam.set(1, frameNumber)
                ret, frame = cam.read()
                if downscaleOutput != 1.0:
                    width = int(frame.shape[1] * downscaleOutput)
                    height = int(frame.shape[0] * downscaleOutput)
                    dsize = (width, height)
                    frame = cv2.resize(frame, dsize)
                newName = video_filename.split(".")[0] + "_" + filename_additional +"_"+ str(len(os.listdir(outputFolder)) )+ ".jpg"
                cv2.imwrite(os.path.join(outputFolder ,newName), frame)
                print("Thumbnails have been created." + newName)
    cam.release()
    cv2.destroyAllWindows()
    global frame_extraction
    frame_extraction=frameExtractionEnds-frameExtractionStarts
    return

def groupFrames(frames_folder, close_up_model, logo_detection_model, faceDetModel, runFaceDetection, runLogoDetection, runCloseUpDetection, close_up_threshold, logo_threshold):
    test_generator = None
    TEST_SIZE = 0
    faceDetectionStarts=0
    faceDetectionEnds=0
    if runCloseUpDetection or runLogoDetection:
        test_data_generator = ImageDataGenerator(rescale=1./255)
        IMAGE_SIZE = 200
        TEST_SIZE = len(next(os.walk(frames_folder))[2])
        IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
        test_generator = test_data_generator.flow_from_directory(
                frames_folder + "../",
                target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                batch_size=1,
                class_mode="binary",
                shuffle=False)

    logos = []
    if runLogoDetection:
        logoDetectionStarts=time.time()

        logo_probabilities = logo_detection_model.predict(test_generator, TEST_SIZE)

        for index, probability in enumerate(logo_probabilities):
            image_path = frames_folder + test_generator.filenames[index].split("/")[-1]
            if probability > logo_threshold:
                logos.append(image_path)
        logoDetectionEnds=time.time()
        if runLogoDetection:
             global logo_detection
             logo_detection=logoDetectionEnds-logoDetectionStarts
    priority_images = [{} for x in range(4)]
    if runCloseUpDetection:
        closeUpDetectionStarts=time.time()
        probabilities = close_up_model.predict(test_generator, TEST_SIZE)

        for index, probability in enumerate(probabilities):
            #The probability score is inverted:
            if close_up_model_inverted:
                probability = 1 - probability

            image_path = frames_folder + test_generator.filenames[index].split("/")[-1]

            if image_path in logos:
                priority_images[3][image_path] = probability

            elif probability > close_up_threshold:
                if runFaceDetection:
                    faceDetectionStarts=time.time()
                    face_size = detect_faces(image_path, faceDetModel)
                    if face_size > 0:
                        priority_images[0][image_path] = face_size
                    else:
                        priority_images[1][image_path] = probability
                    faceDetectionEnds=time.time()


                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[2][image_path] = probability
        closeUpDetectionEnds=time.time()
        if runCloseUpDetection:
            global closeup_detection
            closeup_detection=closeUpDetectionEnds-closeUpDetectionStarts
        if runFaceDetection:
            global face_detection
            face_detection=faceDetectionEnds-faceDetectionStarts

    else:
        probability = 1
        for image in os.listdir(frames_folder):
            image_path = frames_folder + image
            if image_path in logos:
                priority_images[3][image_path] = probability
            if runFaceDetection:
                face_size = detect_faces(image_path, faceDetModel)
                if face_size > 0:
                    priority_images[0][image_path] = face_size
                else:
                    priority_images[1][image_path] = probability
            else:
                priority_images[1][image_path] = probability
    return priority_images

def get_static(video_path, secondExtract, downscaleOutput, outputFolder):
    video_filename = video_path.split("/")[-1]
    #frames_folder_outer = os.path.dirname(os.path.abspath(__file__)) + "/extractedFrames/"
    frames_folder = frames_folder_outer + "/temp/"


    cam = cv2.VideoCapture(video_path)
    totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))-1
    fps = cam.get(cv2.CAP_PROP_FPS)

    duration = totalFrames/fps


    cutStartFrames = fps * secondExtract


    if totalFrames < cutStartFrames:
        raise Exception("All the frames are cut out")

    currentframe = 0
    while(True):
        # reading from frame
        ret,frame = cam.read()
        if not ret:
            break
        if currentframe <= cutStartFrames:
            currentframe += 1
            continue
        width = int(frame.shape[1] * downscaleOutput)
        height = int(frame.shape[0] * downscaleOutput)
        dsize = (width, height)
        img = cv2.resize(frame, dsize)
        newName = video_filename.split(".")[0] + "_static_thumbnail.jpg"
        cv2.imwrite(outputFolder + newName, img)
        break


def predictBrisque(image_path):
    img = cv2.imread(image_path)
    brisquePredictor = BRISQUE()
    brisqueScore = brisquePredictor.get_score(np.asarray(img))
    return brisqueScore

def estimate_blur_svd(image_file, sv_num=10):
    img = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
    u, s, v = np.linalg.svd(img)
    top_sv = np.sum(s[0:sv_num])
    total_sv = np.sum(s)
    return top_sv/total_sv


def estimate_blur_laplacian(image_file):
    #img = cv2.imread(
    img = cv2.imread(image_file,cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(img, cv2.CV_64F)
    score = np.var(blur_map)
    return score

def detect_faces(image, faceDetModel):
    biggestFace = 0
    if faceDetModel == dlibStr:
        detector = dlib.get_frontal_face_detector()
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        for result in faces:
            x = result.left()
            y = result.top()
            x1 = result.right()
            y1 = result.bottom()
            size = y1-y
            if biggestFace < size:
                biggestFace = size

    elif faceDetModel == haarStr:
        face_cascade = cv2.CascadeClassifier(haarXml)
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == mtcnnStr:
        detector = MTCNN()
        img = cv2.imread(image)
        faces = detector.detect_faces(img)

        for result in faces:
            x, y, w, h = result['box']
            size = h
            if biggestFace < h:
                biggestFace = h

    elif faceDetModel == dnnStr:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        img = cv2.imread(image)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                height = y1 - y
                if biggestFace < height:
                    biggestFace = height

    else:
        print("No face detection model in use")
    return biggestFace

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def above_zero_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x <=0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x

def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x < 0:
        raise argparse.ArgumentTypeError("%r not a positive int"%(x,))
    return x

def above_zero_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not an int literal" % (x,))
    if x <= 0:
        raise argparse.ArgumentTypeError("%r not above zero"%(x,))
    return x


if __name__ == "__main__":
    main()
