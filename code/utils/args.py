import argparse
from utils.num import above_zero_float,positive_int,above_zero_int,restricted_float

def parseArguments():
    parser = argparse.ArgumentParser(description="Thumbnail generator")
    #General
    parser.add_argument('-conf','--configuration', type=str,default="../config.json", required=False ,help="Full path to a configuration file formatted as JSON.")
    parser.add_argument('-iter','--iteration', type=int, required=False,help='Number of executions per configuration')
    parser.add_argument('-pa','--performanceAnalysis', type=bool,default=False,  required=False ,help="Run performance analyis.")
    parser.add_argument("-tc", "--thumbnailCount", type=above_zero_int,  nargs=1, help="Number of desired output thumbnail candidates " )

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

    #Preprocessing
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

    return parser.parse_args()