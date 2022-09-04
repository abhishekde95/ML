import cv2
import numpy as np


def create_shorter_video(video_filepath: str, start: float, stop: float):
    """Creates a shorter video of the original one
    Parameters:
        video_filepath <str>: Filepath to the original video
        start <float>: Starting frame in terms of complete movie length
        stop <float>: Starting frame in terms of complete movie length
    """

    # Details from the original file
    video = cv2.VideoCapture(video_filepath)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start * total_frames)
    stop_frame = int(np.min([total_frames, int(stop * total_frames)]))

    # Setting up the output file
    if ".avi" in video_filepath:
        output_video_filepath = video_filepath.split(".avi")[0] + "_clipped.avi"
    elif ".mp4" in video_filepath:
        output_video_filepath = video_filepath.split(".mp4")[0] + "_clipped.mp4"
    else:
        raise RuntimeError("Should be either .mp4 or .avi file")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        filename=output_video_filepath,
        fourcc=_getVideoFourCC(video_filepath),
        fps=int(video.get(cv2.CAP_PROP_FPS)),
        frameSize=(frame_width, frame_height),
    )

    for i in range(start_frame, stop_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            raise RuntimeError(" Frame is corrupted")
        out.write(frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()


def _getVideoFourCC(filename: str) -> int:
    """Return the four-byte video codec to use for a given movie based on its file extension.
    Parameters:
         filename: path to an input video file
    """
    fourcc_mapping = {
        "mp4": int(cv2.VideoWriter_fourcc("m", "p", "4", "v")),
        "avi": int(cv2.VideoWriter_fourcc("m", "p", "4", "v")),
    }
    file_extension = filename.split(".")[-1].lower()
    if file_extension not in fourcc_mapping:
        raise ValueError(
            "COM-tracking only supports the following file extensions: {0}".format(
                fourcc_mapping.keys()
            )
        )
    else:
        return fourcc_mapping[file_extension]
