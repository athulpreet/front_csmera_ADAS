#!/usr/bin/env python3

"""
Copyright 2022-2024 NXP
SPDX-License-Identifier: BSD-3-Clause

Model: face_detection_ptq.tflite
Model licensed under Apache-2.0 License
Original model available at
https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite
Model card: https://mediapipe.page.link/blazeface-mc

Model: face_landmark_ptq.tflite
Model licensed under Apache-2.0 License
Original model available at https://storage.googleapis.com/mediapipe-assets/face_landmark.tflite
Model card: https://mediapipe.page.link/facemesh-mc

Model: iris_landmark_ptq.tflite
Model licensed under Apache-2.0 License
Original model available at https://storage.googleapis.com/mediapipe-assets/iris_landmark.tflite
Model card: https://mediapipe.page.link/iris-mc

Model: yolov4_tiny_smk_call.tflite
Model licensed under Apache-2.0 License
This model is trained by NXP.
Original model structure available at https://github.com/AlexeyAB/darknet/
"""

import os
import sys
import logging
import math
import time
import argparse
import numpy as np
import gi
import cairo
from face_detection import FaceDetector
from face_landmark import FaceLandmark
from eye import Eye
from mouth import Mouth
from smoking_calling_yolov4 import SmokingCallingDetector

# Set up logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
except Exception as e:
    logger.error(f"Failed to import GStreamer: {e}")
    sys.exit(1)

# Constants
DRAW_SMK_CALL_CORDS = False
DRAW_LANDMARKS = False
FRAME_WIDTH = 300
FRAME_HEIGHT = 300

# Penalties
BAD_FACE_PENALTY = 0.01
NO_FACE_PENALTY = 0.7
YAWN_PENALTY = 7.0
DISTRACT_PENALTY = 2.0
SLEEP_PENALTY = 5.0
SMK_PENALTY = 2.0
CALL_PENALTY = 2.0
RESTORE_CREDIT = -5.0

# Thresholds
FACE_THRESHOLD = 0.7
LEFT_EYE_THRESHOLD = 0.3
RIGHT_EYE_THRESHOLD = 0.3
MOUTH_THRESHOLD = 0.4
FACING_LEFT_THRESHOLD = 0.5
FACING_RIGHT_THRESHOLD = 2
SMK_CALL_THRESHOLD = 0.7

# Window sizes
LEFT_W = 3
RIGHT_W = 3

# Status arrays
LEFT_EYE_STATUS = np.zeros(LEFT_W)
RIGHT_EYE_STATUS = np.zeros(RIGHT_W)

class DMSDemo:
    """The class to run the DMS demo"""

    def __init__(self, video_device, inf_device, model_path):
        """Initialize the DMS Demo with terminal-only output"""
        try:
            logger.info("Starting DMS Demo initialization")
            
            # Basic initialization
            self.inited = False
            self.distracted = False
            self.drowsy = False
            self.yawn = False
            self.smoking = False
            self.phone = False
            self.face_cords = []
            self.marks = []
            self.safe_value = 0.0
            self.smk_call_cords = []
            self._last_status_print = 0

            # Platform detection
            logger.info("Detecting platform...")
            if os.path.exists("/usr/lib/libvx_delegate.so"):
                self.platform = "i.MX8MP"
                logger.info("Detected platform: i.MX8MP")
                videoconvert = "imxvideoconvert_g2d ! "
            elif os.path.exists("/usr/lib/libethosu_delegate.so"):
                self.platform = "i.MX93"
                logger.info("Detected platform: i.MX93")
                videoconvert = "imxvideoconvert_pxp ! "
            else:
                logger.error("Unsupported platform!")
                raise RuntimeError("Target platform not supported")

            # Check camera device
            if not os.path.exists(video_device):
                logger.error(f"Camera device not found: {video_device}")
                raise FileNotFoundError(f"Camera device not found: {video_device}")

            # Set up simplified GStreamer pipeline (terminal only)
            logger.info("Setting up GStreamer pipeline...")
            cam_pipeline = (
                f"v4l2src device={video_device} ! "
                f"video/x-raw,framerate=30/1,height=480,width=640,format=YUY2 ! "
                f"{videoconvert}"
                f"video/x-raw,height={FRAME_HEIGHT},width={FRAME_WIDTH},format=RGB16 ! "
                f"videoconvert ! video/x-raw,format=RGB ! "
                f"appsink emit-signals=true drop=true max-buffers=2 name=ml_sink"
            )

            # Create pipeline
            logger.info("Creating GStreamer pipeline...")
            logger.debug(f"Pipeline: {cam_pipeline}")
            
            pipeline = Gst.parse_launch(cam_pipeline)
            if not pipeline:
                raise RuntimeError("Failed to create GStreamer pipeline")

            # Get bus and connect error handler
            bus = pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message::error', self.on_error)
            bus.connect('message::warning', self.on_warning)
            
            # Connect ML sink
            logger.info("Connecting pipeline elements...")
            ml_sink = pipeline.get_by_name("ml_sink")
            if not ml_sink:
                raise RuntimeError("Failed to get ml_sink element")
            ml_sink.connect("new-sample", self.inference)

            # Set pipeline state
            logger.info("Setting pipeline state to PLAYING...")
            ret = pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to set pipeline to PLAYING state")
            
            # Store pipeline reference
            self.pipeline = pipeline

            # Initialize ML models
            logger.info("Initializing ML models...")
            self._init_models(model_path, inf_device)

            logger.info("DMS Demo initialization completed successfully")
            self.inited = True

        except Exception as e:
            logger.error(f"Failed to initialize DMS Demo: {e}", exc_info=True)
            raise

    def _init_models(self, model_path, inf_device):
        """Initialize ML models with error checking"""
        try:
            # Model paths
            if self.platform == "i.MX93" and inf_device == "NPU":
                face_model = os.path.join(model_path, "face_detection_ptq_vela.tflite")
                landmark_model = os.path.join(model_path, "face_landmark_ptq_vela.tflite")
                iris_model = os.path.join(model_path, "iris_landmark_ptq_vela.tflite")
                smk_call_model = os.path.join(model_path, "yolov4_tiny_smk_call_vela.tflite")
            else:
                face_model = os.path.join(model_path, "face_detection_ptq.tflite")
                landmark_model = os.path.join(model_path, "face_landmark_ptq.tflite")
                iris_model = os.path.join(model_path, "iris_landmark_ptq.tflite")
                smk_call_model = os.path.join(model_path, "yolov4_tiny_smk_call.tflite")

            # Check if model files exist
            for model_file in [face_model, landmark_model, iris_model, smk_call_model]:
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found: {model_file}")

            # Initialize detectors
            self.face_detector = FaceDetector(face_model, inf_device, self.platform, FACE_THRESHOLD)
            self.face_landmark = FaceLandmark(landmark_model, inf_device, self.platform)
            self.mouth = Mouth()
            self.eye = Eye(iris_model, inf_device, self.platform)
            self.smoking_calling_detector = SmokingCallingDetector(
                smk_call_model, inf_device, self.platform, conf=SMK_CALL_THRESHOLD
            )

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}", exc_info=True)
            raise

    def on_error(self, bus, message):
        """Handle GStreamer pipeline errors"""
        err, debug = message.parse_error()
        logger.error(f"GStreamer pipeline error: {err.message}")
        logger.debug(f"Debug info: {debug}")

    def on_warning(self, bus, message):
        """Handle GStreamer pipeline warnings"""
        warn, debug = message.parse_warning()
        logger.warning(f"GStreamer pipeline warning: {warn.message}")
        logger.debug(f"Debug info: {debug}")

    def print_status(self):
        """Print status to terminal with rate limiting"""
        current_time = time.time()
        if current_time - self._last_status_print < 0.5:  # Limit to 2 updates per second
            return
        
        self._last_status_print = current_time
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("\n=== Driver Monitoring System Status ===")
        print(f"Safety Score: {round(self.safe_value, 2)}%")
        
        if self.face_cords:
            if self.safe_value < 33:
                status = "Driver OK"
            elif self.safe_value < 66:
                status = "WARNING!"
            else:
                status = "DANGER!"
            print(f"Overall Status: {status}")
            
            print("\nDetailed Status:")
            print(f"{'Distracted:':<15} {'Yes' if self.distracted else 'No'}")
            print(f"{'Drowsy:':<15} {'Yes' if self.drowsy else 'No'}")
            print(f"{'Yawning:':<15} {'Yes' if self.yawn else 'No'}")
            print(f"{'Smoking:':<15} {'Yes' if self.smoking else 'No'}")
            print(f"{'On Phone:':<15} {'Yes' if self.phone else 'No'}")
        else:
            print("\nStatus: Driver Not Found!")
            
        print("=====================================\n")

    def inference(self, data):
        """Run inference with enhanced error handling and status display"""
        try:
            frame = data.emit("pull-sample")
            if frame is None or not self.inited:
                return 0

            buffer = frame.get_buffer()
            caps = frame.get_caps()
            ret, mem_buf = buffer.map(Gst.MapFlags.READ)
            height = caps.get_structure(0).get_value("height")
            width = caps.get_structure(0).get_value("width")
            frame = np.ndarray(
                shape=(height, width, 3), dtype=np.uint8, buffer=mem_buf.data
            )[..., ::-1]

            boxes = self.face_detector.detect(frame)
            face_cords = []
            smk_call_cords = []
            mark_group = []

            call = True
            smk = True
            attention = True
            yawn = True
            sleep = True

            if np.size(boxes, 0) > 0:
                # do smoking/calling detection
                smk_call_result = self.smoking_calling_detector.inference(frame, False)

                if np.size(smk_call_result, 0) > 0:
                    for i in range(np.size(smk_call_result, 0)):
                        if int(smk_call_result[i][5]) == 0:
                            call = False
                        elif int(smk_call_result[i][5]) == 1:
                            smk = False
                        x1 = int(smk_call_result[i][0])
                        y1 = int(smk_call_result[i][1])
                        x2 = int(smk_call_result[i][2])
                        y2 = int(smk_call_result[i][3])
                        smk_call_cords.append([x1, y1, x2, y2])

                for i in range(np.size(boxes, 0)):
                    boxes[i][[0, 2]] *= FRAME_WIDTH
                    boxes[i][[1, 3]] *= FRAME_HEIGHT

                # Transform the boxes into squares.
                boxes = self.transform_to_square(boxes, scale=1.26, offset=(0, 0))

                # Clip the boxes if they cross the image boundaries.
                boxes, _ = self.clip_boxes(boxes, (0, 0, FRAME_WIDTH, FRAME_HEIGHT))
                boxes = boxes.astype(np.int32)

                # only do landmark for one face closest to the center
                face_in_center = 0
                distance_to_center = math.hypot(FRAME_WIDTH / 2, FRAME_HEIGHT / 2)
                for i in range(np.size(boxes, 0)):
                    x1, y1, x2, y2 = boxes[i]
                    mid_to_center = math.hypot(
                        (x2 + x1 - FRAME_WIDTH) / 2, (y2 + y1 - FRAME_HEIGHT) / 2
                    )
                    if mid_to_center < distance_to_center:
                        face_in_center = i
                        distance_to_center = mid_to_center

                x1, y1, x2, y2 = boxes[face_in_center]
                face_cords.append([x1, y1, x2, y2])

                # now do face landmark inference
                face_image = frame[y1:y2, x1:x2]
                face_marks = self.face_landmark.get_landmark(face_image, (x1, y1, x2, y2))
                face_marks = np.array(face_marks)
                mark_group.append(face_marks)

                # process landmarks for left eye
                x1, y1, x2, y2 = self.eye.get_eye_roi(face_marks, 0)
                left_eye_image = frame[y1:y2, x1:x2]
                left_eye_marks, left_iris_marks = self.eye.get_landmark(
                    left_eye_image, (x1, y1, x2, y2), 0
                )
                mark_group.append(np.array(left_iris_marks))

                # process landmarks for right eye
                x1, y1, x2, y2 = self.eye.get_eye_roi(face_marks, 1)
                right_eye_image = frame[y1:y2, x1:x2]
                right_eye_marks, right_iris_marks = self.eye.get_landmark(
                    right_eye_image, (x1, y1, x2, y2), 1
                )
                mark_group.append(np.array(right_iris_marks))

                # process landmarks for eyes
                left_eye_ratio = self.eye.blinking_ratio(left_eye_marks, 0)
                right_eye_ratio = self.eye.blinking_ratio(right_eye_marks, 1)

                # average the left eye status in a window of LEFT_W frames
                for i in range(LEFT_W - 1):
                    LEFT_EYE_STATUS[i] = LEFT_EYE_STATUS[i + 1]

                if left_eye_ratio > LEFT_EYE_THRESHOLD:
                    LEFT_EYE_STATUS[LEFT_W - 1] = 1
                else:
                    LEFT_EYE_STATUS[LEFT_W - 1] = 0

                # average the right eye status in a window of RIGHT_W frames
                for i in range(RIGHT_W - 1):
                    RIGHT_EYE_STATUS[i] = RIGHT_EYE_STATUS[i + 1]

                if right_eye_ratio > RIGHT_EYE_THRESHOLD:
                    RIGHT_EYE_STATUS[RIGHT_W - 1] = 1
                else:
                    RIGHT_EYE_STATUS[RIGHT_W - 1] = 0

                if np.mean(LEFT_EYE_STATUS) < 0.5 and np.mean(RIGHT_EYE_STATUS) < 0.5:
                    sleep = False
                else:
                    sleep = True

                mouth_ratio = self.mouth.yawning_ratio(face_marks)
                if mouth_ratio > MOUTH_THRESHOLD:
                    yawn = False
                else:
                    yawn = True

                mouth_face_ratio = self.mouth.mouth_face_ratio(face_marks)
                if (
                    mouth_face_ratio < FACING_LEFT_THRESHOLD
                    or mouth_face_ratio > FACING_RIGHT_THRESHOLD
                ):
                    attention = False
                else:
                    attention = True
            else:
                face_cords = []

            self.marks = mark_group
            self.face_cords = face_cords
            self.smk_call_cords = smk_call_cords
            self.distracted = not attention
            self.drowsy = not sleep
            self.yawn = not yawn
            self.smoking = not smk
            self.phone = not call

            # Update safety score
            if not attention:
                self.safe_value = min(self.safe_value + DISTRACT_PENALTY, 100.00)
            if not sleep:
                self.safe_value = min(self.safe_value + SLEEP_PENALTY, 100.00)
            if not yawn:
                self.safe_value = min(self.safe_value + YAWN_PENALTY, 100.00)
            if not smk:
                self.safe_value = min(self.safe_value + SMK_PENALTY, 100.00)
            if not call:
                self.safe_value = min(self.safe_value + CALL_PENALTY, 100.00)
            if not face_cords:
                self.safe_value = min(self.safe_value + NO_FACE_PENALTY, 100.00)
            if attention and sleep and yawn and smk and call and face_cords:
                self.safe_value = max(self.safe_value + RESTORE_CREDIT, 0.00)

            # Print status to terminal
            self.print_status()

            buffer.unmap(mem_buf)
            return 0

        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
            return 1

    def draw(self, overlay, context, timestamp, duration):
        """Draw function for visualizing results"""
        try:
            context.select_font_face(
                "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            scale = 1072.00 / FRAME_HEIGHT
            offset = 840
            context.set_source_rgb(0, 1, 0)
            context.set_line_width(3)

            # draw smk_call_cords if enabled
            if self.smk_call_cords and DRAW_SMK_CALL_CORDS:
                for cords in self.smk_call_cords:
                    context.rectangle(
                        (cords[0] * scale) + offset,
                        (cords[1] * scale),
                        (cords[2] - cords[0]) * scale,
                        (cords[3] - cords[1]) * scale,
                    )
                context.stroke()

            # draw landmark points if enabled
            if self.marks and DRAW_LANDMARKS:
                for m in self.marks:
                    for mark in m:
                        mark = mark * scale
                        mark[0] = mark[0] + offset
                        point = tuple(mark.astype(int))
                        context.arc(point[0], point[1], 1, 0, 1)
                        context.stroke()

            # Draw status text and face box
            if self.face_cords:
                self.write_text(context, self.distracted, 410, 680)
                self.write_text(context, self.drowsy, 410, 765)
                self.write_text(context, self.yawn, 410, 850)
                self.write_text(context, self.smoking, 410, 935)
                self.write_text(context, self.phone, 410, 1020)
                context.set_source_rgb(0, 1, 0)
                context.rectangle(
                    (self.face_cords[0][0] * scale) + offset,
                    (self.face_cords[0][1] * scale),
                    (self.face_cords[0][2] - self.face_cords[0][0]) * scale,
                    (self.face_cords[0][3] - self.face_cords[0][1]) * scale,
                )
                context.stroke()
            else:
                self.write_text(context, None, 410, 680)
                self.write_text(context, None, 410, 765)
                self.write_text(context, None, 410, 850)
                self.write_text(context, None, 410, 935)
                self.write_text(context, None, 410, 1020)

            self.write_status(context)

        except Exception as e:
            logger.error(f"Error in draw function: {e}", exc_info=True)

    def write_text(self, context, yes, y, x):
        """Write text on the display"""
        context.set_font_size(int(45.0))
        context.move_to(y, x)
        if yes is None:
            context.set_source_rgb(0, 0, 0)
            context.show_text("N/A")
            return
        if yes:
            context.set_source_rgb(1, 0, 0)
            context.show_text("Yes")
        else:
            context.set_source_rgb(0, 1, 0)
            context.show_text("No")

    def write_status(self, context):
        """Write driver's status on the display"""
        context.set_font_size(int(60.0))
        context.move_to(25, 600)
        r = min(self.safe_value / 50.0, 1.0)
        g = min(1.0, (100.0 - self.safe_value) / 50.0)
        b = 0
        context.set_source_rgb(r, g, b)
        if self.face_cords:
            if self.safe_value < 33:
                context.show_text("Driver OK (" + str(round(self.safe_value, 2)) + "%)")
            elif self.safe_value < 66:
                context.show_text("Warning! (" + str(round(self.safe_value, 2)) + "%)")
            else:
                context.show_text("Danger! (" + str(round(self.safe_value, 2)) + "%)")
        else:
            context.show_text(
                "Driver not found! (" + str(round(self.safe_value, 2)) + "%)"
            )

    def transform_to_square(self, boxes, scale=1.0, offset=(0, 0)):
        """Transform boxes to squares"""
        xmins, ymins, xmaxs, ymaxs = np.split(boxes, 4, 1)
        width = xmaxs - xmins
        height = ymaxs - ymins

        offset_x = offset[0] * width
        offset_y = offset[1] * height

        center_x = np.floor_divide(xmins + xmaxs, 2) + offset_x
        center_y = np.floor_divide(ymins + ymaxs, 2) + offset_y

        margin = np.floor_divide(np.maximum(height, width) * scale, 2)
        boxes = np.concatenate(
            (
                center_x - margin,
                center_y - margin,
                center_x + margin,
                center_y + margin,
            ),
            axis=1,
        )

        return boxes

    def clip_boxes(self, boxes, margins):
        """Clip boxes to safe margins"""
        left, top, right, bottom = margins

        clip_mark = (
            boxes[:, 1] < top,
            boxes[:, 0] < left,
            boxes[:, 3] > bottom,
            boxes[:, 2] > right,
        )

        boxes[:, 1] = np.maximum(boxes[:, 1], top)
        boxes[:, 0] = np.maximum(boxes[:, 0], left)
        boxes[:, 3] = np.minimum(boxes[:, 3], bottom)
        boxes[:, 2] = np.minimum(boxes[:, 2], right)

        return boxes, clip_mark

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

def main():
    """Main function with enhanced error handling"""
    dms = None
    try:
        logger.info("Starting DMS Demo application")
        
        # Parse arguments
        parser = argparse.ArgumentParser(description="DMS Demo")
        parser.add_argument("--device", type=str, default="/dev/video0", 
                          help="Camera device to be used")
        parser.add_argument("--backend", type=str, default="NPU", 
                          help="Use NPU or CPU to do inference")
        parser.add_argument("--model_path", type=str, default=os.path.dirname(os.path.abspath(__file__)), 
                          help="Path for models and image")
        args = parser.parse_args()

        # Log configuration
        logger.info(f"Configuration:")
        logger.info(f"  Camera Device: {args.device}")
        logger.info(f"  Backend: {args.backend}")
        logger.info(f"  Model Path: {args.model_path}")

        # Initialize GStreamer
        logger.info("Initializing GStreamer...")
        Gst.init(None)

        # Create DMS Demo instance
        logger.info("Creating DMS Demo instance...")
        dms = DMSDemo(args.device, args.backend, args.model_path)

        # Main loop
        logger.info("Entering main loop...")
        print("\nDMS Demo is running. Press 'q' to exit.")
        
        while True:
            try:
                if input().lower() == 'q':
                    logger.info("Quit command received")
                    break
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                break

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        return 1
    finally:
        if dms:
            logger.info("Cleaning up...")
            dms.cleanup()
        logger.info("DMS Demo shutting down")
    return 0

if __name__ == "__main__":
    # Set up environment
    os.environ["XDG_RUNTIME_DIR"] = "/run/user/0"
    
    # Run main with error handling
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Uncaught exception: {e}", exc_info=True)
        sys.exit(1)
