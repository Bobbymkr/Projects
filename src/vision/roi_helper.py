"""
Interactive ROI Configuration Helper

This module provides utilities to create, edit, and visualize ROI configurations
for traffic queue estimation. It can work with static images or live video feeds
to help users define regions of interest interactively.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from .video_pipeline import VideoConfig, VideoInputStream, VideoSourceType
from .yolo_queue import ROIConfig


class ROIEditor:
    """Interactive ROI editor using OpenCV mouse callbacks."""
    
    def __init__(self, frame: np.ndarray):
        """Initialize the ROI editor with a reference frame.
        
        Args:
            frame: Reference frame for ROI definition
        """
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.rois: List[ROIConfig] = []
        self.current_roi_points: List[Tuple[int, int]] = []
        self.drawing = False
        self.current_roi_name = ""
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.drawing:
                self.drawing = True
                self.current_roi_points = [(x, y)]
            else:
                self.current_roi_points.append((x, y))
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.drawing and len(self.current_roi_points) >= 3:
                # Finish current ROI
                self._finish_current_roi()
                
    def _finish_current_roi(self):
        """Complete the current ROI and add it to the list."""
        if len(self.current_roi_points) >= 3:
            # Create ROI name if not set
            if not self.current_roi_name:
                self.current_roi_name = f"roi_{len(self.rois) + 1}"
                
            roi = ROIConfig(
                name=self.current_roi_name,
                points=self.current_roi_points,
                max_queue_length=50,  # Default value
                direction="north"  # Default direction
            )
            self.rois.append(roi)
            print(f"Added ROI: {self.current_roi_name} with {len(self.current_roi_points)} points")
            
            # Reset for next ROI
            self.current_roi_points = []
            self.drawing = False
            self.current_roi_name = ""
            
    def _draw_rois(self):
        """Draw all ROIs on the frame."""
        display_frame = self.original_frame.copy()
        
        # Draw completed ROIs
        for i, roi in enumerate(self.rois):
            points = np.array(roi.points, dtype=np.int32)
            cv2.polylines(display_frame, [points], True, (0, 255, 0), 2)
            cv2.fillPoly(display_frame, [points], (0, 255, 0, 50))
            
            # Add ROI label
            centroid = np.mean(points, axis=0).astype(int)
            cv2.putText(display_frame, roi.name, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
        # Draw current ROI being drawn
        if self.drawing and len(self.current_roi_points) > 0:
            points = np.array(self.current_roi_points, dtype=np.int32)
            if len(self.current_roi_points) > 1:
                cv2.polylines(display_frame, [points], False, (0, 0, 255), 2)
            for point in self.current_roi_points:
                cv2.circle(display_frame, point, 4, (0, 0, 255), -1)
                
        return display_frame
        
    def edit_rois(self, window_name: str = "ROI Editor") -> List[ROIConfig]:
        """Start interactive ROI editing session.
        
        Args:
            window_name: Name of the OpenCV window
            
        Returns:
            List of configured ROIs
        """
        cv2.namedWindow(window_name, cv2.WINDOW_RESIZABLE)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("ROI Editor Instructions:")
        print("- Left click to add points to current ROI")
        print("- Right click to finish current ROI (minimum 3 points)")
        print("- Press 'r' to reset current ROI")
        print("- Press 'd' to delete last ROI")
        print("- Press 'n' to set name for next ROI")
        print("- Press 'q' to quit and save")
        print("- Press ESC to quit without saving")
        
        while True:
            display_frame = self._draw_rois()
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Finish any current ROI if it has enough points
                if self.drawing and len(self.current_roi_points) >= 3:
                    self._finish_current_roi()
                break
            elif key == 27:  # ESC key
                self.rois = []  # Clear ROIs if user cancels
                break
            elif key == ord('r'):
                # Reset current ROI
                self.current_roi_points = []
                self.drawing = False
                print("Reset current ROI")
            elif key == ord('d'):
                # Delete last ROI
                if self.rois:
                    deleted = self.rois.pop()
                    print(f"Deleted ROI: {deleted.name}")
            elif key == ord('n'):
                # Set name for next ROI
                print("Enter name for next ROI:")
                # Note: In a full implementation, you'd want a proper text input
                # For now, we'll use a simple counter-based naming
                self.current_roi_name = f"lane_{len(self.rois) + 1}"
                print(f"Next ROI will be named: {self.current_roi_name}")
                
        cv2.destroyWindow(window_name)
        return self.rois


def create_roi_config_interactive(video_source: str, 
                                config_path: str,
                                fps: float = 15.0,
                                width: int = 640,
                                height: int = 480) -> bool:
    """Create ROI configuration interactively using video source.
    
    Args:
        video_source: Video source (webcam index, file path, or stream URL)
        config_path: Path to save the ROI configuration
        fps: Target FPS for video processing
        width: Frame width
        height: Frame height
        
    Returns:
        True if configuration was created successfully, False otherwise
    """
    try:
        # Parse video source
        if video_source.isdigit():
            source_type = VideoSourceType.WEBCAM
            source = int(video_source)
        elif video_source.startswith(('http://', 'https://', 'rtsp://')):
            source_type = VideoSourceType.RTSP if video_source.startswith('rtsp://') else VideoSourceType.HTTP
            source = video_source
        elif Path(video_source).exists():
            source_type = VideoSourceType.FILE
            source = video_source
        else:
            print(f"Invalid video source: {video_source}")
            return False
            
        # Create video config and stream
        video_config = VideoConfig(
            source=source,
            source_type=source_type,
            target_fps=fps,
            frame_width=width,
            frame_height=height,
            buffer_size=10,
            auto_resize=True,
            recording_enabled=False
        )
        
        video_stream = VideoInputStream(video_config)
        
        if not video_stream.start():
            print("Failed to start video stream")
            return False
            
        # Get a reference frame
        print("Capturing reference frame...")
        import time
        time.sleep(1)  # Let stream stabilize
        
        frame = video_stream.get_frame()
        if frame is None:
            print("Failed to capture frame")
            video_stream.stop()
            return False
            
        video_stream.stop()
        
        # Start ROI editing
        editor = ROIEditor(frame)
        rois = editor.edit_rois()
        
        if not rois:
            print("No ROIs created")
            return False
            
        # Save configuration
        roi_data = {
            "rois": [asdict(roi) for roi in rois],
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
            "created_from_source": video_source
        }
        
        config_dir = Path(config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(roi_data, f, indent=2)
            
        print(f"ROI configuration saved to: {config_path}")
        print(f"Created {len(rois)} ROIs:")
        for roi in rois:
            print(f"  - {roi.name}: {len(roi.points)} points")
            
        return True
        
    except Exception as e:
        print(f"Error creating ROI configuration: {e}")
        return False


def visualize_roi_config(config_path: str, video_source: str, 
                        fps: float = 15.0, width: int = 640, height: int = 480):
    """Visualize existing ROI configuration on live video.
    
    Args:
        config_path: Path to ROI configuration file
        video_source: Video source for visualization
        fps: Target FPS
        width: Frame width  
        height: Frame height
    """
    try:
        # Load ROI configuration
        with open(config_path, 'r') as f:
            roi_data = json.load(f)
            
        rois = [ROIConfig(**roi_dict) for roi_dict in roi_data['rois']]
        
        # Setup video stream
        if video_source.isdigit():
            source_type = VideoSourceType.WEBCAM
            source = int(video_source)
        elif video_source.startswith(('http://', 'https://', 'rtsp://')):
            source_type = VideoSourceType.RTSP if video_source.startswith('rtsp://') else VideoSourceType.HTTP
            source = video_source
        elif Path(video_source).exists():
            source_type = VideoSourceType.FILE
            source = video_source
        else:
            print(f"Invalid video source: {video_source}")
            return
            
        video_config = VideoConfig(
            source=source,
            source_type=source_type,
            target_fps=fps,
            frame_width=width,
            frame_height=height,
            buffer_size=10,
            auto_resize=True,
            recording_enabled=False
        )
        
        video_stream = VideoInputStream(video_config)
        
        if not video_stream.start():
            print("Failed to start video stream")
            return
            
        print("Visualizing ROI configuration...")
        print("Press 'q' to quit")
        
        window_name = "ROI Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_RESIZABLE)
        
        while True:
            frame = video_stream.get_frame()
            if frame is None:
                continue
                
            # Draw ROIs
            for roi in rois:
                points = np.array(roi.points, dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                cv2.fillPoly(frame, [points], (0, 255, 0, 30))
                
                # Add ROI label
                centroid = np.mean(points, axis=0).astype(int)
                cv2.putText(frame, f"{roi.name} ({roi.direction})", tuple(centroid),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                           
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyWindow(window_name)
        video_stream.stop()
        
    except Exception as e:
        print(f"Error visualizing ROI configuration: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ROI Configuration Helper")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Create ROI config
    create_parser = subparsers.add_parser('create', help='Create new ROI configuration')
    create_parser.add_argument('--video_source', required=True, help='Video source (webcam, file, or stream)')
    create_parser.add_argument('--config_path', required=True, help='Output path for ROI config JSON')
    create_parser.add_argument('--fps', type=float, default=15.0, help='Target FPS')
    create_parser.add_argument('--width', type=int, default=640, help='Frame width')
    create_parser.add_argument('--height', type=int, default=480, help='Frame height')
    
    # Visualize ROI config
    viz_parser = subparsers.add_parser('visualize', help='Visualize existing ROI configuration')
    viz_parser.add_argument('--config_path', required=True, help='Path to ROI config JSON')
    viz_parser.add_argument('--video_source', required=True, help='Video source for visualization')
    viz_parser.add_argument('--fps', type=float, default=15.0, help='Target FPS')
    viz_parser.add_argument('--width', type=int, default=640, help='Frame width')
    viz_parser.add_argument('--height', type=int, default=480, help='Frame height')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        success = create_roi_config_interactive(
            args.video_source, args.config_path, args.fps, args.width, args.height
        )
        if success:
            print("ROI configuration created successfully!")
        else:
            print("Failed to create ROI configuration")
            
    elif args.command == 'visualize':
        visualize_roi_config(
            args.config_path, args.video_source, args.fps, args.width, args.height
        )