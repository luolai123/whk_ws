#!/usr/bin/env python3
"""Inference node that loads a trained PyTorch model and publishes classification results."""

import pathlib

import numpy as np
import rospy
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class DistanceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class InferenceNode:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = rospy.get_param("~model_path", str(pathlib.Path.home() / "autonomy_demo" / "model.pt"))
        self.model = DistanceClassifier()
        if pathlib.Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            rospy.loginfo("Loaded model from %s", model_path)
        else:
            rospy.logwarn("Model path %s does not exist - running with random weights", model_path)
        self.model.to(self.device)
        self.model.eval()

        self.softmax = torch.nn.Softmax(dim=1)
        self.image_sub = rospy.Subscriber("drone/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.label_pub = rospy.Publisher("drone/rgb/distance_class", Image, queue_size=1)

    def image_callback(self, msg: Image) -> None:
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        tensor = torch.from_numpy(cv_image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = self.softmax(logits)
        classes = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        # Map classes to RGB colors: 1 (near) -> red, 0 (far) -> green
        color_map = np.zeros((*classes.shape, 3), dtype=np.uint8)
        color_map[:, :, 1] = np.where(classes == 0, 200, 0)
        color_map[:, :, 0] = np.where(classes == 1, 200, 0)
        label_msg = self.bridge.cv2_to_imgmsg(color_map, encoding="rgb8")
        label_msg.header = msg.header
        label_msg.header.frame_id = rospy.get_param("~output_frame", msg.header.frame_id)
        self.label_pub.publish(label_msg)


def main() -> None:
    rospy.init_node("distance_inference")
    InferenceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
