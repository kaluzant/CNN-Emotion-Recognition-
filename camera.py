import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from train import EmotionCNN, IMAGE_SIZE


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[EmotionCNN, list[str]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = checkpoint.get("classes")
    if not isinstance(classes, list) or not classes or not all(
        isinstance(name, str) for name in classes
    ):
        raise ValueError(f"Checkpoint at {checkpoint_path} has invalid class names.")
    if checkpoint.get("image_size") != IMAGE_SIZE:
        raise ValueError(
            f"Checkpoint image size must be {IMAGE_SIZE}, "
            f"got {checkpoint.get('image_size')}."
        )

    model = EmotionCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


def preprocess_face(grayscale_face: np.ndarray) -> torch.Tensor:
    resized = Image.fromarray(grayscale_face).resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.array(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)


class GradCAM:
    def __init__(self, model: EmotionCNN):
        self.model = model
        convolution_layers = [
            layer
            for layer in model.features.modules()
            if isinstance(layer, torch.nn.Conv2d)
        ]
        if not convolution_layers:
            raise ValueError("The model has no convolutional layer for Grad-CAM.")
        self.activation: torch.Tensor | None = None
        self.hook = convolution_layers[-1].register_forward_hook(self._capture)

    def _capture(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activation = output

    @torch.enable_grad()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
        self.activation = None
        logits = self.model(images)
        if self.activation is None:
            raise RuntimeError("Grad-CAM did not capture the model features.")

        class_index = int(logits[0].argmax().item())
        gradients = torch.autograd.grad(
            logits[0, class_index], self.activation
        )[0]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * self.activation).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(
            heatmap,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heatmap_min = heatmap.min()
        heatmap = (heatmap - heatmap_min) / (
            heatmap.max() - heatmap_min
        ).clamp_min(1e-8)
        probabilities = torch.softmax(logits[0], dim=0).detach()
        return probabilities, heatmap.detach().cpu().numpy()

    def close(self) -> None:
        self.hook.remove()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show live facial emotion predictions from your webcam."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "models" / "fer_cnn.pth",
        help="model checkpoint to load",
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(args.checkpoint, device)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_detector.empty():
        raise RuntimeError("Could not load OpenCV's frontal-face detector.")

    grad_cam = GradCAM(model)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        grad_cam.close()
        raise RuntimeError(
            f"Could not open webcam {args.camera}. Check the camera index and permissions."
        )

    window_name = "Facial Emotion Recognition (press q to quit)"
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                grayscale,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )

            if len(faces) == 0:
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            for x, y, width, height in faces:
                face = preprocess_face(grayscale[y : y + height, x : x + width])
                probabilities, heatmap = grad_cam.predict(face.to(device))
                class_index = int(probabilities.argmax().item())
                emotion = classes[class_index]
                confidence = probabilities[class_index].item()

                heatmap_image = cv2.applyColorMap(
                    np.uint8(heatmap * 255), cv2.COLORMAP_JET
                )
                face_region = frame[y : y + height, x : x + width]
                heatmap_image = cv2.resize(
                    heatmap_image, (width, height), interpolation=cv2.INTER_LINEAR
                )
                frame[y : y + height, x : x + width] = cv2.addWeighted(
                    face_region, 0.55, heatmap_image, 0.45, 0
                )
                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"{emotion}: {confidence:.0%}",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        grad_cam.close()


if __name__ == "__main__":
    main()
