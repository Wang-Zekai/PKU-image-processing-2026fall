# Peking University Image Processing — Fall 2026

A collection of the five homework demos for the Fall 2026 Image Processing course at Peking University. The assignments share one Gradio interface.

## Assignments

| Homework | Demo |
| --- | --- |
| 1 | Adjust hue, saturation, lightness, brightness, and contrast. Compare manual and OpenCV HLS (hue, lightness, and saturation) conversion, and try vibrance adjustment. |
| 2 | Generate 64 × 64 faces with a pretrained DCGAN (deep convolutional generative adversarial network), then edit attributes using differences between groups of latent vectors. |
| 3 | Replace a screen in an image or video with an uploaded image. Set the four screen corners in the image or first video frame. |
| 4 | Sharpen and denoise images with guided and bilateral filters, an integral-image implementation of guided filtering, and two-stage BM3D (block-matching and 3D filtering). |
| 5 | Implement global histogram equalization and two-image histogram matching. Both methods process grayscale values or the lightness component of color images. |

## Run locally

Use Python 3.11. In the repository directory, run:

```sh
python -m pip install -r requirements.txt
python web_demo.py
```

Open [http://127.0.0.1:8088/](http://127.0.0.1:8088/) and choose a homework tab.

## Models and sample media

- Homework 2 requires PyTorch and the pretrained checkpoint supplied with the coursework. Install PyTorch with `python -m pip install torch` and put the checkpoint at `5 图像生成模型上 DCGAN代码示例/checkpoint/dcgan_checkpoint.pth` relative to the repository root.
- Homework 3 accepts your own images and video. The original sample media can optionally be placed in the `作业3测试图` directory. Installing FFmpeg adds H.264 export when the encoder is available.
- The repository contains the application source and Python dependency list. Course checkpoints and sample media are supplied separately.
