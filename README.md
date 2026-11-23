# Peking University Image Processing — Fall 2026, Homework 5

This interactive demo implements global histogram equalization and two-image histogram matching. It computes histogram counts, cumulative distributions, and intensity mappings with NumPy. OpenCV handles color-space conversion.

## Run locally

Use Python 3.11 and run these commands from the repository directory.

```sh
python -m pip install -r requirements.txt
python web_demo.py
```

Open [http://127.0.0.1:8088/](http://127.0.0.1:8088/) in a browser.

## Try it

- **Histogram equalization** spreads occupied intensity levels over the available range. Upload a low-contrast image to see the effect. A constant-lightness image is left unchanged.
- **Histogram matching** maps input intensity levels to reference levels with the closest cumulative probabilities. Upload an input and a reference image. Their dimensions can differ. Discrete intensity groups make the resulting distribution an approximation.

Both methods process grayscale intensity or the lightness component of HLS (hue, lightness, and saturation) for color images. Select a method and click **Run** to see the PNG output.
