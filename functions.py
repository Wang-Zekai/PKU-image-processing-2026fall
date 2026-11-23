# -*- coding: utf-8 -*-
import numpy as np
import gradio as gr


def function_hw1(input_image, mode, h_slider, s_slider, l_slider, brightness, contrast,
                 vibrance=0, hls_method="手动实现", protect_skin=False, comparison=False):
    """HW1: all manual color math and both advanced features live in this function.

    comparison=True returns the two reconstructed images, amplified error map
    and numerical table. Default calls retain the original seven-image output.
    Vibrance uses a sourced SweetFX algorithm with documented bounds/extensions.
    """
    import cv2

    def normalized_rgb(image):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
            raise ValueError('Expected a nonempty H x W x 3 RGB image.')
        if array.dtype == np.uint8:
            return array.astype(np.float64) / 255.0
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError('RGB must be uint8 or normalized floating point.')
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError('Floating point RGB must be finite and in [0,1].')
        return array.astype(np.float64, copy=True)

    def _hls_array(hls):
        array = np.asarray(hls, dtype=np.float64)
        if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
            raise ValueError('Expected a nonempty H x W x 3 HLS array.')
        # HLS is produced internally from RGB already validated in [0,1].
        # OpenCV float32 saturation can overshoot substantially near white.
        # Clip only the processing copy; comparison metrics retain raw HLS.
        if not np.isfinite(array).all():
            raise ValueError('HLS must be finite.')
        result = array.copy()
        result[..., 1:] = np.clip(result[..., 1:], 0, 1)
        return result

    def _amount(value):
        value = float(value)
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Adjustment amount must be finite and in [-1,1].')
        return value

    def to_uint8(array):
        """Round normalized values instead of truncating; clip before conversion."""
        return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)

    def manual_rgb_to_hls(image):
        rgb = normalized_rgb(image)
        r, g, b = np.moveaxis(rgb, -1, 0)
        maximum, minimum = rgb.max(axis=-1), rgb.min(axis=-1)
        delta = maximum - minimum
        lightness = (maximum + minimum) / 2
        denominator = np.where(lightness <= .5, maximum + minimum, 2 - maximum - minimum)
        saturation = np.divide(delta, denominator, out=np.zeros_like(delta), where=delta > 0)
        # Safe denominators avoid NaNs for gray, white and black pixels.
        safe_delta = np.where(delta > 0, delta, 1)
        hue = np.where(maximum == r, (g - b) / safe_delta,
                       np.where(maximum == g, 2 + (b - r) / safe_delta, 4 + (r - g) / safe_delta))
        hue = np.where(delta > 0, (hue * 60) % 360, 0)
        return np.stack((hue, lightness, saturation), axis=-1)

    def manual_hls_to_rgb(hls):
        array = _hls_array(hls)
        hue, lightness, saturation = np.moveaxis(array, -1, 0)
        h = (hue % 360) / 60
        chroma = (1 - np.abs(2 * lightness - 1)) * saturation
        x = chroma * (1 - np.abs(h % 2 - 1))
        zero = np.zeros_like(chroma)
        sector = np.floor(h).astype(np.intp)
        r = np.choose(sector, (chroma, x, zero, zero, x, chroma))
        g = np.choose(sector, (x, chroma, chroma, x, zero, zero))
        b = np.choose(sector, (zero, zero, x, chroma, chroma, x))
        offset = lightness - chroma / 2
        return np.clip(np.stack((r, g, b), axis=-1) + offset[..., None], 0, 1)

    def rgb_to_hls(image, method='手动实现'):
        if method == '手动实现':
            return manual_rgb_to_hls(image)
        if method == 'OpenCV':
            rgb = normalized_rgb(image)
            # Gradio supplies RGB. Explicitly swap to BGR for COLOR_BGR2HLS.
            bgr = np.ascontiguousarray(rgb[..., ::-1], dtype=np.float32)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float64)
        raise ValueError('Unknown HLS conversion method.')

    def hls_to_rgb(hls, method='手动实现'):
        if method == '手动实现':
            return manual_hls_to_rgb(hls)
        if method == 'OpenCV':
            array = _hls_array(hls)
            array[..., 0] %= 360
            return np.clip(cv2.cvtColor(array.astype(np.float32), cv2.COLOR_HLS2RGB), 0, 1)
        raise ValueError('Unknown HLS conversion method.')

    def _towards_endpoint(values, amount):
        return values + amount * (1 - values) if amount >= 0 else values * (1 + amount)

    def adjust_hls(hls, hue=0, saturation=0, lightness=0):
        result = _hls_array(hls)
        hue, saturation, lightness = (_amount(value) for value in (hue, saturation, lightness))
        result[..., 0] = (result[..., 0] + 180 * hue) % 360
        # Multiplicative saturation preserves neutral colors rather than tinting gray.
        result[..., 2] = np.clip(result[..., 2] * (1 + saturation), 0, 1)
        result[..., 1] = _towards_endpoint(result[..., 1], lightness)
        return result

    def apply_vibrance(image, amount=0, protect_skin=False):
        """NumPy port of CeeJay.dk's SweetFX Vibrance 1.1.1, RGB balance=(1,1,1).

        Source: https://github.com/CeeJayDK/SweetFX/blob/master/Shaders/SweetFX/Vibrance.fx
        SHA256: b9189a28ca4a645a0e188f8396c81889d217e3c706e8900dfe6594d43e7c33eb
        Source gain: 1 + v*(1-sign(v)*(max(RGB)-min(RGB))).
        Declared adaptations: clamp negative gain to0 to avoid hue inversion;
        clip final RGB to[0,1] for image output. Skin weighting is an optional
        custom extension, disabled by default, not part of SweetFX/Photoshop.
        """
        rgb = normalized_rgb(image)
        amount = _amount(amount)
        if amount == 0:
            return rgb
        luma = np.sum(rgb * np.array([0.212656, 0.715158, 0.072186]), axis=-1)
        chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
        strength = np.full(chroma.shape, amount, dtype=np.float64)
        if protect_skin:
            hue = manual_rgb_to_hls(rgb)[..., 0]
            distance = np.abs((hue - 30 + 180) % 360 - 180)
            strength *= 1 - .65 * np.exp(-.5 * (distance / 18) ** 2)
        gain = 1 + strength * (1 - np.sign(strength) * chroma)
        gain = np.maximum(gain, 0)
        output = luma[..., None] + gain[..., None] * (rgb - luma[..., None])
        return np.clip(output, 0, 1)

    def adjust_brightness_contrast(image, brightness=0, contrast=0):
        """Basic centered contrast followed by brightness; no Photoshop fitting claim."""
        rgb = normalized_rgb(image)
        brightness, contrast = _amount(brightness), _amount(contrast)
        if contrast == 1:
            rgb = np.where(rgb > .5, 1.0, np.where(rgb < .5, 0.0, .5))
        else:
            gain = 1 + contrast if contrast <= 0 else 1 / (1 - contrast)
            rgb = np.clip((rgb - .5) * gain + .5, 0, 1)
        return _towards_endpoint(rgb, brightness)

    def compare_hls(image):
        manual = manual_rgb_to_hls(image)
        official = rgb_to_hls(image, 'OpenCV')
        error = np.abs(manual - official)
        error[..., 0] = np.minimum(error[..., 0], 360 - error[..., 0])
        # Hue is undefined for achromatic pixels and must not inflate the metric.
        chromatic = (manual[..., 2] > 1e-7) & (official[..., 2] > 1e-7)
        error[..., 0] = np.where(chromatic, error[..., 0], 0)
        metrics = {}
        for index, name in enumerate(('H', 'L', 'S')):
            values = error[..., index][chromatic] if name == 'H' else error[..., index].ravel()
            metrics[name] = {'mean': float(values.mean()) if values.size else 0.0,
                             'max': float(values.max()) if values.size else 0.0}
        scaled_error = error.copy()
        scaled_error[..., 0] /= 180  # Normalize hue distance to the same [0,1] scale.
        peak = float(scaled_error.max())
        heatmap = to_uint8(scaled_error / peak) if peak else np.zeros_like(image, dtype=np.uint8)
        return {'manual': manual, 'opencv': official, 'metrics': metrics, 'heatmap': heatmap}

    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)

    if comparison:
        result = compare_hls(input_image)
        manual = to_uint8(manual_hls_to_rgb(result['manual']))
        official = to_uint8(hls_to_rgb(result['opencv'], 'OpenCV'))
        lines = ["### 手动 HLS 与 OpenCV COLOR_BGR2HLS 对比",
                 "RGB 输入已转换为 BGR，并归一化到 [0,1] 后使用 float32 官方接口。",
                 "| 通道 | 平均绝对误差 | 最大绝对误差 |",
                 "| --- | ---: | ---: |"]
        for name in ('H', 'L', 'S'):
            metric = result['metrics'][name]
            unit = '（度）' if name == 'H' else '（0~1）'
            lines.append(f"| {name}{unit} | {metric['mean']:.8g} | {metric['max']:.8g} |")
        lines.append("H 使用环形距离，忽略无彩色像素。误差图按当前最大误差增强显示，"
                     "R/G/B 分别表示 H/L/S 误差，不是实际视觉色差。")
        return manual, official, result['heatmap'], '\n'.join(lines)

    if mode == "色相/饱和度":
        hls = rgb_to_hls(input_image, hls_method)
        hls = adjust_hls(hls, h_slider, s_slider, l_slider)
        rgb = hls_to_rgb(hls, hls_method)
        output_image = to_uint8(apply_vibrance(rgb, vibrance, protect_skin))
    elif mode == "亮度/对比度":
        output_image = to_uint8(adjust_brightness_contrast(input_image, brightness, contrast))
    else:
        raise ValueError("未知的调整模式")

    # Channels describe the final quantized image, in the original order.
    output_hls = rgb_to_hls(output_image, hls_method)
    r_channel, g_channel, b_channel = [output_image[..., index].copy() for index in range(3)]
    h_channel = to_uint8(output_hls[..., 0] / 360)
    s_channel = to_uint8(output_hls[..., 2])
    l_channel = to_uint8(output_hls[..., 1])
    return output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel


def function_hw2(seed=42, action='generate', count=64, positive='', negative='',
                 attribute='浅发色', pool=None, vector_path=None, strength=1, pool_token=None):
    """作业二：附件 DCGAN 推理、候选采样、均值差向量保存/载入和编辑。

    所有模型与处理逻辑均在此函数内；不训练，不依赖额外的代码模块。
    候选噪声由 UI 的 gr.State 保存，模型缓存仅用于共享只读推理参数。
    """
    import hashlib
    import json
    from pathlib import Path
    import re
    import tempfile
    import threading
    from uuid import uuid4

    def integer(value, name, low, high):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name}必须是 {low}～{high} 的整数。')
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f'{name}必须是 {low}～{high} 的整数。') from None
        if not np.isfinite(number) or not number.is_integer() or not low <= number <= high:
            raise ValueError(f'{name}必须是 {low}～{high} 的整数。')
        return int(number)

    def indices(text, size):
        values = [v for v in re.split(r'[,，、;；\s]+', str(text or '').strip()) if v]
        if not values or any(not re.fullmatch(r'\d+', v) for v in values):
            raise ValueError('每组至少填写一个整数编号，以逗号或空格分隔。')
        values = [int(v) for v in values]
        if len(values) != len(set(values)) or any(v >= size for v in values):
            raise ValueError(f'组内编号不能重复，且必须在 0～{size - 1} 内。')
        return values

    def grouped_noise(data):
        size = integer(data['count'], '候选数量', 64, 256)
        integer(data['seed'], '候选 seed', 0, 2**32 - 1)
        p = indices(' '.join(map(str, data['positive_indices'])), size)
        n = indices(' '.join(map(str, data['negative_indices'])), size)
        if set(p) & set(n):
            raise ValueError('+A 和 -A 的编号不能重叠。')
        pos = np.asarray(data['positive_noise'], dtype=np.float32)
        neg = np.asarray(data['negative_noise'], dtype=np.float32)
        if pos.shape != (len(p), 100) or neg.shape != (len(n), 100):
            raise ValueError('分组噪声维度不正确。')
        if not np.isfinite(pos).all() or not np.isfinite(neg).all():
            raise ValueError('噪声必须为有限数值。')
        return pos.mean(axis=0) - neg.mean(axis=0)

    def model_hash():
        if not hasattr(function_hw2, '_checkpoint_hash'):
            function_hw2._checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return function_hw2._checkpoint_hash

    def read_vector(path):
        if not path:
            raise ValueError('请先保存或载入编辑向量。')
        # JSON has no executable objects or array headers. Bound input size.
        with open(path, 'rb') as file:
            content = file.read(2 * 1024**2 + 1)
        if len(content) > 2 * 1024**2:
            raise ValueError('向量文件过大。')
        data = json.loads(content)
        if not isinstance(data, dict) or data['format_version'] != 1 or data['nz'] != 100:
            raise ValueError('不支持的向量文件格式。')
        if data['checkpoint_sha256'] != model_hash():
            raise ValueError('向量对应的模型与当前附件模型不同。')
        if not isinstance(data['attribute'], str) or not 1 <= len(data['attribute']) <= 60:
            raise ValueError('属性名称无效。')
        vector = np.asarray(data['vector'], dtype=np.float32)
        if vector.shape != (100,) or not np.isfinite(vector).all():
            raise ValueError('编辑向量必须包含 100 个有限数值。')
        if not np.array_equal(vector, grouped_noise(data)):
            raise ValueError('编辑向量与分组噪声的均值差不符。')
        return vector.reshape(1, 100, 1, 1), data

    try:
        checkpoint = (Path(__file__).resolve().parent / '5 图像生成模型上 DCGAN代码示例'
                      / 'checkpoint' / 'dcgan_checkpoint.pth')
        if action == 'save_vector':
            if not isinstance(pool, dict) or not pool_token or pool.get('token') != pool_token:
                raise ValueError('候选图已更新或尚未生成，请按当前图像重新选择编号。')
            noise = np.asarray(pool['noise'], dtype=np.float32)
            if noise.ndim != 4 or noise.shape[1:] != (100, 1, 1) or not 64 <= len(noise) <= 256:
                raise ValueError('请先生成至少 64 张候选图。')
            p, n = indices(positive, len(noise)), indices(negative, len(noise))
            attribute = str(attribute or '').strip()
            if not 1 <= len(attribute) <= 60:
                raise ValueError('属性名称需要 1～60 个字符。')
            data = {'format_version': 1, 'nz': 100, 'attribute': attribute,
                    'seed': pool['seed'], 'count': len(noise),
                    'positive_indices': p, 'negative_indices': n,
                    'positive_noise': noise[p].reshape(len(p), 100).tolist(),
                    'negative_noise': noise[n].reshape(len(n), 100).tolist(),
                    'checkpoint_sha256': model_hash()}
            vector = grouped_noise(data)
            data['vector'] = vector.tolist()
            # Create only on explicit save; Gradio offers the file for download.
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix='dcgan_vector_',
                                              suffix='.json', delete=False) as file:
                json.dump(data, file, ensure_ascii=False, allow_nan=False, indent=2)
                path = file.name
            return path, f'向量已保存：+A {len(p)} 张，-A {len(n)} 张；vA 为两组原始噪声的均值差。'
        if action == 'load_vector':
            _, data = read_vector(vector_path)
            return str(vector_path), f'已载入属性「{data["attribute"]}」，候选 seed={data["seed"]}。'
        if action not in ('generate', 'candidates', 'edit'):
            raise ValueError('未知的作业二操作。')

        seed = integer(seed, 'seed', 0, 2**32 - 1)
        size = integer(count, '候选数量', 64, 256) if action == 'candidates' else 1
        vector = None
        if action == 'edit':
            vector, data = read_vector(vector_path)
            strength = float(strength)
            if not np.isfinite(strength) or not -3 <= strength <= 3:
                raise ValueError('编辑强度必须是 -3～3 的有限数。')

        import torch
        from torch import nn

        class Generator(nn.Module):
            # Same layer names and dimensions as cell 22 of the supplied tutorial.
            def __init__(self):
                super().__init__()
                self.main = nn.Sequential(
                    nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(512), nn.ReLU(True),
                    nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(256), nn.ReLU(True),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(128), nn.ReLU(True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(64), nn.ReLU(True),
                    nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False), nn.Tanh())

            def forward(self, noise):
                return self.main(noise)

        lock = function_hw2.__dict__.setdefault('_model_lock', threading.Lock())
        with lock:
            if not hasattr(function_hw2, '_model'):
                torch.set_num_threads(min(4, torch.get_num_threads()))
                with torch.random.fork_rng(devices=[]):
                    model = Generator()
                model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
                function_hw2._model = model.eval().requires_grad_(False)

        def render(noise):
            images = []
            with torch.inference_mode():
                for start in range(0, len(noise), 32):
                    output = function_hw2._model(torch.from_numpy(np.ascontiguousarray(noise[start:start + 32])))
                    output = ((output.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
                    images.extend(output.permute(0, 2, 3, 1).numpy())
            return images

        rng = torch.Generator(device='cpu').manual_seed(seed)
        # Pad single-image sampling to avoid CPU normal-sampler tail differences;
        # the same seed then also reproduces the first candidate's latent.
        noise = torch.randn(max(size, 4), 100, 1, 1, generator=rng).numpy()[:size].copy()
        images = render(noise)
        if action == 'generate':
            return images[0]
        if action == 'candidates':
            pool = {'seed': seed, 'noise': noise, 'token': uuid4().hex}
            gallery = [(image, f'编号 {i}') for i, image in enumerate(images)]
            return gallery, pool, f'已生成 {size} 张候选图，seed={seed}，编号 0～{size - 1}。'
        edited = render(noise + strength * vector)[0]
        return images[0], edited, (f'属性：{data["attribute"]}；seed={seed}，α={strength:g}。'
                                   '原图 G(z)，编辑图 G(z + αvA)；α=1 对应作业要求。')
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError,
            AttributeError, OverflowError, UnicodeError) as error:
        raise gr.Error(str(error)) from error

def function_hw3(bg_image, fg_image, mode, points=None, implementation='手动实现',
                 video_path=None, action='image', return_details=False):
    """四点单应、逆向采样及视频追踪；所有 HW3 算法限定在本函数内。

    points 按展示图的左上、右上、右下、左下顺序给出对应目标点。
    默认返回 RGB 图；return_details=True 时另返回 H 和说明。
    action='preview_video' 返回首帧；'video' 返回视频路径、轨迹 JSON、说明。
    兼容原 ImageEditor 三参数调用（自动排列四个标记的中心）。
    """
    import cv2
    import json
    import pathlib
    import shutil
    import subprocess
    import tempfile

    def image_array(value, name):
        if value is None:
            raise ValueError(f'请先输入{name}。')
        array = np.asarray(value)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if (array.ndim != 3 or array.shape[2] not in (3, 4)
                or min(array.shape[:2]) < 2):
            raise ValueError(f'{name}必须为宽高至少 2 像素的 RGB / RGBA 图像。')
        if array.dtype != np.uint8:
            if (not np.issubdtype(array.dtype, np.floating)
                    or not np.isfinite(array).all()
                    or np.any((array < 0) | (array > 1))):
                raise ValueError(f'{name}像素必须为 uint8 或 [0,1] 浮点数。')
            array = np.rint(array * 255).astype(np.uint8)
        return array

    def rgb_background(value):
        array = image_array(value, '底图')
        if array.shape[2] == 4:
            alpha = array[..., 3:4].astype(np.float64) / 255
            array = np.rint(array[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        return array

    def validate_quad(value, shape, inside=True):
        quad = np.asarray(value, dtype=np.float64)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            raise ValueError('请填写四个有限坐标 (x,y)，顺序为左上、右上、右下、左下。')
        edges = np.roll(quad, -1, axis=0) - quad
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        if not (np.all(cross > 1e-6) or np.all(cross < -1e-6)):
            raise ValueError('四点必须沿屏幕边缘依次组成凸四边形，不能重复、共线或交叉。')
        area = abs(np.sum(quad[:, 0] * np.roll(quad[:, 1], -1)
                          - quad[:, 1] * np.roll(quad[:, 0], -1))) / 2
        if area < 1:
            raise ValueError('四边形面积过小，请重新选择屏幕内缘。')
        h, w = shape[:2]
        if inside and (np.any(quad < 0) or np.any(quad[:, 0] > w - 1)
                       or np.any(quad[:, 1] > h - 1)):
            raise ValueError(f'角点超出图像范围：x 为 0～{w-1}，y 为 0～{h-1}。')
        return quad

    def project(matrix, xy):
        homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ matrix.T
        denominator = homogeneous[:, 2:3]
        if not np.isfinite(homogeneous).all() or np.any(abs(denominator) < 1e-12):
            raise ValueError('透视映射经过无穷远，请重新选点。')
        return homogeneous[:, :2] / denominator

    def homography(source, destination):
        # 坐标归一化提高数值稳定性，仍手工列出并求解八元一次方程。
        def normalize(xy):
            center = xy.mean(axis=0)
            scale = np.sqrt(2) / np.sqrt(np.mean(np.sum((xy - center) ** 2, axis=1)))
            transform = np.array([[scale, 0, -scale * center[0]],
                                  [0, scale, -scale * center[1]], [0, 0, 1.]])
            return (xy - center) * scale, transform

        src, ts = normalize(source)
        dst, td = normalize(destination)
        equations, values = [], []
        for (x, y), (u, v) in zip(src, dst):
            # u=(h11*x+h12*y+h13)/(h31*x+h32*y+1)，v 同理。
            equations.extend(([x, y, 1, 0, 0, 0, -u*x, -u*y],
                              [0, 0, 0, x, y, 1, -v*x, -v*y]))
            values.extend((u, v))
        coefficients = np.linalg.solve(np.asarray(equations), np.asarray(values))
        normalized = np.append(coefficients, 1).reshape(3, 3)
        matrix = np.linalg.solve(td, normalized @ ts)
        matrix /= matrix[2, 2]
        if np.max(np.linalg.norm(project(matrix, source) - destination, axis=1)) > 1e-5:
            raise ValueError('四点方程数值不稳定，请重新选择角点。')
        return matrix

    def cubic_weight(distance):
        # Keys 三次卷积核 a=-0.75，与 OpenCV INTER_CUBIC 的核参数一致。
        t = abs(distance)
        a = -0.75
        return np.where(t <= 1, (a + 2)*t**3 - (a + 3)*t**2 + 1,
                        np.where(t < 2, a*t**3 - 5*a*t**2 + 8*a*t - 4*a, 0))

    def sample(source, x, y):
        sh, sw = source.shape[:2]
        # BORDER_REPLICATE：邻域索引截断到边缘，保留小数位置计算权重。
        x = np.clip(x, 0, sw - 1)
        y = np.clip(y, 0, sh - 1)
        if mode == 'Nearest':
            return source[np.floor(y + .5).astype(int), np.floor(x + .5).astype(int)]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        dx, dy = (x - x0)[:, None], (y - y0)[:, None]
        if mode == 'Bilinear':
            x1, y1 = np.minimum(x0 + 1, sw - 1), np.minimum(y0 + 1, sh - 1)
            return ((1-dx)*(1-dy)*source[y0, x0] + dx*(1-dy)*source[y0, x1]
                    + (1-dx)*dy*source[y1, x0] + dx*dy*source[y1, x1])
        result = np.zeros((len(x), source.shape[2]), dtype=np.float64)
        for j in (-1, 0, 1, 2):
            wy = cubic_weight(y - (y0 + j))[:, None]
            for i in (-1, 0, 1, 2):
                weight = wy * cubic_weight(x - (x0 + i))[:, None]
                result += weight * source[np.clip(y0+j, 0, sh-1), np.clip(x0+i, 0, sw-1)]
        return result

    def composite(background, source, quad):
        sh, sw = source.shape[:2]
        source_corners = np.array([[0, 0], [sw-1, 0], [sw-1, sh-1], [0, sh-1]], float)
        matrix = homography(source_corners, quad)
        inverse = np.linalg.inv(matrix)
        output = background.copy()
        bh, bw = background.shape[:2]
        left, top = np.maximum(np.floor(quad.min(axis=0)).astype(int), 0)
        right, bottom = np.minimum(np.ceil(quad.max(axis=0)).astype(int), [bw-1, bh-1])
        if right < left or bottom < top:
            raise ValueError('屏幕已离开画面，无法继续替换。')
        # 预乘 alpha 后插值，透明像素不会产生黑色边缘。
        alpha = source[..., 3:4] / 255. if source.shape[2] == 4 else np.ones((sh, sw, 1))
        rgba = np.concatenate((source[..., :3] * alpha, alpha), axis=2)
        warped = None
        if implementation == 'OpenCV':
            translate = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1.]])
            warped = cv2.warpPerspective(
                rgba, translate @ matrix, (right-left+1, bottom-top+1),
                flags={'Nearest': cv2.INTER_NEAREST, 'Bilinear': cv2.INTER_LINEAR,
                       'Bicubic': cv2.INTER_CUBIC}[mode], borderMode=cv2.BORDER_REPLICATE)
        # 只遍历四边形包围盒，分块限制手写采样的临时内存。
        signed_area = np.sum(quad[:, 0]*np.roll(quad[:, 1], -1)
                             - quad[:, 1]*np.roll(quad[:, 0], -1))
        for start in range(top, bottom+1, 96):
            stop = min(start+96, bottom+1)
            yy, xx = np.mgrid[start:stop, left:right+1]
            inside = np.ones(xx.shape, bool)
            for p, q in zip(quad, np.roll(quad, -1, axis=0)):
                cross = (q[0]-p[0])*(yy-p[1]) - (q[1]-p[1])*(xx-p[0])
                inside &= cross * np.sign(signed_area) >= -1e-7
            if not inside.any():
                continue
            if warped is None:
                # 手工 H^-1 [x,y,1]^T 及齐次除法，实现目标→源的逐点逆映射。
                xy = project(inverse, np.column_stack((xx[inside], yy[inside])))
                pixels = sample(rgba, xy[:, 0], xy[:, 1])
            else:
                pixels = warped[start-top:stop-top][inside]
            opacity = np.clip(pixels[:, 3:4], 0, 1)
            colors = np.clip(pixels[:, :3], 0, 255*opacity)
            target = output[start:stop, left:right+1]
            target[inside] = np.rint(colors + target[inside]*(1-opacity)).clip(0, 255).astype(np.uint8)
        return output, matrix

    def editor_points(editor):
        layers = editor.get('layers') or []
        if not layers:
            raise ValueError('请提供四个角点坐标，或在编辑器中画四个独立标记。')
        layer = np.asarray(layers[0])
        mask = (layer[..., 0] > 127).astype(np.uint8)
        if layer.shape[2] == 4:
            mask &= (layer[..., 3] > 127).astype(np.uint8)
        count, _, _, centers = cv2.connectedComponentsWithStats(mask)
        if count != 5:
            raise ValueError('请只画四个互不相连的角点标记。')
        quad = centers[1:]
        angles = np.arctan2(quad[:, 1]-quad[:, 1].mean(), quad[:, 0]-quad[:, 0].mean())
        quad = quad[np.argsort(angles)]
        return np.roll(quad, -np.argmin(quad.sum(axis=1)), axis=0)

    def track_video(source, initial, capture, first, fps):
        # 在原始帧的屏幕平面上追踪，RANSAC 剔除动画引起的不一致运动。
        height, width = first.shape[:2]
        gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        previous = gray0
        quad = initial.copy()
        accumulated = np.eye(3)
        orb = cv2.ORB_create(nfeatures=1800, edgeThreshold=10, fastThreshold=8)

        def plane_mask(q, padding=4):
            mask = np.zeros((height, width), np.uint8)
            cv2.fillConvexPoly(mask, np.rint(q).astype(np.int32), 255)
            # 参考特征及 LK 窗口留在平面内部，防止描述子/邻域跨入背景。
            kernel = np.ones((2*abs(padding)+1, 2*abs(padding)+1), np.uint8)
            return cv2.erode(mask, kernel) if padding < 0 else cv2.dilate(mask, kernel)

        key0, desc0 = orb.detectAndCompute(gray0, plane_mask(initial, -16))
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        def estimate(a, b):
            if len(a) < 8:
                return None, 0
            h, inliers = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
            n = int(inliers.sum()) if inliers is not None else 0
            if h is None or n < 8 or n / len(a) < .5:
                return None, n
            return h, n

        def acceptable(candidate, last):
            try:
                validate_quad(candidate, first.shape, inside=False)
                old_area = abs(cv2.contourArea(last.astype(np.float32)))
                new_area = abs(cv2.contourArea(candidate.astype(np.float32)))
                return (.65 < new_area/old_area < 1.5
                        and np.max(np.linalg.norm(candidate-last, axis=1)) < 80
                        and np.all(candidate > -40)
                        and np.all(candidate < np.array([width+40, height+40])))
            except ValueError:
                return False

        folder = pathlib.Path(tempfile.mkdtemp(prefix='hw3_video_'))
        raw_path = folder / 'screen_raw.mp4'
        writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('无法创建 MP4 视频，请检查 OpenCV 视频编码器。')
        records = []
        frame = first
        try:
            index = 0
            while True:
                inliers, method = 0, 'manual_initialization'
                if index:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    features = cv2.goodFeaturesToTrack(previous, maxCorners=500, qualityLevel=.008,
                                                      minDistance=5, mask=plane_mask(quad, -12), blockSize=5)
                    motion = None
                    if features is not None and len(features) >= 8:
                        lk = dict(winSize=(25, 25), maxLevel=3,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                        forward, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, features, None, **lk)
                        if forward is not None:
                            backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None, **lk)
                            if backward is not None:
                                good = ((status.ravel() == 1) & (back_status.ravel() == 1)
                                        & (np.linalg.norm(backward-features, axis=2).ravel() < 1.0))
                                motion, inliers = estimate(features[good, 0], forward[good, 0])
                    candidate = None
                    proposed = accumulated
                    if motion is not None:
                        proposed = motion @ accumulated
                        candidate = project(proposed, initial)
                        method = 'LK_forward_backward_RANSAC'
                    # 定期对齐首帧特征以消除累积漂移；光流失效时也尝试重定位。
                    if index % 10 == 0 or candidate is None or not acceptable(candidate, quad):
                        key, desc = orb.detectAndCompute(gray, plane_mask(quad, 12))
                        if desc0 is not None and desc is not None and len(desc) >= 2:
                            pairs = matcher.knnMatch(desc0, desc, k=2)
                            matches = [m for pair in pairs if len(pair) == 2
                                       for m, n in [pair] if m.distance < .7*n.distance]
                            if len(matches) >= 8:
                                a = np.float32([key0[m.queryIdx].pt for m in matches])
                                b = np.float32([key[m.trainIdx].pt for m in matches])
                                anchor, count = estimate(a, b)
                                if anchor is not None:
                                    anchored = project(anchor, initial)
                                    consistent = (candidate is None or not acceptable(candidate, quad)
                                                  or np.max(np.linalg.norm(anchored-candidate, axis=1)) < 6)
                                    if acceptable(anchored, quad) and consistent:
                                        candidate, proposed, inliers = anchored, anchor, count
                                        method = 'ORB_first_frame_RANSAC'
                    if candidate is None or not acceptable(candidate, quad):
                        raise ValueError(f'第 {index+1} 帧屏幕追踪丢失，请选用屏幕内部有纹理、清晰且无遮挡的视频。'
                                         '为避免错误贴图，已停止处理。')
                    quad, accumulated, previous = candidate, proposed, gray
                result, h = composite(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), source, quad)
                writer.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                records.append(dict(frame=index, time_seconds=index/fps, points=quad.tolist(),
                                    H=h.tolist(), method=method, inliers=inliers))
                index += 1
                ok, frame = capture.read()
                if not ok:
                    break
        finally:
            writer.release()
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected > 0 and len(records) != expected:
            raise ValueError(f'视频解码不完整：读取 {len(records)} 帧，文件声明 {expected} 帧。')
        # H.264 + yuv420p 可在浏览器播放；无 ffmpeg 时由 Gradio 转码。
        output_path = raw_path
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            encoded = folder / 'screen_replaced.mp4'
            converted = subprocess.run(
                [ffmpeg, '-y', '-loglevel', 'error', '-i', str(raw_path), '-an', '-c:v',
                 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-vf',
                 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', str(encoded)],
                capture_output=True, timeout=300, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if converted.returncode == 0:
                output_path = encoded
        report_path = folder / 'tracking.json'
        report = dict(fps=fps, frame_count=len(records), width=width, height=height,
                      interpolation=mode, implementation=implementation,
                      coordinate_order='top-left,top-right,bottom-right,bottom-left',
                      audio='not copied', frames=records)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        status = (f'完成 {len(records)} 帧，{fps:g} fps，{len(records)/fps:.2f} 秒。'
                  '在原始屏幕平面上使用双向 LK 光流 + RANSAC，定期用首帧 ORB 重定位；'
                  '下载 JSON 可查看逐帧角点与矩阵。输出视频不含音轨。')
        return str(output_path), str(report_path), status

    capture = None
    try:
        if action not in ('image', 'video', 'preview_video'):
            raise ValueError('未知的作业三操作。')
        if action in ('video', 'preview_video'):
            if not video_path:
                raise ValueError('请先上传底图视频。')
            capture = cv2.VideoCapture(str(video_path))
            ok, first = capture.read()
            if not ok:
                raise ValueError('无法读取视频首帧，请检查文件。')
            if action == 'preview_video':
                return cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
            bg_image = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        if mode not in ('Nearest', 'Bilinear', 'Bicubic'):
            raise ValueError('插值方法必须为 Nearest、Bilinear 或 Bicubic。')
        if implementation not in ('手动实现', 'OpenCV'):
            raise ValueError('请选择手动实现或 OpenCV。')
        if isinstance(bg_image, dict):
            if points is None:
                points = editor_points(bg_image)
            bg_image = bg_image.get('background')
        background = rgb_background(bg_image)
        foreground = image_array(fg_image, '展示图')
        quad = validate_quad(points, background.shape)
        if action == 'video':
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('无法读取视频帧率。')
            return track_video(foreground, quad, capture, first, fps)
        output, matrix = composite(background, foreground, quad)
        description = (f'{implementation} / {mode}；H 将展示图映射到底图。'
                       '手动列出八元方程求 H，手写路径用 H⁻¹ 逆映射并按像素插值。'
                       '双三次使用 a=-0.75 的 4×4 邻域，边界复制；四边形外保持底图。')
        return (output, matrix.tolist(), description) if return_details else output
    except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error, OSError,
            subprocess.TimeoutExpired) as error:
        raise gr.Error(str(error)) from error
    finally:
        if capture is not None:
            capture.release()

def function_hw4(input_image, mode='锐化对比', radius=4, epsilon=0.01, amount=1.5,
                 sigma_space=2.0, sigma_range=0.10, noise_sigma=25.0, add_noise=False,
                 seed=42, focus_x=50, focus_y=50, crop_size=96, *, return_details=False):
    from time import perf_counter

    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)

    image = np.asarray(input_image)
    grayscale = image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1)
    if not grayscale and (image.ndim != 3 or image.shape[2] not in (3, 4)):
        raise gr.Error('请输入灰度、RGB 或 RGBA 图像。')
    if not image.size:
        raise gr.Error('输入图像不能为空。')

    pixels = image[..., None] if image.ndim == 2 else image
    pixels = pixels.astype(np.float64)
    if not np.isfinite(pixels).all():
        raise gr.Error('输入图像存在无效像素。')
    if image.dtype == np.bool_ or (np.issubdtype(image.dtype, np.floating) and
                                    pixels.min() >= 0 and pixels.max() <= 1):
        pixels *= 255
    pixels = np.clip(pixels, 0, 255) / 255
    alpha = pixels[..., 3:] if pixels.shape[2] == 4 else None
    pixels = pixels[..., :3]
    height, width = pixels.shape[:2]

    def to_image(values):
        if alpha is not None:
            values = np.concatenate((values, alpha), axis=2)
        result = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
        return result[..., 0] if grayscale else result

    def direct_box_sum(values, radius_y, radius_x):
        # 沿两个轴分别逐项累加窗口，边界只统计图内像素。
        horizontal = np.zeros_like(values)
        for offset in range(-radius_x, radius_x + 1):
            start, end = max(0, -offset), min(width, width - offset)
            horizontal[:, start:end] += values[:, start + offset:end + offset]
        total = np.zeros_like(horizontal)
        for offset in range(-radius_y, radius_y + 1):
            start, end = max(0, -offset), min(height, height - offset)
            total[start:end] += horizontal[start + offset:end + offset]
        return total

    def integral_box_sum(values, radius_y, radius_x):
        # 零值外圈让边界窗口也能沿固定偏移读积分图的四个角。
        diameter_y, diameter_x = 2 * radius_y + 1, 2 * radius_x + 1
        integral = np.zeros((height + diameter_y, width + diameter_x, values.shape[2]),
                            dtype=values.dtype)
        integral[radius_y + 1:radius_y + 1 + height,
                 radius_x + 1:radius_x + 1 + width] = values
        np.cumsum(integral, axis=1, out=integral)
        for row in range(1, integral.shape[0]):
            integral[row] += integral[row - 1]
        strips = integral[diameter_y:] - integral[:-diameter_y]
        return strips[:, diameter_x:] - strips[:, :-diameter_x]

    def guided_filter(box_sum):
        radius_y, radius_x = min(radius, height - 1), min(radius, width - 1)
        rows = np.arange(height)
        cols = np.arange(width)
        row_count = np.minimum(rows + radius_y + 1, height) - np.maximum(rows - radius_y, 0)
        col_count = np.minimum(cols + radius_x + 1, width) - np.maximum(cols - radius_x, 0)
        count = (row_count[:, None] * col_count)[..., None]

        def mean(values):
            return box_sum(values, radius_y, radius_x) / count

        # 每个颜色通道自导向：此时局部协方差等于导向图的方差。
        average = mean(pixels)
        variance = np.maximum(mean(pixels ** 2) - average ** 2, 0)
        slope = variance / (variance + epsilon)
        intercept = average * (1 - slope)
        return mean(slope) * pixels + mean(intercept)

    def bilateral_filter(values, support):
        # 每个偏移的权重是空间高斯与颜色距离高斯的乘积。
        padded = np.pad(values, ((support, support), (support, support), (0, 0)),
                        mode='reflect')
        offsets = np.arange(-support, support + 1)
        spatial = np.exp(-offsets ** 2 / (2 * sigma_space ** 2))
        range_scale = -0.5 / sigma_range ** 2
        weighted_sum = np.zeros_like(values)
        weight_sum = np.zeros(values.shape[:2], dtype=values.dtype)
        for iy, weight_y in enumerate(spatial):
            for ix, weight_x in enumerate(spatial):
                neighbors = padded[iy:iy + height, ix:ix + width]
                delta = neighbors - values
                distance = np.einsum('...c,...c->...', delta, delta)
                weights = weight_y * weight_x * np.exp(range_scale * distance)
                weighted_sum += weights[..., None] * neighbors
                weight_sum += weights
        return weighted_sum / weight_sum[..., None]

    def bm3d(values, sigma):
        if sigma == 0:
            return values.copy(), values.copy()

        # 在正交的亮度/色差空间计算，共用亮度通道找到的相似块。
        colors = np.array([[1, 1, 1], [1, 0, -1], [1, -2, 1]], dtype=np.float32)
        colors /= np.linalg.norm(colors, axis=1, keepdims=True)
        noisy = (values * 255).astype(np.float32)
        if noisy.shape[2] == 3:
            noisy = noisy @ colors.T
        if height < 8 or width < 8:
            noisy = np.pad(noisy, ((0, max(0, 8 - height)), (0, max(0, 8 - width)),
                                   (0, 0)), mode='reflect')

        patch_size, search_radius, step = 8, 12, 4
        channels = noisy.shape[2]
        last_row = noisy.shape[0] - patch_size
        last_col = noisy.shape[1] - patch_size
        ref_rows = list(range(0, last_row, step)) + [last_row]
        ref_cols = list(range(0, last_col, step)) + [last_col]

        # 块内使用正交离散余弦变换，块之间使用正交 Hadamard 变换。
        n = np.arange(patch_size)
        dct = np.cos(np.pi / patch_size * n[:, None] * (n[None, :] + 0.5))
        dct *= np.sqrt(2 / patch_size)
        dct[0] /= np.sqrt(2)
        dct = dct.astype(np.float32)
        group_bases = {1: np.ones((1, 1), dtype=np.float32)}
        for size in (2, 4, 8, 16, 32):
            previous = group_bases[size // 2]
            group_bases[size] = np.block([[previous, previous],
                                          [previous, -previous]]) / np.float32(np.sqrt(2))

        taper = np.kaiser(patch_size, 2).astype(np.float32)
        window = (taper[:, None] * taper)[..., None]
        noisy_patches = np.lib.stride_tricks.sliding_window_view(
            noisy, (patch_size, patch_size), axis=(0, 1))

        def transform(patches, group_basis):
            spatial = dct @ patches @ dct.T
            return (group_basis @ spatial.reshape(len(patches), -1)).reshape(spatial.shape)

        def inverse(coefficients, group_basis):
            spatial = group_basis.T @ coefficients.reshape(len(coefficients), -1)
            return dct.T @ spatial.reshape(coefficients.shape) @ dct

        def estimate(basic=None):
            hard_stage = basic is None
            pilot_patches = noisy_patches if hard_stage else np.lib.stride_tricks.sliding_window_view(
                basic, (patch_size, patch_size), axis=(0, 1))
            matching = pilot_patches[:, :, 0]
            max_group = 16 if hard_stage else 32
            cutoff = max(2500, 3.5 * sigma ** 2) if hard_stage else max(400, 0.8 * sigma ** 2)
            sum_values = np.zeros_like(noisy)
            sum_weights = np.zeros_like(noisy)
            for row in ref_rows:
                top, bottom = max(0, row - search_radius), min(last_row + 1, row + search_radius + 1)
                for col in ref_cols:
                    left, right = max(0, col - search_radius), min(last_col + 1, col + search_radius + 1)
                    differences = matching[top:bottom, left:right] - matching[row, col]
                    distances = np.einsum('ijmn,ijmn->ij', differences, differences)
                    distances /= patch_size ** 2
                    # 固定将参考块排在首位，确保每个像素至少参与一次重叠聚合。
                    distances[row - top, col - left] = -1
                    eligible = np.flatnonzero(distances.ravel() <= cutoff)
                    count = 1 << (min(max_group, len(eligible)).bit_length() - 1)
                    scores = distances.ravel()[eligible]
                    if len(eligible) > count:
                        selected = np.argpartition(scores, count - 1)[:count]
                    else:
                        selected = np.arange(count)
                    indices = eligible[selected[np.argsort(scores[selected], kind='stable')]]
                    block_rows = indices // (right - left) + top
                    block_cols = indices % (right - left) + left
                    group_basis = group_bases[count]
                    coefficients = transform(noisy_patches[block_rows, block_cols], group_basis)
                    if hard_stage:
                        keep = np.abs(coefficients) >= 2.7 * sigma
                        coefficients *= keep
                        strength = np.count_nonzero(keep, axis=(0, 2, 3))
                    else:
                        pilot = transform(pilot_patches[block_rows, block_cols], group_basis)
                        power = pilot ** 2
                        gain = power / (power + sigma ** 2)
                        coefficients *= gain
                        strength = np.sum(gain ** 2, axis=(0, 2, 3))

                    weights = np.divide(1.0, strength, out=np.ones(channels, dtype=np.float32),
                                        where=strength > 1e-8)
                    weighted_window = window * weights
                    patches = inverse(coefficients, group_basis).transpose(0, 2, 3, 1)
                    for patch, block_row, block_col in zip(patches, block_rows, block_cols):
                        region = (slice(block_row, block_row + patch_size),
                                  slice(block_col, block_col + patch_size))
                        sum_values[region] += patch * weighted_window
                        sum_weights[region] += weighted_window
            return sum_values / sum_weights

        def restore(estimate):
            cropped = estimate[:height, :width]
            if channels == 3:
                cropped = cropped @ colors
            return np.clip(cropped / 255, 0, 1)

        basic = estimate()
        final = estimate(basic)
        return restore(basic), restore(final)

    def timed(operation, *args):
        started = perf_counter()
        result = operation(*args)
        return result, (perf_counter() - started) * 1000

    def psnr(reference, result):
        # 在实际显示的 8 位像素上评估，排除未参与处理的透明通道。
        if alpha is not None:
            reference, result = reference[..., :3], result[..., :3]
        mse = np.mean((reference.astype(np.float64) - result) ** 2)
        return '∞' if mse == 0 else f'{10 * np.log10(255 ** 2 / mse):.2f}'

    if mode == '锐化对比':
        radius, epsilon, amount = int(radius), float(epsilon), float(amount)
        if radius < 1 or not np.isfinite([epsilon, amount]).all() or epsilon <= 0 or amount < 0:
            raise gr.Error('导向滤波半径和正则参数须大于 0，锐化强度须大于或等于 0。')
    elif mode == 'BM3D 去噪':
        noise_sigma = float(noise_sigma)
        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise gr.Error('噪声标准差须大于或等于 0。')
        if add_noise and int(seed) < 0:
            raise gr.Error('随机种子须大于或等于 0。')
    else:
        raise gr.Error('请选择有效的处理模式。')

    if return_details:
        sigma_space, sigma_range = float(sigma_space), float(sigma_range)
        if not np.isfinite([sigma_space, sigma_range]).all() or min(sigma_space, sigma_range) <= 0:
            raise gr.Error('双边滤波的两个标准差都须大于 0。')
        # 支撑范围覆盖中心像素两倍空间标准差的邻域。
        bilateral_radius = int(np.ceil(2 * sigma_space))

    if mode == '锐化对比':
        def sharpen(base):
            return to_image(pixels + amount * (pixels - base))

        guided, guided_ms = timed(guided_filter, direct_box_sum)
        guided_image = sharpen(guided)
        if not return_details:
            return guided_image
        fast, fast_ms = timed(guided_filter, integral_box_sum)
        bilateral, bilateral_ms = timed(bilateral_filter, pixels, bilateral_radius)
        images = (to_image(pixels), guided_image, sharpen(bilateral), sharpen(fast))
        max_difference = np.max(np.abs(guided - fast)) * 255
        speedup = guided_ms / max(fast_ms, 1e-12)
        report = (
            '| 方法 | 滤波耗时 |\n|---|---:|\n'
            f'| 普通导向滤波 | {guided_ms:.2f} ms |\n'
            f'| 双边滤波 | {bilateral_ms:.2f} ms |\n'
            f'| 积分图快速导向滤波 | {fast_ms:.2f} ms |\n\n'
            f'普通与快速导向滤波输出的最大差值为 **{max_difference:.2e} 灰度级**'
            f'（量化前），本次速度比为 **{speedup:.2f}×**。'
            f'导向窗口为 {2 * radius + 1} × {2 * radius + 1}，'
            f'双边窗口为 {2 * bilateral_radius + 1} × {2 * bilateral_radius + 1}。'
        )
    else:
        noisy = pixels
        if add_noise:
            random = np.random.default_rng(int(seed))
            noisy = np.rint(np.clip(pixels * 255 + random.normal(0, noise_sigma, pixels.shape),
                                    0, 255)) / 255
        (basic, final), bm3d_ms = timed(bm3d, noisy, noise_sigma)
        if not return_details:
            return to_image(final)
        bilateral, bilateral_ms = timed(bilateral_filter, noisy, bilateral_radius)
        images = (to_image(noisy), to_image(bilateral), to_image(basic), to_image(final))
        report = (f'双边滤波耗时 **{bilateral_ms:.2f} ms**，'
                  f'BM3D 两阶段合计耗时 **{bm3d_ms / 1000:.3f} 秒**。')
        if add_noise:
            clean = to_image(pixels)
            report += (
                '\n\n| 图像 | 峰值信噪比（PSNR，dB） |\n|---|---:|\n'
                f'| 加噪图像 | {psnr(clean, images[0])} |\n'
                f'| 双边滤波 | {psnr(clean, images[1])} |\n'
                f'| BM3D 基础估计 | {psnr(clean, images[2])} |\n'
                f'| BM3D 最终结果 | {psnr(clean, images[3])} |'
            )
        else:
            report += ' 上传的图像直接作为待去噪输入。'

    crop_height = min(height, max(1, int(crop_size)))
    crop_width = min(width, max(1, int(crop_size)))
    center_x = round((width - 1) * float(focus_x) / 100)
    center_y = round((height - 1) * float(focus_y) / 100)
    left = max(0, min(width - crop_width, center_x - crop_width // 2))
    top = max(0, min(height - crop_height, center_y - crop_height // 2))
    crops = [result[top:top + crop_height, left:left + crop_width] for result in images]
    # 先按像素最近邻放大，方便在浏览器中看清同一区域的细节。
    scale = max(1, min(4, 256 // max(crop_height, crop_width)))
    crops = [np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1) for crop in crops]
    margin = 4
    tile_h, tile_w = crops[0].shape[:2]
    panel = np.full((2 * tile_h + margin, 2 * tile_w + margin) + crops[0].shape[2:],
                    255, dtype=np.uint8)
    for index, crop in enumerate(crops):
        y, x = divmod(index, 2)
        y, x = y * (tile_h + margin), x * (tile_w + margin)
        panel[y:y + tile_h, x:x + tile_w] = crop
    return (*images, panel, report)

def function_hw5(input_image, mode="直方图均衡化", reference_image=None):
    """手动均衡化或匹配灰度图，也可处理 RGB 图像的 HLS 明度。"""
    def get_lightness(image, label):
        image = np.asarray(image)
        if image.size == 0 or image.dtype != np.uint8:
            raise gr.Error(f"{label}必须是一张非空的 8 位图像。")
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            # HLS 明度是 RGB 最大和最小通道的平均值，取整到 0～255。
            channel_sum = image.max(axis=2).astype(np.uint16) + image.min(axis=2)
            return ((channel_sum + 1) // 2).astype(np.uint8)
        raise gr.Error(f"{label}必须是灰度或 RGB 图像。")

    if input_image is None:
        raise gr.Error("请先上传输入图像。")
    source = np.asarray(input_image)
    lightness = get_lightness(source, "输入图像")

    if mode == "直方图均衡化":
        counts = np.bincount(lightness.ravel(), minlength=256)
        cumulative = counts.cumsum()
        first_count = counts[counts > 0][0]
        if first_count == lightness.size:
            return source.copy()
        # 将最暗的已出现灰度映射到 0，最亮灰度映射到 255。
        lookup = np.rint((cumulative - first_count) * 255.0 / (lightness.size - first_count))
        enhanced = lookup.clip(0, 255).astype(np.uint8)[lightness]
    elif mode == "直方图匹配":
        if reference_image is None:
            raise gr.Error("请选择一张参考图像。")
        reference = get_lightness(reference_image, "参考图像")
        source_counts = np.bincount(lightness.ravel(), minlength=256)
        reference_counts = np.bincount(reference.ravel(), minlength=256)
        source_cdf = source_counts.cumsum() / lightness.size
        reference_levels = np.flatnonzero(reference_counts)
        reference_cdf = reference_counts[reference_levels].cumsum() / reference.size

        # 对每个输入灰度寻找累计比例最接近的、参考图中实际出现的灰度。
        right = np.searchsorted(reference_cdf, source_cdf)
        left = np.maximum(right - 1, 0)
        right = np.minimum(right, reference_cdf.size - 1)
        use_left = source_cdf - reference_cdf[left] <= reference_cdf[right] - source_cdf
        lookup = reference_levels[np.where(use_left, left, right)].astype(np.uint8)
        enhanced = lookup[lightness]
    else:
        raise gr.Error("请选择有效的处理方式。")

    if source.ndim == 2:
        return enhanced

    import cv2

    # HLS（色相、明度、饱和度）转换用于保留输入图像的色相和饱和度。
    hls = cv2.cvtColor(source.astype(np.float32) / 255.0, cv2.COLOR_RGB2HLS)
    changed = enhanced != lightness
    hls[..., 1] = np.where(changed, enhanced.astype(np.float32) / 255.0, hls[..., 1])
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return np.rint(rgb.clip(0, 1) * 255.0).astype(np.uint8)