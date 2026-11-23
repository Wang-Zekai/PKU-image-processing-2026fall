# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np

def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业一：图像调整与两项进阶功能")
        gr.Markdown("已实现进阶 **1：手动 HLS 与 OpenCV 对比**、**3：自然饱和度**。"
                    "自然饱和度参考 [SweetFX Vibrance 1.1.1](https://github.com/CeeJayDK/SweetFX/blob/master/Shaders/SweetFX/Vibrance.fx)。"
                    "极端负值限制为去色，避免反色；额外肤色保护是可选自定义扩展，默认关闭。"
                    "这是开源算法参考实现，并非 Photoshop 数值复现。")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy',
                                       format='png', label='输入图像')
                mode = gr.Radio(["色相/饱和度", "亮度/对比度"], label="模式", value="色相/饱和度")
                hls_method = gr.Radio(["手动实现", "OpenCV"], label="HLS 转换方法", value="手动实现")
                h_slider = gr.Slider(-1, 1, label="色相（-180°～180°）", value=0, step=0.01)
                s_slider = gr.Slider(-1, 1, label="饱和度", value=0, step=0.01)
                l_slider = gr.Slider(-1, 1, label="明度", value=0, step=0.01)
                vibrance = gr.Slider(-1, 1, label="自然饱和度（进阶3）", value=0, step=0.01)
                protect_skin = gr.Checkbox(label="额外肤色保护（自定义扩展）", value=False,
                                           info="SweetFX 原版不含此项；启用后按橙色相减弱调整。")
                brightness = gr.Slider(-1, 1, label="亮度", value=0, step=0.01, visible=False)
                contrast = gr.Slider(-1, 1, label="对比度", value=0, step=0.01, visible=False)
                reset_button = gr.Button("重置调整")
            with gr.Column():
                output_image = gr.Image(type='numpy', format='png', label='输出图像', interactive=False)
                with gr.Row():
                    r_channel = gr.Image(type='numpy', format='png', label='R通道', interactive=False)
                    g_channel = gr.Image(type='numpy', format='png', label='G通道', interactive=False)
                    b_channel = gr.Image(type='numpy', format='png', label='B通道', interactive=False)
                with gr.Row():
                    h_channel = gr.Image(type='numpy', format='png', label='H通道（灰度映射）', interactive=False)
                    s_channel = gr.Image(type='numpy', format='png', label='S通道', interactive=False)
                    l_channel = gr.Image(type='numpy', format='png', label='L通道', interactive=False)
                gr.Markdown("通道均来自调整后的图像；H 的 0～360° 映射为 0～255。"
                            "输出为无损 PNG，可下载后核对像素。")

        controls = [input_image, mode, h_slider, s_slider, l_slider, brightness, contrast,
                    vibrance, hls_method, protect_skin]
        outputs = [output_image, r_channel, g_channel, b_channel, h_channel, s_channel, l_channel]

        def render_result(*values):
            if values[0] is None:
                return (None,) * 7
            return process(*values)

        input_image.change(render_result, inputs=controls, outputs=outputs, api_name='adjust_hw1')

        def update_mode(selected_mode):
            color_mode = selected_mode == "色相/饱和度"
            return [gr.update(visible=color_mode)] * 5 + [gr.update(visible=not color_mode)] * 2

        mode.change(update_mode, inputs=[mode],
                    outputs=[h_slider, s_slider, l_slider, vibrance, protect_skin, brightness, contrast],
                    api_name='switch_hw1_mode').then(render_result, inputs=controls, outputs=outputs, api_name=False)
        for slider in (h_slider, s_slider, l_slider, vibrance, brightness, contrast):
            slider.release(render_result, inputs=controls, outputs=outputs, api_name=False)
        hls_method.change(render_result, inputs=controls, outputs=outputs, api_name=False)
        protect_skin.change(render_result, inputs=controls, outputs=outputs, api_name=False)
        reset_button.click(lambda: (0, 0, 0, 0, 0, 0, False),
                           outputs=[h_slider, s_slider, l_slider, brightness, contrast, vibrance, protect_skin],
                           api_name=False).then(render_result, inputs=controls, outputs=outputs, api_name=False)

        with gr.Accordion("进阶1：手动 HLS 与官方 COLOR_BGR2HLS 对比", open=True):
            gr.Markdown("点击下方按钮对比当前**输入图像**的两种转换。"
                        "float32 官方参考避免了 uint8 HLS 的量化误差；两侧显示转换再还原的图像。")
            compare_button = gr.Button("比较手动 HLS 与 OpenCV", variant='primary')
            with gr.Row():
                manual_image = gr.Image(type='numpy', format='png', label='手动转换后还原', interactive=False)
                official_image = gr.Image(type='numpy', format='png', label='OpenCV 转换后还原', interactive=False)
                error_image = gr.Image(type='numpy', format='png', label='H/L/S 误差（增强显示）', interactive=False)
            comparison_text = gr.Markdown("上传图像后点击比较，查看各通道平均及最大误差。")
            compare_outputs = [manual_image, official_image, error_image, comparison_text]

            def render_comparison(image):
                if image is None:
                    return None, None, None, "请先上传图像，再进行 HLS 对比。"
                return process(image, "色相/饱和度", 0, 0, 0, 0, 0, comparison=True)

            compare_button.click(
                render_comparison, inputs=[input_image], outputs=compare_outputs,
                concurrency_id='hw1_hls_comparison', concurrency_limit=1,
                api_name='compare_hls')
            # Clear after in-flight comparison output has finished postprocessing.
            input_image.change(lambda: (None, None, None, "输入图像已更新，请重新点击比较。"),
                               outputs=compare_outputs, concurrency_id='hw1_hls_comparison',
                               concurrency_limit=1, api_name=False)

    return demo

def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.HTML('<style>#hw2_generated img,#hw2_original img,#hw2_edited img'
                '{width:100%!important;height:100%!important;object-fit:contain!important}'
                '#hw2_gallery .grid-wrap{max-height:660px;overflow-y:auto!important}</style>')
        gr.Markdown('## 作业二：DCGAN 人脸生成与属性编辑')
        gr.Markdown('使用附件预训练权重，生成 **64×64 RGB** 人脸。无需训练，也无需上传输入图像。'
                    '相同 seed 和模型可复现相同结果；属性由观察候选图后人工分组。')
        pool_state = gr.State(None)
        pool_token = gr.Textbox(value='', visible=False)
        vector_state = gr.State(None)
        gr.Markdown('### 1. 输入 seed，生成图像')
        with gr.Row():
            with gr.Column():
                seed = gr.Number(value=42, label='生成 seed（0～4294967295 的整数）')
                generate_button = gr.Button('生成图像', variant='primary')
            with gr.Column():
                output_image = gr.Image(type='numpy', format='png', label='生成图像 G(z)',
                                        interactive=False, height=256, elem_id='hw2_generated')
        generate_button.click(lambda s: process(s), inputs=[seed], outputs=[output_image],
                              api_name='generate_hw2')

        gr.Markdown('### 2. 生成候选图，并选择相反属性')
        gr.Markdown('例如 +A 为浅发色，-A 为深发色。建议每组选择 5～10 张特征明确的图像。'
                    '编号从 **0** 开始；在候选区内滚动或拖动右侧滚动条查看更多。重新生成后请重新选择编号。')
        with gr.Row():
            pool_seed = gr.Number(value=2026, label='候选 seed')
            count = gr.Dropdown([64, 128, 256], value=128, label='候选数量')
            candidates_button = gr.Button('生成候选图（至少64张）')
        gallery = gr.Gallery(label='候选图与编号（点击可放大）', columns=8, rows=8,
                             height=680, object_fit='contain', format='png', interactive=False, elem_id='hw2_gallery')
        pool_status = gr.Markdown('尚未生成候选图。', elem_id='hw2_pool_status')
        with gr.Row():
            attribute = gr.Textbox(value='浅发色', label='属性 A 名称')
            positive = gr.Textbox(label='+A 图像编号', placeholder='例如：0, 3, 12')
            negative = gr.Textbox(label='-A 图像编号', placeholder='例如：1, 5, 20')
        with gr.Accordion('可复现的浅发色示例分组', open=False):
            gr.Markdown('先生成 **候选 seed=2026、数量=128**，再观察图像并填写：\n\n'
                        '- +A 浅发色：`6,7,26,29,35,44,99,115`\n'
                        '- -A 深发色：`1,9,11,50,59,61,83,103`\n\n'
                        )

        def candidates(s, n):
            images, pool, message = process(s, action='candidates', count=n)
            return images, pool, message, '', '', pool['token']

        candidates_button.click(candidates, inputs=[pool_seed, count],
                                 outputs=[gallery, pool_state, pool_status, positive, negative, pool_token],
                                 api_name='candidates_hw2', concurrency_id='hw2_pool', concurrency_limit=1)

        gr.Markdown('### 3. 计算并保存编辑向量')
        gr.Markdown('`vA = mean(z_+) − mean(z_−)`。点击保存后生成 JSON 文件供下载，'
                    '包含向量、所选噪声、编号、seed 和模型校验值，可重新载入。')
        save_button = gr.Button('计算并保存向量', variant='primary')
        vector_download = gr.File(label='下载本次编辑向量', interactive=False, elem_id='hw2_download')
        vector_upload = gr.File(label='载入以前保存的 JSON 向量', file_types=['.json'], type='filepath',
                                 elem_id='hw2_upload')
        vector_status = gr.Markdown('尚未载入向量，请先选择两组候选图并保存。', elem_id='hw2_vector_status')

        def save_vector(pool, p, n, name, token):
            path, message = process(action='save_vector', pool=pool, positive=p, negative=n,
                                    attribute=name, pool_token=token)
            return path, path, message

        save_button.click(save_vector, inputs=[pool_state, positive, negative, attribute, pool_token],
                          outputs=[vector_state, vector_download, vector_status], api_name='save_vector_hw2',
                          concurrency_id='hw2_pool', concurrency_limit=1)

        def load_vector(path):
            if path is None:
                return None, None, '尚未载入向量。'
            active, message = process(action='load_vector', vector_path=path)
            return active, active, message

        vector_upload.change(load_vector, inputs=[vector_upload],
                              outputs=[vector_state, vector_download, vector_status], api_name='load_vector_hw2')

        gr.Markdown('### 4. 用新 seed 对比原图与编辑图')
        with gr.Row():
            edit_seed = gr.Number(value=42, label='编辑 seed（可与候选 seed 不同）')
            strength = gr.Slider(-3, 3, value=1, step=0.1, label='编辑强度 α（作业要求为1）')
            edit_button = gr.Button('生成原图与编辑图', variant='primary')
        with gr.Row():
            original = gr.Image(type='numpy', format='png', label='原图 G(z)', interactive=False, height=300,
                                elem_id='hw2_original')
            edited = gr.Image(type='numpy', format='png', label='编辑图 G(z + αvA)', interactive=False, height=300,
                              elem_id='hw2_edited')
        edit_status = gr.Markdown('先保存或载入编辑向量，再输入任意新 seed。', elem_id='hw2_edit_status')
        edit_button.click(lambda s, path, a: process(s, action='edit', vector_path=path, strength=a),
                          inputs=[edit_seed, vector_state, strength], outputs=[original, edited, edit_status],
                          api_name='edit_hw2')
        gr.Markdown('潜空间均值差是近似的属性方向，可能同时影响姿态、背景和其他外观；'
                    '不同 seed 的效果可能不同，大强度可能导致失真。α=0 返回原图，α<0 反向编辑。')
    return demo

def create_demo_hw3(process):
    import cv2
    from PIL import Image

    empty_points = [[None, None] for _ in range(4)]
    materials = pathlib.Path(__file__).resolve().parent / '作业3测试图'
    presets = [
        ('底图1.jpg', '展示图1.jpg', [[258, 115], [858, 75], [905, 462], [302, 521]]),
        ('底图2.png', '展示图2.jpg', [[251, 95], [417, 102], [406, 241], [239, 220]]),
        ('底图3.jpg', '卡通1.jpg', [[610, 350], [867, 522], [625, 868], [367, 688]]),
        ('底图4.jpg', '卡通2.jpg', [[800, 143], [1047, 218], [819, 675], [579, 570]])]
    scene_names = ['笔记本 · 白色桌面', '笔记本 · 使用场景', '平板 · 粉色桌面', '手机 · 手持屏幕']
    content_names = ['花卉插画', '小猫举牌', '卡通人物', '卡通猫咪']
    scenes = {name: (str(materials/b), q) for name, (b, _, q) in zip(scene_names, presets)
              if (materials/b).is_file()}
    contents = {name: str(materials/f) for name, (_, f, _) in zip(content_names, presets)
                if (materials/f).is_file()}
    custom_scene, custom_content, custom_video = '上传自己的底图', '上传自己的图片', '上传自己的视频'
    default_scene = next(iter(scenes), custom_scene)
    default_content = next(iter(contents), custom_content)
    video_file = materials/'底图视频.mp4'
    video_name = '桌面显示器 · 镜头移动'
    video_vertices = [[337, 164], [674, 168], [680, 399], [349, 442]]

    def read_rgb(path):
        with Image.open(path) as image:
            return np.asarray(image.convert('RGB')).copy()

    initial_bg = read_rgb(scenes[default_scene][0]) if scenes else None
    initial_fg = read_rgb(contents[default_content]) if contents else None
    initial_points = scenes[default_scene][1] if scenes else empty_points
    initial_frame = (process(None, None, 'Nearest', video_path=str(video_file), action='preview_video')
                     if video_file.is_file() else None)
    # 素材更新与生成共用队列；进行中的旧任务完成后，再应用选择并清空旧效果。
    image_events = dict(concurrency_id='hw3_image', concurrency_limit=1)
    video_events = dict(concurrency_id='hw3_video', concurrency_limit=1)

    def mark_points(image, coordinates):
        if image is None:
            return None
        preview = image.copy()
        values = np.asarray(coordinates, dtype=float)
        valid = []
        if values.shape == (4, 2):
            for i, xy in enumerate(values):
                if np.isfinite(xy).all() and np.all(xy >= 0) and xy[0] < image.shape[1] and xy[1] < image.shape[0]:
                    point = tuple(np.rint(xy).astype(int))
                    valid.append(point)
                    cv2.circle(preview, point, 5, (255, 40, 40), -1)
                    cv2.putText(preview, str(i+1), (point[0]+7, point[1]-7),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 40, 40), 2)
        if len(valid) == 4:
            cv2.polylines(preview, [np.asarray(valid, np.int32)], True, (255, 40, 40), 2)
        return preview

    def click_point(image, coordinates, event: gr.SelectData):
        if image is None:
            raise gr.Error('请先上传底图或载入视频首帧。')
        values = np.asarray(coordinates, dtype=float).copy()
        if values.shape != (4, 2):
            values = np.full((4, 2), np.nan)
        available = np.flatnonzero(~np.isfinite(values).all(axis=1))
        if not len(available):
            raise gr.Error('已选择四点；可直接改坐标，或点击“清空角点”重新选择。')
        values[available[0]] = event.index[:2]
        return values, mark_points(image, values)

    def render_image(image, foreground, method, coordinates, backend):
        result, matrix, _ = process(image, foreground, method, points=coordinates,
                                    implementation=backend, return_details=True)
        reference_backend = 'OpenCV' if backend == '手动实现' else '手动实现'
        reference = process(image, foreground, method, points=coordinates,
                            implementation=reference_backend)
        return result, reference, matrix, '替换完成。可以下载结果，或更换上方素材继续尝试。'

    with gr.Blocks() as demo:
        gr.Markdown('## 作业三: 替换屏幕内容工具')
        gr.Markdown('选一个场景，再挑选要放进屏幕的图片。内置场景已选好屏幕范围，可以直接生成效果；也支持上传自己的素材。')
        with gr.Tabs():
            with gr.Tab('图片'):
                gr.Markdown('### 1. 选择底图和展示图')
                with gr.Row():
                    with gr.Column():
                        scene_choice = gr.Dropdown(list(scenes)+[custom_scene], value=default_scene, label='选择底图场景')
                        scene_gallery = gr.Gallery([(p, name) for name, (p, _) in scenes.items()],
                                                   label='点击选择场景', columns=4, rows=1, height=145,
                                                   object_fit='contain', allow_preview=False, interactive=False)
                        bg_img = gr.Image(value=initial_bg, sources=['upload', 'clipboard'], type='numpy',
                                          image_mode='RGB', label='当前底图（也可上传，点击画面可重新选点）',
                                          format='png', height=320)
                    with gr.Column():
                        content_choice = gr.Dropdown(list(contents)+[custom_content], value=default_content, label='选择展示内容')
                        content_gallery = gr.Gallery([(p, name) for name, p in contents.items()],
                                                     label='点击选择展示图', columns=4, rows=1, height=145,
                                                     object_fit='contain', allow_preview=False, interactive=False)
                        fg_img = gr.Image(value=initial_fg, sources=['upload', 'clipboard'], type='numpy',
                                          image_mode='RGB', label='当前展示图（也可上传）', format='png', height=320)
                gr.Markdown('### 2. 选择屏幕四角并设置处理方式')
                gr.Markdown('内置场景可直接保留当前范围。上传底图后，按 **左上 → 右上 → 右下 → 左下** 点击屏幕内缘；也可填写原图像素坐标。')
                with gr.Row():
                    with gr.Column(scale=1):
                        coordinates = gr.Dataframe(value=initial_points, headers=['x', 'y'],
                                                   datatype=['number', 'number'], type='numpy',
                                                   row_count=(4, 'fixed'), col_count=(2, 'fixed'),
                                                   label='四个目标顶点（依次为①②③④）', interactive=True)
                        reset_btn = gr.Button('清空角点')
                    with gr.Column(scale=2):
                        preview = gr.Image(value=mark_points(initial_bg, initial_points), type='numpy', format='png', label='屏幕范围预览',
                                           interactive=False, height=260)
                    with gr.Column():
                        mode = gr.Radio([('快速 · 最近邻', 'Nearest'), ('平滑 · 双线性', 'Bilinear'),
                                         ('细腻 · 双三次', 'Bicubic')], value='Bilinear', label='画面采样')
                        with gr.Accordion('更多设置', open=False):
                            backend = gr.Radio([('精细处理', '手动实现'), ('快速处理', 'OpenCV')],
                                               value='手动实现', label='处理方式')
                run_btn = gr.Button('生成替换效果', variant='primary')
                gr.Markdown('### 3. 查看替换效果')
                with gr.Row():
                    output_image = gr.Image(type='numpy', format='png', label='替换效果',
                                            interactive=False, height=420)
                    reference = gr.Image(type='numpy', format='png', label='对照效果',
                                         interactive=False, height=420)
                status = gr.Markdown('已准备好默认素材。选择喜欢的搭配，再点击“生成替换效果”。')
                with gr.Accordion('查看变换信息', open=False):
                    matrix = gr.Dataframe(headers=['h1', 'h2', 'h3'], datatype='number',
                                          label='手工求解的 3×3 H（展示图 → 底图）', interactive=False)
                def choose_scene(name):
                    image, vertices = (read_rgb(scenes[name][0]), scenes[name][1]) if name in scenes else (None, empty_points)
                    return (image, vertices, mark_points(image, vertices), None, None, None,
                            '屏幕范围已就绪，可直接生成效果。' if image is not None else '请上传底图，然后选择屏幕四角。')

                def choose_content(name):
                    return ((read_rgb(contents[name]) if name in contents else None), None, None, None,
                            '展示内容已更新，请点击“生成替换效果”。' if name in contents else '请上传要放进屏幕的图片。')

                def pick_scene(event: gr.SelectData):
                    name = list(scenes)[event.index]
                    return (name, *choose_scene(name))

                def pick_content(event: gr.SelectData):
                    name = list(contents)[event.index]
                    return (name, *choose_content(name))

                scene_outputs = [bg_img, coordinates, preview, output_image, reference, matrix, status]
                content_outputs = [fg_img, output_image, reference, matrix, status]
                scene_choice.input(choose_scene, inputs=[scene_choice], outputs=scene_outputs, api_name='choose_screen_scene', **image_events)
                content_choice.input(choose_content, inputs=[content_choice], outputs=content_outputs, api_name='choose_screen_content', **image_events)
                scene_gallery.select(pick_scene, outputs=[scene_choice]+scene_outputs, api_name=False, **image_events)
                content_gallery.select(pick_content, outputs=[content_choice]+content_outputs, api_name=False, **image_events)
                bg_img.select(click_point, inputs=[bg_img, coordinates], outputs=[coordinates, preview], api_name=False, **image_events)
                coordinates.change(mark_points, inputs=[bg_img, coordinates], outputs=[preview], api_name=False, **image_events)
                bg_img.upload(lambda image: (custom_scene, empty_points, image, None, None, None, '请在底图中依次点击屏幕四角。'),
                              inputs=[bg_img], outputs=[scene_choice, coordinates, preview, output_image, reference, matrix, status], api_name=False, **image_events)
                bg_img.clear(lambda: (custom_scene, empty_points, None, None, None, None, '请选择或上传底图。'),
                             outputs=[scene_choice, coordinates, preview, output_image, reference, matrix, status], api_name=False, **image_events)
                fg_img.upload(lambda: (custom_content, None, None, None, '展示内容已更新，请重新生成效果。'),
                              outputs=[content_choice, output_image, reference, matrix, status], api_name=False, **image_events)
                fg_img.clear(lambda: (custom_content, None, None, None, '请选择或上传展示图。'),
                             outputs=[content_choice, output_image, reference, matrix, status], api_name=False, **image_events)
                reset_btn.click(lambda image: (empty_points, image), inputs=[bg_img],
                                outputs=[coordinates, preview], api_name=False, **image_events)
                run_btn.click(render_image, inputs=[bg_img, fg_img, mode, coordinates, backend],
                              outputs=[output_image, reference, matrix, status], api_name='replace_hw3_image', **image_events)

            with gr.Tab('视频'):
                gr.Markdown('选择视频和展示内容，屏幕范围会随画面自动跟踪。内置视频已选好首帧四角；输出视频不含音轨。'
                            '生成期间切换素材会在当前处理完成后生效。')
                gr.Markdown('### 1. 选择底图视频和展示图')
                with gr.Row():
                    with gr.Column():
                        video_choice = gr.Dropdown(([video_name] if video_file.is_file() else [])+[custom_video],
                                                   value=video_name if video_file.is_file() else custom_video, label='选择视频场景')
                        video = gr.Video(value=str(video_file) if video_file.is_file() else None,
                                         sources=['upload'], label='当前视频（也可上传）', format='mp4', height=300)
                    with gr.Column():
                        video_content_choice = gr.Dropdown(list(contents)+[custom_content], value=default_content, label='选择视频展示内容')
                        video_content_gallery = gr.Gallery([(p, name) for name, p in contents.items()],
                                                           label='点击选择视频展示图', columns=4, rows=1, height=145,
                                                           object_fit='contain', allow_preview=False, interactive=False)
                        video_fg = gr.Image(value=initial_fg, type='numpy', image_mode='RGB', label='当前视频展示图（也可上传）',
                                            format='png', height=300)
                gr.Markdown('### 2. 在首帧选择屏幕四角')
                with gr.Row():
                    with gr.Column():
                        first_frame = gr.Image(value=initial_frame, type='numpy', image_mode='RGB', format='png',
                                               label='首帧：依次点击四个角点', interactive=False, height=280)
                    with gr.Column(scale=1, min_width=220):
                        video_points = gr.Dataframe(value=video_vertices if initial_frame is not None else empty_points, headers=['x', 'y'],
                                                    datatype=['number', 'number'], type='numpy',
                                                    row_count=(4, 'fixed'), col_count=(2, 'fixed'),
                                                    label='首帧四角（依次为①②③④）', interactive=True)
                        reset_video = gr.Button('清空视频角点')
                    with gr.Column():
                        video_preview = gr.Image(value=mark_points(initial_frame, video_vertices), type='numpy', format='png', label='首帧屏幕范围',
                                                 interactive=False, height=280)
                with gr.Row():
                    video_mode = gr.Radio([('快速 · 最近邻', 'Nearest'), ('平滑 · 双线性', 'Bilinear'),
                                           ('细腻 · 双三次', 'Bicubic')], value='Bilinear', label='视频画面采样')
                    video_backend = gr.Radio([('精细处理', '手动实现'), ('快速处理', 'OpenCV')], value='OpenCV', label='视频处理方式')
                run_video = gr.Button('生成替换视频', variant='primary')
                gr.Markdown('### 3. 查看视频效果')
                output_video = gr.Video(label='替换后视频', format='mp4', interactive=False, height=480)
                video_status = gr.Markdown('内置视频已准备就绪，选择展示内容后即可生成。' if initial_frame is not None else '请选择或上传视频。')
                with gr.Accordion('下载屏幕定位信息', open=False):
                    tracking_file = gr.File(label='屏幕定位数据', interactive=False)

                def load_video(path):
                    frame = process(None, None, 'Nearest', video_path=path, action='preview_video') if path else None
                    return frame, empty_points, frame, None, None, '请指定新视频首帧的四个角点。'

                def choose_video(name):
                    if name == video_name and video_file.is_file():
                        return (str(video_file), initial_frame, video_vertices,
                                mark_points(initial_frame, video_vertices), None, None, '视频与屏幕范围已就绪，可直接生成。')
                    return None, None, empty_points, None, None, None, '请上传视频，然后选择首帧屏幕四角。'

                def choose_video_content(name):
                    return ((read_rgb(contents[name]) if name in contents else None), None, None,
                            '展示内容已更新，请重新生成视频。' if name in contents else '请上传要放进屏幕的图片。')

                def pick_video_content(event: gr.SelectData):
                    name = list(contents)[event.index]
                    return (name, *choose_video_content(name))

                video_choice.input(choose_video, inputs=[video_choice],
                                   outputs=[video, first_frame, video_points, video_preview, output_video, tracking_file, video_status],
                                   api_name='choose_screen_video', **video_events)
                video_content_choice.input(choose_video_content, inputs=[video_content_choice],
                                           outputs=[video_fg, output_video, tracking_file, video_status], api_name='choose_video_content', **video_events)
                video_content_gallery.select(pick_video_content,
                                             outputs=[video_content_choice, video_fg, output_video, tracking_file, video_status], api_name=False, **video_events)
                video.upload(lambda: custom_video, outputs=[video_choice], api_name=False, **video_events)
                video_fg.upload(lambda: (custom_content, None, None, '展示内容已更新，请重新生成视频。'),
                                outputs=[video_content_choice, output_video, tracking_file, video_status], api_name=False, **video_events)
                video_fg.clear(lambda: (custom_content, None, None, '请选择或上传展示图。'),
                               outputs=[video_content_choice, output_video, tracking_file, video_status], api_name=False, **video_events)

                video.upload(load_video, inputs=[video],
                             outputs=[first_frame, video_points, video_preview, output_video,
                                      tracking_file, video_status], api_name='preview_hw3_video', **video_events)
                video.clear(lambda: (custom_video, *load_video(None)),
                            outputs=[video_choice, first_frame, video_points, video_preview, output_video,
                                     tracking_file, video_status], api_name=False, **video_events)
                first_frame.select(click_point, inputs=[first_frame, video_points],
                                   outputs=[video_points, video_preview], api_name=False, **video_events)
                video_points.change(mark_points, inputs=[first_frame, video_points],
                                    outputs=[video_preview], api_name=False, **video_events)
                reset_video.click(lambda image: (empty_points, image), inputs=[first_frame],
                                  outputs=[video_points, video_preview], api_name=False, **video_events)

                def render_video(path, foreground, method, vertices, selected_backend):
                    path, tracking, _ = process(None, foreground, method, points=vertices,
                                                implementation=selected_backend, video_path=path, action='video')
                    return path, tracking, '视频已生成，可以预览或下载。输出视频不含音轨。'

                run_video.click(render_video, inputs=[video, video_fg, video_mode, video_points, video_backend],
                                outputs=[output_video, tracking_file, video_status], api_name='replace_hw3_video',
                                **video_events)
    return demo

def create_demo_hw4(process):
    sharpen_mode, denoise_mode = '锐化对比', 'BM3D 去噪'
    image_labels = {
        sharpen_mode: ('原图', '导向滤波锐化', '双边滤波锐化', '快速导向滤波锐化'),
        denoise_mode: ('待去噪图像', '双边滤波去噪', 'BM3D 初步估计', 'BM3D 最终结果'),
    }

    def change_mode(mode):
        sharpen = mode == sharpen_mode
        hint = (
            '用相同的锐化强度对比三种滤波。快速导向滤波使用积分图计算窗口均值。'
            if sharpen else
            'BM3D（Block-Matching and 3D Filtering，块匹配三维滤波）先按相似图块做初步去噪，'
            '再用初步结果估计应保留的细节。'
            '上传干净图像并勾选模拟加噪，可对照原图计算峰值信噪比（PSNR）。大图处理时间较长。'
        )
        return gr.update(visible=sharpen), gr.update(visible=not sharpen), hint

    def select_focus(image, event: gr.SelectData):
        if image is None or not isinstance(event.index, (tuple, list)) or len(event.index) < 2:
            return gr.update(), gr.update()
        height, width = image.shape[:2]
        x, y = event.index[:2]
        percent_x = 100 * x / max(width - 1, 1)
        percent_y = 100 * y / max(height - 1, 1)
        return percent_x, percent_y

    def run(image, mode, radius, epsilon, amount, sigma_space, sigma_range,
            noise_sigma, add_noise, seed, focus_x, focus_y, crop_size):
        comparison, first, second, third, detail, report = process(
            image, mode, radius, epsilon, amount, sigma_space, sigma_range,
            noise_sigma, add_noise, seed, focus_x, focus_y, crop_size, return_details=True
        )
        labeled = tuple(gr.update(label=label, value=result) for label, result in zip(
            image_labels[mode], (comparison, first, second, third)
        ))
        return (*labeled, detail, report)

    with gr.Blocks() as demo:
        gr.Markdown('## 作业四 · 图像锐化与去噪')
        mode = gr.Radio([sharpen_mode, denoise_mode], value=sharpen_mode, label='实验模式')
        mode_hint = gr.Markdown(change_mode(sharpen_mode)[2])
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                input_image = gr.Image(
                    sources=['upload', 'webcam', 'clipboard'], type='numpy', image_mode='RGB',
                    label='输入图像（点击选择局部中心）', height=270,
                )
                with gr.Group() as sharpen_controls:
                    radius = gr.Slider(1, 12, value=4, step=1, label='导向滤波窗口半径（像素）')
                    epsilon = gr.Slider(0.0001, 0.1, value=0.01, step=0.0001,
                                        label='导向滤波正则化强度 ε',
                                        info='增大时平滑更强，像素按 0–1 计算。')
                    amount = gr.Slider(0, 4, value=1.5, step=0.1, label='锐化强度',
                                       info='将原图与平滑图的差值乘以该强度，再加回原图。')
                with gr.Group(visible=False) as denoise_controls:
                    add_noise = gr.Checkbox(value=True, label='模拟添加高斯噪声')
                    noise_sigma = gr.Slider(0, 75, value=25, step=1, label='噪声标准差（0–255）',
                                            info='关闭模拟加噪时，填入对输入图像噪声强度的估计。')
                    seed = gr.Number(value=42, precision=0, minimum=0, maximum=2**32 - 1,
                                     label='随机种子', info='同一图片、噪声强度和种子可复现相同噪声。')
                with gr.Accordion('双边滤波参数', open=False):
                    sigma_space = gr.Slider(0.5, 5, value=2, step=0.5, label='空间标准差（像素）')
                    sigma_range = gr.Slider(0.01, 0.5, value=0.10, step=0.01,
                                            label='像素差标准差（0–1）')
                with gr.Accordion('局部对比位置', open=False):
                    focus_x = gr.Slider(0, 100, value=50, step=0.01, label='中心横坐标（%）')
                    focus_y = gr.Slider(0, 100, value=50, step=0.01, label='中心纵坐标（%）')
                    crop_size = gr.Slider(32, 256, value=96, step=16, label='裁剪边长（像素）')
                run_button = gr.Button('运行对比', variant='primary')
            with gr.Column(scale=2, min_width=320):
                with gr.Row():
                    comparison_image = gr.Image(label=image_labels[sharpen_mode][0],
                                                interactive=False, format='png', height=245)
                    first_image = gr.Image(label=image_labels[sharpen_mode][1],
                                           interactive=False, format='png', height=245)
                with gr.Row():
                    second_image = gr.Image(label=image_labels[sharpen_mode][2],
                                            interactive=False, format='png', height=245)
                    third_image = gr.Image(label=image_labels[sharpen_mode][3],
                                           interactive=False, format='png', height=245)
                detail_image = gr.Image(label='同一位置的局部对比（顺序与上方结果一致）',
                                        interactive=False, format='png', height=230)
                report = gr.Markdown()

        mode.change(change_mode, inputs=[mode], outputs=[sharpen_controls, denoise_controls, mode_hint],
                    queue=False, api_name=False)
        input_image.select(select_focus, inputs=[input_image], outputs=[focus_x, focus_y],
                           queue=False, api_name=False)
        run_button.click(
            run, inputs=[input_image, mode, radius, epsilon, amount, sigma_space, sigma_range,
                         noise_sigma, add_noise, seed, focus_x, focus_y, crop_size],
            outputs=[comparison_image, first_image, second_image, third_image, detail_image, report],
            api_name='process_hw4',
        )
    return demo


def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown("## 作业五：直方图均衡化与匹配")
        gr.Markdown("均衡化增强图像对比度。匹配按参考图的明暗分布调整输入图。")
        mode = gr.Radio(["直方图均衡化", "直方图匹配"], value="直方图均衡化", label="处理方式")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["upload", "webcam", "clipboard"],
                                       type="numpy", format="png", label="输入图像")
                reference_image = gr.Image(sources=["upload", "clipboard"],
                                           type="numpy", format="png", label="参考图像",
                                           visible=False)
                run_button = gr.Button("运行", variant="primary")
            output_image = gr.Image(type="numpy", format="png", label="输出图像", interactive=False)

        mode.change(lambda selected: gr.update(visible=selected == "直方图匹配"),
                    inputs=mode, outputs=reference_image, api_name=False)
        run_button.click(fn=process, inputs=[input_image, mode, reference_image],
                         outputs=output_image, api_name="process_hw5")
    return demo