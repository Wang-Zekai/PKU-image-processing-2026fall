import gradio as gr

from functions import EQUALIZATION, MATCHING


def create_demo_hw5(process):
    with gr.Blocks(title="Image Processing Homework 5") as demo:
        gr.Markdown("## Histogram Equalization and Matching")
        gr.Markdown(
            "Equalization increases contrast. Matching adjusts the input image "
            "toward the reference image's lightness distribution."
        )
        mode = gr.Radio([EQUALIZATION, MATCHING], value=EQUALIZATION, label="Method")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=["upload", "webcam", "clipboard"],
                                       type="numpy", format="png", label="Input image")
                reference_image = gr.Image(sources=["upload", "clipboard"],
                                           type="numpy", format="png", label="Reference image",
                                           visible=False)
                run_button = gr.Button("Run", variant="primary")
            output_image = gr.Image(type="numpy", format="png", label="Output image", interactive=False)

        mode.change(lambda selected: gr.update(visible=selected == MATCHING),
                    inputs=mode, outputs=reference_image, api_name=False)
        run_button.click(fn=process, inputs=[input_image, mode, reference_image],
                         outputs=output_image, api_name="process_hw5")
    return demo
