import gradio as gr
import pandas as pd

from pipeline import analyze_shelf_image


def run_analysis(image, shelf_id):
    if image is None:
        return None, "Please upload a shelf image first.", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    try:
        result = analyze_shelf_image(image, shelf_id)

        annotated_image = result["annotated_image"]
        summary_text = result["summary_text"]
        product_table = pd.DataFrame(result["products"])
        row_summary = pd.DataFrame(result["row_summary"])
        planogram_table = pd.DataFrame(result["planogram_comparison"])

        return annotated_image, summary_text, product_table, row_summary, planogram_table

    except Exception as e:
        error_message = f"Error while running pipeline:\n{str(e)}"
        return image, error_message, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


with gr.Blocks(title="PLANOGRAM") as demo:
    gr.Markdown("# PLANOGRAM")
    gr.Markdown(
        "Upload your shelf image. The system detects products, assigns shelf rows, "
        "retrieves SKU candidates using FAISS, and returns an annotated shelf report."
    )

    with gr.Row():
        image_input = gr.Image(label="Shelf Image", type="numpy")
        shelf_input = gr.Dropdown(
            choices=["Shelf A", "Shelf B", "Shelf C"],
            value="Shelf A",
            label="Shelf Name"
        )

    run_button = gr.Button("Run Analysis")

    annotated_output = gr.Image(label="AI Output Image")
    summary_output = gr.Textbox(label="Summary", lines=7)

    product_table_output = gr.Dataframe(label="Detected Product Table")
    row_summary_output = gr.Dataframe(label="Row Summary")
    planogram_output = gr.Dataframe(label="Planogram Comparison")

    run_button.click(
        fn=run_analysis,
        inputs=[image_input, shelf_input],
        outputs=[
            annotated_output,
            summary_output,
            product_table_output,
            row_summary_output,
            planogram_output
        ]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)