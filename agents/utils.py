import pdfkit
import markdown
import argparse
import signal


def save_to_pdf(report_markdown, pdf_file_name="competitor_analysis_report.pdf", input_query=None):
    if input_query:
        pdf_file_name = f"{input_query.replace(' ', '_')}_{pdf_file_name}"

    report_markdown = preprocess_markdown(report_markdown)

    report_html = markdown.markdown(report_markdown)

    options = {
        'page-size': 'A4',
        'encoding': "UTF-8",
    }

    pdfkit.from_string(report_html, pdf_file_name, options=options)
    print(f"Report saved as {pdf_file_name}")


def preprocess_markdown(markdown_text):
    lines = markdown_text.splitlines()
    processed_lines = []

    for line in lines:
        # Ensure proper spacing before list items
        if line.strip().startswith("-") and not line.startswith(" "):
            processed_lines.append("\n" + line)  # Add a blank line before the list item
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--query', help='startup website name or prodcut name', dest='query', default='llm agent')
    # program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    # program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--device', help='machine type (choices: cpu, cuda, ...)', dest='device', default="cuda")
    program.add_argument('--model-name', help='vllm model name', dest='model_name', action='store_true', default="Qwen/Qwen2.5-0.5B-Instruct")

    args = program.parse_args()
    return args
