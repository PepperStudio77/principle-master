import datetime
import os

class Metric(object):
    def __init__(self, output_token, input_token, total_token, time_taken):
        self.output_token = output_token
        self.input_token = input_token
        self.total_token = total_token
        self.time_taken = time_taken

class Response(object):
    def __init__(self, message, metric):
        self.message = message
        self.metric = metric

def metric_to_md(metric):
    content = (f"## Statistic \n\n" +
               f"- Input Token: {metric.input_token} \n" +
               f"- Output Token: {metric.output_token}\n" +
               f"- Total Token: {metric.total_token}\n" +
               f"- Time Taken: {str(metric.time_taken)} \n\n")

    return content


def write_response_to_mark_down(query, response, query_mode):
    log_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "output")  # Change this to your actual PDF file path
    log_time = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    file_name = os.path.join(log_directory, f"output_{datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')}.md")
    header = f"# [{query_mode}] {query}@{log_time}\n\n"
    if response.metric is not None:
        header += metric_to_md(response.metric)
    response = header + f"# Plan \n\n {response.message}"
    with open(file_name, "w+") as f:
        f.write(response)
