import time, json, pathlib


# Log
_LOG_PATH = pathlib.Path("run_log.jsonl").resolve()
def write_agent_log(agent, input_data, output_data):
    log_record = {
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "agent": agent,
        "input": input_data,
        "output": output_data
    }
    with _LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_record, ensure_ascii=False) + "\n")
        
