from copy import deepcopy


_LATEST_REPORT = {}


def store_latest_report(report_payload):
    _LATEST_REPORT.clear()
    _LATEST_REPORT.update(deepcopy(report_payload))


def get_latest_report():
    return deepcopy(_LATEST_REPORT)
